"""Localhost web app: upload → pipeline → progress → review → re-frame → download.

Single-user, local-first. The pipeline (transcribe → select → render) runs in a
background thread; each stage writes its status into `data/<job_id>/job.json`, so
the browser just polls `/api/job/{id}`. The library is the filesystem itself —
`data/*/job.json` — no database needed for one local user (ponytail: a directory
glob is the index; the architecture research endorsed this).
"""

from __future__ import annotations

import json
import os
import shutil
import threading
import traceback
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Body, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from jinja2 import Environment, FileSystemLoader, select_autoescape

from . import captions, config, render, select, transcribe, vision
from .jobs import STAGES, Job, compute_job_id, read_json
from .logging_setup import get_logger, job_log, job_log_path, tail

log = get_logger(__name__)

config.DATA_DIR.mkdir(parents=True, exist_ok=True)
UPLOADS = config.DATA_DIR / "_uploads"
UPLOADS.mkdir(parents=True, exist_ok=True)
VIDEO_EXTS = {".mp4", ".mov", ".m4v", ".mkv", ".webm", ".avi"}


def new_job_id(content_id: str) -> str:
    """A unique job id per upload: content-hash prefix (for grouping) + a nonce,
    so re-uploading the same file is a fresh run, not the previous one."""
    import uuid
    return f"{content_id[:10]}{uuid.uuid4().hex[:6]}"


def safe_upload_name(filename: str) -> str:
    """Basename-only, dotfile-free, video-extension filename. Raises ValueError otherwise."""
    name = Path(filename or "").name
    if not name or name.startswith(".") or Path(name).suffix.lower() not in VIDEO_EXTS:
        raise ValueError(f"unsafe or unsupported filename: {filename!r}")
    if not (UPLOADS / name).resolve().is_relative_to(UPLOADS.resolve()):
        raise ValueError("filename escapes uploads dir")
    return name

# Drive Jinja2 directly with caching disabled — Starlette's Jinja2Templates hits
# a jinja2 LRUCache bug on Python 3.14 (unhashable dict in the cache key).
# ponytail: cache_size=0 sidesteps the broken cache path; templates are tiny.
_env = Environment(
    loader=FileSystemLoader(str(Path(__file__).parent / "templates")),
    autoescape=select_autoescape(["html"]), cache_size=0,
)


def render_template(name: str, **ctx) -> HTMLResponse:
    return HTMLResponse(_env.get_template(name).render(**ctx))

@asynccontextmanager
async def _lifespan(_app):
    """Graceful shutdown (REL-04): on stop, signal background pipeline/re-render
    workers to stop draining. They're daemons so the process still exits promptly;
    in-flight ffmpeg children are terminated by process exit."""
    yield
    _SHUTTING_DOWN.set()
    log.info("shutdown: signalled background workers to stop")


# VAL-05: the data dir holds media AND job metadata (job.json / transcript.json /
# logs). Serving the whole tree leaks those. MediaFiles serves only an allowlist of
# media extensions and 404s everything else — the UI only ever loads mp4 + jpg/png.
MEDIA_EXTS = {".mp4", ".mov", ".webm", ".m4v", ".jpg", ".jpeg", ".png", ".gif"}


class MediaFiles(StaticFiles):
    """StaticFiles that only serves media files (VAL-05); 404s non-media paths."""

    async def get_response(self, path, scope):
        if Path(path).suffix.lower() not in MEDIA_EXTS:
            raise HTTPException(404)
        return await super().get_response(path, scope)


app = FastAPI(title="Video Transcription and Rendering Pipeline", lifespan=_lifespan)
app.mount("/media", MediaFiles(directory=str(config.DATA_DIR)), name="media")
app.mount("/static", StaticFiles(directory=str(Path(__file__).parent / "static")), name="static")

# job_id -> {"error": str|None} for surfacing background failures
RUNNING: dict[str, dict] = {}

# REL-05: a global render-slot semaphore bounds concurrent encodes so multiple
# simultaneous jobs (or a job render racing a clip re-render) can't exhaust a
# consumer GPU's 2–3 NVENC session limit. Default 1 (fully serial) is the safest
# "must not break" choice on a single GPU; override with CM_MAX_RENDERS.
try:
    _MAX_RENDERS = max(1, int(os.environ.get("CM_MAX_RENDERS", "1")))
except ValueError:
    _MAX_RENDERS = 1
_RENDER_SLOTS = threading.Semaphore(_MAX_RENDERS)

# Flipped by the lifespan shutdown so background workers can stop cleanly (REL-04).
_SHUTTING_DOWN = threading.Event()


def media_url(path: str | Path) -> str:
    """Stable /media URL for a data-dir file, cache-busted by the file's mtime.

    FE-08: appending ``?v=<int mtime>`` makes the URL change whenever the file
    content changes, so reconcile-rebuilt clip cards and editor previews bust
    automatically (no manual ``?t=`` needed). A missing/unstatable file omits the
    query rather than failing — the bare /media path still works."""
    p = Path(path)
    rel = p.resolve().relative_to(config.DATA_DIR.resolve())
    url = f"/media/{rel.as_posix()}"
    try:
        return f"{url}?v={int(p.stat().st_mtime)}"
    except OSError:
        return url


# --- background clip re-render (non-blocking + queued, per clip) --------------
# (job_id, idx) -> tracker dict. One worker thread per clip drains a single-slot
# queue, so an edit made while a render runs is applied *after* it — never
# blocked, never raced (one ffmpeg writing a clip's outputs at a time). Mirrors
# the main pipeline's in-process daemon-thread model; in-memory is fine for one
# local user. Tracker shape:
#   {status: queued|rendering|done|error, aspects: {a: queued|rendering|done|error},
#    outputs: {a: media_url}, result: payload|None, error: str|None,
#    pending: {edit, aspects}|None, thread: Thread|None}
_RERENDER: dict[tuple[str, int], dict] = {}
_RERENDER_LOCK = threading.Lock()


def _rerender_payload(result: dict) -> dict:
    return {
        "index": result.get("index"),
        "start": result.get("start"),
        "end": result.get("end"),
        "audio": result.get("audio"),
        "transforms": result.get("transforms"),
        "outputs": {a: media_url(p) for a, p in result.get("outputs", {}).items()},
    }


def _rerender_worker(job_id: str, idx: int) -> None:
    key = (job_id, idx)
    while True:
        if _SHUTTING_DOWN.is_set():  # REL-04: stop draining on shutdown
            return
        with _RERENDER_LOCK:
            tr = _RERENDER.get(key)
            req = tr.get("pending") if tr else None
            if not tr or not req:
                if tr:                                   # queue drained — settle
                    tr["thread"] = None
                    if tr.get("status") == "rendering":
                        tr["status"] = "done"
                return
            tr["pending"] = None
            tr["status"] = "rendering"
            tr["error"] = None
            req_aspects = tuple(req.get("aspects") or config.ASPECT_RATIOS)
            # aspects encode serially (one NVENC engine): first is rendering, rest
            # queued — the completion callback advances the next one.
            for i, a in enumerate(req_aspects):
                tr["aspects"][a] = "rendering" if i == 0 else "queued"

        def cb(aspect, path):                            # per-aspect completion
            with _RERENDER_LOCK:
                tr["aspects"][aspect] = "done"
                tr["outputs"][aspect] = media_url(path)
                for a in req_aspects:                    # promote the next pending ratio
                    if tr["aspects"].get(a) == "queued":
                        tr["aspects"][a] = "rendering"
                        break

        try:
            with _RENDER_SLOTS:  # REL-05: share the global render-slot cap
                result = render.rerender_one(
                    job_id, idx, edit=req.get("edit"),
                    aspects=req_aspects or None, on_aspect_done=cb)
            with _RERENDER_LOCK:
                payload = _rerender_payload(result)
                tr["result"] = payload
                tr["outputs"].update(payload["outputs"])
                for a in req_aspects:
                    tr["aspects"][a] = "done"
        except Exception as e:                           # surface, keep draining
            log.error("re-render %s/clip%02d failed: %s\n%s",
                      job_id, idx, e, traceback.format_exc())
            with _RERENDER_LOCK:
                tr["status"] = "error"
                tr["error"] = str(e)
                for a in req_aspects:
                    if tr["aspects"].get(a) == "rendering":
                        tr["aspects"][a] = "error"


def _enqueue_rerender(job_id: str, idx: int, edit: dict, aspects: list[str]) -> None:
    """Queue a clip re-render and ensure a worker is draining it. Non-blocking."""
    key = (job_id, idx)
    with _RERENDER_LOCK:
        tr = _RERENDER.get(key)
        alive = bool(tr and tr.get("thread") and tr["thread"].is_alive())
        if not alive:                                    # fresh tracker, keep last outputs
            tr = {"status": "queued", "aspects": {},
                  "outputs": dict((tr or {}).get("outputs") or {}),
                  "result": (tr or {}).get("result"),
                  "error": None, "pending": None, "thread": None}
            _RERENDER[key] = tr
        tr["pending"] = {"edit": edit, "aspects": list(aspects)}
        tr["status"] = "queued" if not alive else "rendering"
        tr["error"] = None
        for a in aspects:
            if tr["aspects"].get(a) != "rendering":
                tr["aspects"][a] = "queued"
        if not alive:
            t = threading.Thread(target=_rerender_worker, args=(job_id, idx),
                                 daemon=True)
            tr["thread"] = t
            t.start()


def _run_pipeline(job_id: str, source: Path, model: str | None, max_clips: int,
                  x_offset: float, captions_mode: str,
                  transforms: dict | None = None) -> None:
    job = Job.load(job_id)
    RUNNING[job.job_id] = {"error": None}
    with job_log(job.job_id):
        log.info("=== pipeline start: %s (model=%s, max_clips=%d, captions=%s) ===",
                 job.job_id, model or config.DEFAULT_MODEL, max_clips, captions_mode)
        try:
            import json as _json
            no_moments_msg = ("No clip-worthy moments found. Try a longer video, lower "
                              "the minimum clip length, or raise 'max clips'.")
            transcribe.transcribe(source, model=model, job=job)
            # VAL-04: an empty/silent transcript would make select raise a raw
            # ValueError("Transcript has no segments…"). Surface the same friendly
            # "no moments" message on the select stage instead, mirroring 0-clips.
            transcript = _json.loads(job.transcript_path.read_text()) \
                if job.transcript_path.exists() else {}
            if not transcript.get("segments"):
                log.warning(no_moments_msg)
                RUNNING[job.job_id]["error"] = no_moments_msg
                job.update_stage("select", "error", error=no_moments_msg)
                return
            clips_path = select.select_clips(job, max_clips=max_clips)
            n_clips = len(_json.loads(Path(clips_path).read_text()).get("clips", []))
            if n_clips == 0:
                log.warning(no_moments_msg)
                RUNNING[job.job_id]["error"] = no_moments_msg
                job.update_stage("render", "error", error=no_moments_msg)
                return
            with _RENDER_SLOTS:  # REL-05: bound concurrent GPU encodes
                render.render_job(job, x_offset=x_offset, caption_mode=captions_mode,
                                  transforms=transforms)
            log.info("=== pipeline complete: %s ===", job.job_id)
        except Exception as e:  # surface failure on whichever stage was running
            RUNNING[job.job_id]["error"] = str(e)
            log.error("=== pipeline FAILED: %s ===\n%s", e, traceback.format_exc())
            manifest = job.load_manifest()
            marked = False
            for stage in STAGES:
                if manifest.get("stages", {}).get(stage, {}).get("status") == "running":
                    job.update_stage(stage, "error", error=str(e))
                    marked = True
            if not marked:  # failed before any stage started running
                job.update_stage("ingest", "error", error=str(e))


def list_jobs() -> list[dict]:
    jobs = []
    for mf in sorted(config.DATA_DIR.glob("*/job.json")):
        try:
            m = json.loads(mf.read_text())
        except Exception:
            continue
        stages = m.get("stages", {})
        done = stages.get("render", {}).get("status") == "done"
        if done:
            status = "ready"
        elif m.get("awaiting_run"):
            status = "awaiting run"
        else:
            status = _overall_status(stages)
        jobs.append({
            "job_id": m.get("job_id", mf.parent.name),
            "source_name": m.get("source_name", ""),
            "created_at": m.get("created_at", 0),
            "status": status,
            "n_clips": stages.get("render", {}).get("clips")
                       or stages.get("select", {}).get("clips"),
        })
    return sorted(jobs, key=lambda j: j["created_at"], reverse=True)


def _overall_status(stages: dict) -> str:
    for s in STAGES:
        st = stages.get(s, {}).get("status", "pending")
        if st == "error":
            return f"error: {s}"
        if st == "running":
            return f"{s}…"
        if st in ("pending",) and s != "ingest":
            return f"{s} pending"
    return "working…"


# Master-bar weights — render is the long pole, transcribe second; ingest/select
# are quick. Sum to 1.0 so the master bar reads as honest overall completion.
_STAGE_WEIGHTS = {"ingest": 0.02, "transcribe": 0.40, "select": 0.08, "render": 0.50}


def _stage_fraction(entry: dict) -> float:
    st = entry.get("status")
    if st == "done":
        return 1.0
    if st in ("running", "error"):
        return max(0.0, min(1.0, float(entry.get("progress") or 0.0)))
    return 0.0


def _master_progress(stages: dict) -> float:
    return round(sum(_STAGE_WEIGHTS.get(k, 0.0) * _stage_fraction(v)
                     for k, v in stages.items()), 4)


def _job_payload(job: Job) -> dict:
    m = job.load_manifest()
    stages = {s: m.get("stages", {}).get(s, {"status": "pending"}) for s in STAGES}
    clips = []
    render_manifest = job.clips_dir / "render.json"
    if render_manifest.exists():
        for c in read_json(render_manifest, default={"clips": []}).get("clips", []):
            clips.append({
                "index": c["index"],
                "title": c.get("title", ""),
                "score": c.get("score"),
                "captions": c.get("captions"),
                "thumb": media_url(c["thumb"]) if c.get("thumb") else None,
                "outputs": {a: media_url(p) for a, p in c.get("outputs", {}).items()},
                "transforms": c.get("transforms", {}),
            })
    # rationale + timing live in clips.json
    if job.clips_json_path.exists():
        src_clips = json.loads(job.clips_json_path.read_text()).get("clips", [])
        meta = {i + 1: c for i, c in enumerate(src_clips)}
        for c in clips:
            sc = meta.get(c["index"], {})
            c["rationale"] = sc.get("rationale", "")
            c["start"] = sc.get("start", 0)
            c["end"] = sc.get("end", 0)
    source = next(job.data_dir.glob("source.*"), None)
    # Surface the failure message from the in-memory RUNNING tracker if present,
    # else fall back to whichever stage recorded an error on disk — so an in-flight
    # failure still shows after a server restart (REL-04), not just this process.
    err = RUNNING.get(job.job_id, {}).get("error")
    if not err:
        err = next((st.get("error") for st in stages.values()
                    if st.get("status") == "error" and st.get("error")), None)
    return {"job_id": job.job_id, "source_name": m.get("source_name", ""),
            "stages": stages, "clips": clips,
            "progress": _master_progress(stages),
            "awaiting_run": bool(m.get("awaiting_run")),
            "source_url": media_url(source) if source else None,
            "source_dims": m.get("source_dims", [0, 0]),
            "run_params": m.get("run_params", {}),
            "error": err}


# --- routes ------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def index():
    return render_template("index.html", jobs=list_jobs())


@app.post("/upload")
def upload(video: UploadFile = File(...)):
    """Stage the upload for preview — does NOT start the pipeline.

    The source is moved into data/<job_id>/source<ext> (previewable via /media),
    its dimensions are probed for the crop overlay, and the manifest is marked
    `awaiting_run`. The user picks the crop + run options on the job page, then
    POSTs /api/job/{id}/run to actually start transcribe→select→render.

    Defined `def` (not `async def`) on purpose (REL-03): the body does synchronous,
    potentially multi-second work — streaming the upload to disk and SHA-256ing the
    whole file for the content id. A sync endpoint runs in Starlette's threadpool, so
    a large upload no longer blocks the event loop (and every in-flight progress poll).
    """
    if not video.filename:
        raise HTTPException(400, "No file")
    try:
        safe_name = safe_upload_name(video.filename)  # blocks path traversal
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    # VAL-03: stage under a unique nonce so two concurrent uploads of the same
    # filename can't clobber each other's temp file. The final src path is derived
    # from the content-hash job id below, so the nonce only scopes the staging file.
    tmp = UPLOADS / f"{uuid.uuid4().hex}_{safe_name}"
    with open(tmp, "wb") as f:
        shutil.copyfileobj(video.file, f)
    # Each upload is its OWN run, even for an identical file: the job id is the
    # content hash plus a per-upload nonce. (A pure content-hash id made a
    # re-upload land back on the previous completed run and cache-skip every
    # stage — the user wants a fresh run each time they upload.)
    import os as _os
    content_id = compute_job_id(tmp)
    job_id = new_job_id(content_id)
    ext = Path(safe_name).suffix
    job = Job(job_id=job_id, source_name=safe_name, data_dir=config.DATA_DIR / job_id)
    job.ensure_dirs()
    src = job.source_path(ext)
    # Move (not copy) the staged upload into the (brand-new) job dir; os.replace
    # is atomic on the same volume.
    _os.replace(tmp, src)
    try:
        w, h = render.probe_dims(src)
    except Exception as e:  # non-fatal — preview overlay just won't draw
        log.warning("probe_dims failed for %s: %s", src, e)
        w, h = 0, 0
    log.info("upload staged: %s (%.1f MB, %dx%d) -> job %s — awaiting run",
             safe_name, src.stat().st_size / 1e6, w, h, job_id)
    manifest = job.load_manifest()
    manifest["awaiting_run"] = True
    manifest["source_ext"] = ext
    manifest["source_dims"] = [w, h]
    manifest["content_id"] = content_id  # source identity, for reference/grouping
    # Persist the metadata FIRST, then mark ingest done. The previous order
    # (update_stage then save_manifest of the stale in-memory dict) clobbered
    # ingest back to "pending" on disk (bug #6 / Phase 25 finding).
    job.save_manifest(manifest)
    job.update_stage("ingest", "done", source=str(src))  # source already on disk
    return RedirectResponse(f"/job/{job_id}", status_code=303)


def _parse_transforms(raw: str) -> dict | None:
    """Parse the per-aspect transforms JSON from a form field; tolerate junk."""
    if not raw:
        return None
    try:
        data = json.loads(raw)
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    clean = {}
    for aspect, t in data.items():
        if aspect in config.ASPECT_RATIOS and isinstance(t, dict):
            clean[aspect] = {"zoom": t.get("zoom", 1.0), "x": t.get("x", 0.0),
                             "y": t.get("y", 0.0)}
    return clean or None


@app.post("/api/job/{job_id}/run")
def run_job(job_id: str, model: str = Form(""), max_clips: int = Form(6),
            x_offset: float = Form(0.0), captions: str = Form("overlay"),
            transforms: str = Form("")):
    """Start the pipeline for a staged job using the user-chosen framing + options."""
    if not (config.DATA_DIR / job_id / "job.json").exists():
        raise HTTPException(404, "Job not found")
    job = Job.load(job_id)
    manifest = job.load_manifest()
    if not manifest.get("awaiting_run"):
        raise HTTPException(409, "Job already started")
    src = next(job.data_dir.glob("source.*"), None)
    if src is None:
        raise HTTPException(400, "Source video missing")
    tf = _parse_transforms(transforms)
    manifest["awaiting_run"] = False
    manifest["run_params"] = {"model": model or None, "max_clips": max_clips,
                              "x_offset": x_offset, "captions": captions,
                              "transforms": tf}
    job.save_manifest(manifest)
    log.info("run requested: job %s (transforms=%s, x_offset=%.2f, max_clips=%d, captions=%s)",
             job_id, "yes" if tf else "no", x_offset, max_clips, captions)
    threading.Thread(target=_run_pipeline,
                     args=(job_id, src, model or None, max_clips, x_offset, captions, tf),
                     daemon=True).start()
    return {"ok": True}


@app.get("/job/{job_id}", response_class=HTMLResponse)
def job_page(job_id: str):
    if not (config.DATA_DIR / job_id / "job.json").exists():
        raise HTTPException(404, "Job not found")
    return render_template("job.html", job_id=job_id)


@app.get("/api/job/{job_id}")
def api_job(job_id: str):
    if not (config.DATA_DIR / job_id / "job.json").exists():
        raise HTTPException(404, "Job not found")
    return JSONResponse(_job_payload(Job.load(job_id)))


@app.get("/api/job/{job_id}/log")
def api_job_log(job_id: str, lines: int = 200):
    if not (config.DATA_DIR / job_id / "job.json").exists():
        raise HTTPException(404, "Job not found")
    return JSONResponse({"log": tail(job_log_path(job_id), lines)})


@app.post("/api/job/{job_id}/clip/{idx}/reframe")
def reframe(job_id: str, idx: int, x_offset: float = Form(0.0),
            captions: str = Form("overlay")):
    if not (config.DATA_DIR / job_id / "clips.json").exists():
        raise HTTPException(404, "Job/clips not found")
    result = render.rerender_one(job_id, idx, x_offset=x_offset, caption_mode=captions)
    return {"index": result["index"],
            "outputs": {a: media_url(p) for a, p in result["outputs"].items()}}


# --- clip editor -------------------------------------------------------------
def _clip_editor_payload(job: Job, idx: int) -> dict:
    m = job.load_manifest()
    src = next(job.data_dir.glob("source.*"), None)
    clips = json.loads(job.clips_json_path.read_text()).get("clips", []) \
        if job.clips_json_path.exists() else []
    if not 1 <= idx <= len(clips):
        raise HTTPException(404, "Clip not found")
    clip = clips[idx - 1]
    transcript = json.loads(job.transcript_path.read_text()) \
        if job.transcript_path.exists() else {"segments": [], "duration": 0}
    segs = transcript.get("segments", [])
    duration = transcript.get("duration", 0) or (segs[-1]["end"] if segs else 0)

    rentry = None
    rman = job.clips_dir / "render.json"
    if rman.exists():
        rentry = next((c for c in read_json(rman, default={"clips": []}).get("clips", [])
                       if c.get("index") == idx), None)
    edit_path = job.clips_dir / f"clip{idx:02d}" / "edit.json"
    edit = json.loads(edit_path.read_text()) if edit_path.exists() else {}

    start = float(edit.get("start", clip.get("start", 0)))
    end = float(edit.get("end", clip.get("end", 0)))
    transforms = edit.get("transforms") or (rentry or {}).get("transforms") or {}
    audio = edit.get("audio") or (rentry or {}).get("audio") or {"mute": False, "volume": 1.0}
    cap = edit.get("captions") or {}
    cap_mode = cap.get("mode", (rentry or {}).get("captions") or "overlay")
    if cap_mode not in ("overlay", "none", "karaoke"):
        cap_mode = "overlay"
    if cap.get("segments") is not None:
        cap_segs = cap["segments"]
    else:
        cap_segs = captions.clip_caption_events(segs, start, end)

    # Snap points for the trim handles (source time, unique+sorted). Prefer the
    # per-word boundaries whisper already emits in transcript.json (WORD-01) so a
    # trim can land on a word edge, not just a sentence edge; segment start/end are
    # always included as the fallback when a segment has no word timing.
    pts_set = {0.0, float(duration)}
    for s in segs:
        try:
            pts_set.add(round(float(s["start"]), 3))
            pts_set.add(round(float(s["end"]), 3))
        except (KeyError, TypeError, ValueError):
            continue
        for w in s.get("words") or []:
            try:
                pts_set.add(round(float(w["start"]), 3))
                pts_set.add(round(float(w["end"]), 3))
            except (KeyError, TypeError, ValueError):
                continue
    pts = sorted(pts_set)
    return {
        "index": idx, "title": clip.get("title", ""),
        "orig_start": clip.get("start", 0), "orig_end": clip.get("end", 0),
        "start": start, "end": end, "duration": duration,
        "source_url": media_url(src) if src else None,
        "source_dims": m.get("source_dims", [0, 0]),
        "transforms": transforms, "audio": audio,
        "captions": {"mode": cap_mode, "segments": cap_segs},
        "boundaries": pts,
        "outputs": {a: media_url(p) for a, p in (rentry or {}).get("outputs", {}).items()},
    }


@app.get("/job/{job_id}/clip/{idx}/edit", response_class=HTMLResponse)
def clip_editor_page(job_id: str, idx: int):
    if not (config.DATA_DIR / job_id / "job.json").exists():
        raise HTTPException(404, "Job not found")
    return render_template("editor.html", job_id=job_id, idx=idx)


@app.get("/api/job/{job_id}/clip/{idx}")
def api_clip_editor(job_id: str, idx: int):
    if not (config.DATA_DIR / job_id / "job.json").exists():
        raise HTTPException(404, "Job not found")
    return JSONResponse(_clip_editor_payload(Job.load(job_id), idx))


@app.post("/api/job/{job_id}/clip/{idx}/suggest-crop")
def suggest_crop(job_id: str, idx: int, payload: dict = Body(default={})):
    """Analyze the clip's current trim and propose a static 9:16 face crop."""
    if not (config.DATA_DIR / job_id / "job.json").exists():
        raise HTTPException(404, "Job not found")
    job = Job.load(job_id)
    clip = _clip_editor_payload(job, idx)  # validates the clip index
    source = next(job.data_dir.glob("source.*"), None)
    if source is None:
        raise HTTPException(404, "Source video missing")
    try:
        start = float(payload.get("start", clip["start"]))
        end = float(payload.get("end", clip["end"]))
        if end > float(clip["duration"]) + 0.01:
            raise ValueError("Clip window exceeds the source duration.")
        dims = tuple(clip["source_dims"])
        if len(dims) != 2 or min(dims) <= 0:
            dims = render.probe_dims(source)
        return vision.suggest_face_crop(
            source, start, end, job.clips_dir / f"clip{idx:02d}" / "face_suggestion.json",
            dims,
        )
    except vision.FaceSuggestionUnavailable as e:
        raise HTTPException(503, str(e)) from e
    except vision.NoFaceFound as e:
        raise HTTPException(422, str(e)) from e
    except ValueError as e:
        raise HTTPException(400, str(e)) from e


@app.get("/api/job/{job_id}/clip/{idx}/captions")
def api_clip_captions(job_id: str, idx: int, start: float, end: float):
    """FE-05: re-derive caption events for a clip from the transcript, scoped to an
    arbitrary [start, end] source-time window. The editor calls this with the CURRENT
    trim so "Re-derive from transcript" reflects a changed trim window. Same call
    ``_clip_editor_payload`` uses, just with caller-supplied bounds."""
    if not (config.DATA_DIR / job_id / "job.json").exists():
        raise HTTPException(404, "Job not found")
    job = Job.load(job_id)
    clips = json.loads(job.clips_json_path.read_text()).get("clips", []) \
        if job.clips_json_path.exists() else []
    if not 1 <= idx <= len(clips):
        raise HTTPException(404, "Clip not found")
    transcript = json.loads(job.transcript_path.read_text()) \
        if job.transcript_path.exists() else {"segments": []}
    segs = transcript.get("segments", [])
    return JSONResponse({"segments": captions.clip_caption_events(segs, start, end)})


@app.post("/api/job/{job_id}/clip/{idx}/edit")
def save_clip_edit(job_id: str, idx: int, payload: dict = Body(...)):
    if not (config.DATA_DIR / job_id / "clips.json").exists():
        raise HTTPException(404, "Job/clips not found")
    aspects = payload.get("aspects") or None
    if aspects:
        aspects = tuple(a for a in aspects if a in config.ASPECT_RATIOS) or None
    edit = {}
    if "start" in payload and "end" in payload:
        s, e = float(payload["start"]), float(payload["end"])
        if e - s < 0.5:
            raise HTTPException(400, "Clip must be at least 0.5s long")
        # VAL-01: clamp the trim to valid source bounds — start>=0, end<=duration.
        s = max(0.0, s)
        duration = None
        try:
            tj = json.loads((config.DATA_DIR / job_id / "transcript.json").read_text())
            segs = tj.get("segments", [])
            duration = tj.get("duration") or (segs[-1]["end"] if segs else None)
        except Exception:
            duration = None  # missing/unreadable transcript -> skip the upper clamp
        if duration is not None:
            e = min(e, float(duration))
        edit["start"], edit["end"] = s, e
    if isinstance(payload.get("transforms"), dict):
        edit["transforms"] = payload["transforms"]
    if isinstance(payload.get("captions"), dict):
        edit["captions"] = payload["captions"]
    if isinstance(payload.get("audio"), dict):
        edit["audio"] = payload["audio"]
    qaspects = list(aspects) if aspects else list(config.ASPECT_RATIOS)
    # Non-blocking: queue the re-render and return immediately so the editor
    # stays interactive. The editor polls /rerender-status for progress.
    _enqueue_rerender(job_id, idx, edit, qaspects)
    return {"queued": True, "index": idx, "aspects": qaspects}


@app.get("/api/job/{job_id}/clip/{idx}/rerender-status")
def rerender_status(job_id: str, idx: int):
    """Live state of a clip's background re-render (queue + per-aspect)."""
    with _RERENDER_LOCK:
        tr = _RERENDER.get((job_id, idx))
        if not tr:
            return JSONResponse({"active": False, "status": "idle", "aspects": {},
                                 "outputs": {}, "result": None, "error": None,
                                 "queued": False})
        alive = bool(tr.get("thread") and tr["thread"].is_alive())
        return JSONResponse({
            "active": alive,
            "status": tr.get("status", "idle"),
            "aspects": dict(tr.get("aspects") or {}),
            "outputs": dict(tr.get("outputs") or {}),
            "result": tr.get("result"),
            "error": tr.get("error"),
            "queued": bool(tr.get("pending")),
        })


@app.get("/download/{job_id}/{idx}/{aspect}")
def download(job_id: str, idx: int, aspect: str):
    slug = {"9:16": "9x16", "1:1": "1x1", "16:9": "16x9"}.get(aspect, aspect)
    f = config.DATA_DIR / job_id / "clips" / f"clip{idx:02d}" / f"{slug}.mp4"
    if not f.exists():
        raise HTTPException(404, "Clip not found")
    return FileResponse(f, media_type="video/mp4",
                        filename=f"{job_id}_clip{idx:02d}_{slug}.mp4")
