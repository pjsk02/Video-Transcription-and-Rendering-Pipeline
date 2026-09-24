"""Shared pytest fixtures for Video Transcription and Rendering Pipeline (v6 Phase 23).

Two things every higher-level test needs and nobody had before:

* `client` — a Starlette ``TestClient`` bound to a throwaway ``DATA_DIR``. The app
  reads ``config.DATA_DIR`` at *import* time (the ``/media`` mount + ``UPLOADS``),
  so we set ``CM_DATA_DIR`` first and reload ``config`` + ``app`` to rebind them.
* `seed_job` / `seeded_job` — write a fully-rendered job to disk (manifest, transcript,
  clips.json, clips/render.json, per-aspect outputs) so HTTP + Playwright tests can
  exercise the review grid and editor without running the multi-minute pipeline.
"""

from __future__ import annotations

import importlib
import json
import time
from pathlib import Path

import pytest

ASPECT_SLUG = {"9:16": "9x16", "1:1": "1x1", "16:9": "16x9"}

# A 44-byte near-minimal mp4 header — enough that the file *exists* and is served;
# unit/HTTP tests assert structure, not decode. Real playable media for live
# Playwright runs is produced by scripts/seed_fixture.py via ffmpeg.
_PLACEHOLDER_MP4 = bytes.fromhex(
    "0000001c66747970697336300000020069736f6d69736f3261766331"
) + b"\x00" * 16


def _reload_app(data_dir: Path):
    """Point CM_DATA_DIR at ``data_dir`` and reload config+app so module-level
    state (DATA_DIR, UPLOADS, the /media mount) rebinds to it. Returns (app, config)."""
    import os
    os.environ["CM_DATA_DIR"] = str(data_dir)
    from content_machine import config as cfg
    importlib.reload(cfg)
    from content_machine import app as appmod
    importlib.reload(appmod)
    return appmod, cfg


def seed_job(
    data_dir: Path,
    job_id: str = "seedjob0001",
    *,
    n_clips: int = 2,
    aspects=("9:16", "1:1", "16:9"),
    source_dims=(1920, 1080),
    media_bytes: bytes = _PLACEHOLDER_MP4,
    awaiting_run: bool = False,
) -> Path:
    """Write a complete, rendered job to ``data_dir/job_id`` and return its path.

    Produces job.json (all stages done), source.mp4, transcript.json (with word
    timings), clips.json, clips/render.json, and per-clip per-aspect mp4s + thumb +
    edit.json — matching the shapes consumed by ``app._job_payload`` and
    ``app._clip_editor_payload``.
    """
    d = data_dir / job_id
    (d / "clips").mkdir(parents=True, exist_ok=True)

    # source video (probe-able existence; placeholder bytes by default)
    (d / "source.mp4").write_bytes(media_bytes)

    # transcript: a handful of segments with word-level timing
    segments = []
    t = 0.0
    for i in range(8):
        text = f"This is sentence number {i} in the talk."
        words = []
        wt = t
        for w in text.split():
            words.append({"word": w, "start": round(wt, 3), "end": round(wt + 0.3, 3)})
            wt += 0.35
        seg_end = round(wt, 3)
        segments.append({"start": round(t, 3), "end": seg_end, "text": text, "words": words})
        t = seg_end + 0.2
    duration = round(t, 3)
    (d / "transcript.json").write_text(json.dumps(
        {"language": "en", "duration": duration, "segments": segments, "vad_dropped": 0},
        indent=2,
    ))

    # clips.json — candidates spread across the timeline
    clips_meta = []
    seg_per = max(1, len(segments) // n_clips)
    for i in range(n_clips):
        s_seg = i * seg_per
        e_seg = min(len(segments) - 1, s_seg + seg_per)
        clips_meta.append({
            "start": segments[s_seg]["start"], "end": segments[e_seg]["end"],
            "start_seg": s_seg, "end_seg": e_seg,
            "title": f"Clip {i + 1}", "rationale": f"Strong hook {i + 1}", "score": 0.9 - i * 0.1,
        })
    (d / "clips.json").write_text(json.dumps(
        {"transcript_hash": "seedhash", "clips": clips_meta}, indent=2))

    # rendered outputs + render.json
    render_clips = []
    for i, cm in enumerate(clips_meta, start=1):
        cdir = d / "clips" / f"clip{i:02d}"
        cdir.mkdir(parents=True, exist_ok=True)
        outputs = {}
        for a in aspects:
            f = cdir / f"{ASPECT_SLUG[a]}.mp4"
            f.write_bytes(media_bytes)
            outputs[a] = str(f)
        thumb = cdir / "thumb.jpg"
        thumb.write_bytes(b"\xff\xd8\xff\xd9")  # minimal JPEG SOI/EOI
        (cdir / "edit.json").write_text(json.dumps({
            "start": cm["start"], "end": cm["end"],
            "transforms": {a: {"zoom": 1.0, "x": 0.0, "y": 0.0} for a in aspects},
            "audio": {"mute": False, "volume": 1.0},
        }, indent=2))
        render_clips.append({
            "index": i, "dir": str(cdir), "outputs": outputs, "thumb": str(thumb),
            "captions": "overlay", "title": cm["title"], "score": cm["score"],
            "transforms": {a: {"zoom": 1.0, "x": 0.0, "y": 0.0} for a in aspects},
            "start": cm["start"], "end": cm["end"], "audio": {"mute": False, "volume": 1.0},
        })
    (d / "clips" / "render.json").write_text(json.dumps({"clips": render_clips}, indent=2))

    # manifest — all stages done
    (d / "job.json").write_text(json.dumps({
        "job_id": job_id, "source_name": "seed_talk.mp4", "created_at": time.time(),
        "content_id": job_id[:10], "source_ext": ".mp4", "source_dims": list(source_dims),
        "awaiting_run": awaiting_run,
        "run_params": {"model": "base.en", "max_clips": n_clips, "captions": "overlay",
                       "transforms": {a: {"zoom": 1.0, "x": 0.0, "y": 0.0} for a in aspects}},
        "tools": {"whisper_model": "base.en"},
        "stages": {
            "ingest": {"status": "done", "source": "source.mp4"},
            "transcribe": {"status": "done", "segments": len(segments), "duration": duration},
            "select": {"status": "done", "clips": n_clips},
            "render": {"status": "done", "clips": n_clips, "aspects": len(aspects),
                       "clips_done": n_clips, "clips_total": n_clips},
        },
    }, indent=2))
    return d


@pytest.fixture
def data_dir(tmp_path) -> Path:
    return tmp_path


@pytest.fixture
def client(data_dir):
    """A TestClient bound to a throwaway DATA_DIR (also exposes .app/.cfg/.data_dir)."""
    from starlette.testclient import TestClient
    appmod, cfg = _reload_app(data_dir)
    with TestClient(appmod.app) as c:
        c.app_module = appmod  # type: ignore[attr-defined]
        c.cfg = cfg            # type: ignore[attr-defined]
        c.data_dir = data_dir  # type: ignore[attr-defined]
        yield c


@pytest.fixture
def seeded_job(client):
    """A completed job seeded into the client's DATA_DIR. Returns (client, job_id, job_dir)."""
    job_id = "seedjob0001"
    job_dir = seed_job(client.data_dir, job_id)
    return client, job_id, job_dir
