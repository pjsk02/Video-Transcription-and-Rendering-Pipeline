"""Phase 25 (API-01..05): HTTP-boundary integration tests for content_machine/app.py.

These drive the FastAPI routes through the Starlette TestClient bound to a throwaway
DATA_DIR (the ``client`` fixture in conftest.py). They characterise *current* behaviour
at the HTTP edge — status codes, payload shapes, redirects, and the static /media mount —
stubbing only the things that would otherwise shell out (ffprobe via render.probe_dims,
the pipeline thread, and the background re-render worker).
"""

from __future__ import annotations

import json

from tests.conftest import seed_job

# A tiny valid-enough mp4 body for multipart uploads (matches conftest's placeholder).
_MP4 = bytes.fromhex("0000001c66747970697336300000020069736f6d69736f3261766331") + b"\x00" * 16


# --- API-01: /upload ---------------------------------------------------------
def test_upload_valid_video_redirects_and_stages_job(client, monkeypatch):
    # Stub the ffprobe shell-out so no real binary runs.
    monkeypatch.setattr(client.app_module.render, "probe_dims", lambda src: (1920, 1080))
    resp = client.post(
        "/upload",
        files={"video": ("talk.mp4", _MP4, "video/mp4")},
        follow_redirects=False,
    )
    assert resp.status_code == 303
    location = resp.headers["location"]
    assert location.startswith("/job/")
    job_id = location.rsplit("/", 1)[-1]

    job_dir = client.data_dir / job_id
    assert job_dir.is_dir()
    manifest = json.loads((job_dir / "job.json").read_text())
    assert manifest["awaiting_run"] is True
    assert manifest["source_ext"] == ".mp4"
    assert manifest["source_dims"] == [1920, 1080]
    assert (job_dir / "source.mp4").exists()
    # Phase 27 fix (bug #6): /upload now persists metadata first, then marks ingest
    # done — so the ingest stage is correctly "done" on disk (was reverted to
    # "pending" by a stale trailing save_manifest before the fix).
    assert manifest["stages"]["ingest"]["status"] == "done"


def test_upload_no_filename_rejected(client, monkeypatch):
    monkeypatch.setattr(client.app_module.render, "probe_dims", lambda src: (0, 0))
    # CHARACTERIZATION: a multipart part with an empty filename is parsed by Starlette
    # as a plain form field, not an UploadFile, so FastAPI's body validation rejects it
    # with 422 BEFORE the `if not video.filename` 400 guard is ever reached — that guard
    # is effectively dead for real multipart requests. (app.py:319-320)
    resp = client.post("/upload", files={"video": ("", _MP4, "application/octet-stream")})
    assert resp.status_code == 422


def test_upload_bad_extension_rejected(client, monkeypatch):
    monkeypatch.setattr(client.app_module.render, "probe_dims", lambda src: (0, 0))
    resp = client.post("/upload", files={"video": ("evil.sh", b"#!/bin/sh\n", "text/plain")})
    assert resp.status_code == 400


def test_upload_dotfile_rejected(client, monkeypatch):
    monkeypatch.setattr(client.app_module.render, "probe_dims", lambda src: (0, 0))
    resp = client.post("/upload", files={"video": (".hidden.mp4", _MP4, "video/mp4")})
    assert resp.status_code == 400


def test_upload_traversal_name_is_neutralized_not_rejected(client, monkeypatch):
    """CHARACTERIZATION: ``../escape.mp4`` is NOT rejected — safe_upload_name takes the
    basename (``escape.mp4``), which is a valid video name, so the upload SUCCEEDS (303).
    Traversal is defused by basename-only, not by a 4xx. (app.py:42-49)"""
    monkeypatch.setattr(client.app_module.render, "probe_dims", lambda src: (1920, 1080))
    resp = client.post(
        "/upload",
        files={"video": ("../escape.mp4", _MP4, "video/mp4")},
        follow_redirects=False,
    )
    assert resp.status_code == 303


# --- API-02: /run ------------------------------------------------------------
class _NoThread:
    """Drop-in for threading.Thread whose .start() is a no-op (pipeline never runs)."""

    def __init__(self, *a, **k):
        pass

    def start(self):
        pass


def test_run_starts_awaiting_job(client, monkeypatch):
    monkeypatch.setattr(client.app_module.threading, "Thread", _NoThread)
    job_id = "runjob0001"
    seed_job(client.data_dir, job_id, awaiting_run=True)
    resp = client.post(f"/api/job/{job_id}/run", data={"max_clips": 3})
    assert resp.status_code == 200
    assert resp.json() == {"ok": True}
    # awaiting_run flipped off
    manifest = json.loads((client.data_dir / job_id / "job.json").read_text())
    assert manifest["awaiting_run"] is False


def test_run_twice_conflicts(client, monkeypatch):
    monkeypatch.setattr(client.app_module.threading, "Thread", _NoThread)
    job_id = "runjob0002"
    seed_job(client.data_dir, job_id, awaiting_run=True)
    assert client.post(f"/api/job/{job_id}/run").status_code == 200
    # second call: awaiting_run is now False -> 409
    resp = client.post(f"/api/job/{job_id}/run")
    assert resp.status_code == 409


def test_run_unknown_job_404(client, monkeypatch):
    monkeypatch.setattr(client.app_module.threading, "Thread", _NoThread)
    resp = client.post("/api/job/doesnotexist/run")
    assert resp.status_code == 404


def test_run_missing_source_400(client, monkeypatch):
    monkeypatch.setattr(client.app_module.threading, "Thread", _NoThread)
    job_id = "runjob0003"
    seed_job(client.data_dir, job_id, awaiting_run=True)
    # remove the source so the source-presence guard trips
    (client.data_dir / job_id / "source.mp4").unlink()
    resp = client.post(f"/api/job/{job_id}/run")
    assert resp.status_code == 400


# --- API-03: read endpoints --------------------------------------------------
def test_api_job_returns_clips(seeded_job):
    client, job_id, _ = seeded_job
    resp = client.get(f"/api/job/{job_id}")
    assert resp.status_code == 200
    body = resp.json()
    assert body["job_id"] == job_id
    assert len(body["clips"]) == 2
    assert body["clips"][0]["index"] == 1
    assert "outputs" in body["clips"][0]


def test_api_job_log(seeded_job):
    client, job_id, _ = seeded_job
    resp = client.get(f"/api/job/{job_id}/log", params={"lines": 5})
    assert resp.status_code == 200
    assert "log" in resp.json()


def test_api_clip_editor_payload(seeded_job):
    client, job_id, _ = seeded_job
    resp = client.get(f"/api/job/{job_id}/clip/1")
    assert resp.status_code == 200
    body = resp.json()
    assert body["index"] == 1
    assert "transforms" in body and "captions" in body


def test_face_crop_suggestion_uses_current_trim_without_saving(seeded_job, monkeypatch):
    client, job_id, job_dir = seeded_job
    captured = {}
    edit_path = job_dir / "clips" / "clip01" / "edit.json"
    original_edit = edit_path.read_bytes() if edit_path.exists() else None

    def fake_suggest(source, start, end, cache, dims):
        captured.update(source=source, start=start, end=end, cache=cache, dims=dims)
        return {"aspect": "9:16", "transform": {"zoom": 1.2, "x": 0.7, "y": 0.0},
                "faces_found": 8, "frames_sampled": 12, "cached": False}

    monkeypatch.setattr(client.app_module.vision, "suggest_face_crop", fake_suggest)
    resp = client.post(f"/api/job/{job_id}/clip/1/suggest-crop",
                       json={"start": 2.0, "end": 18.0})
    assert resp.status_code == 200
    assert resp.json()["transform"]["x"] == 0.7
    assert captured["start"] == 2.0 and captured["end"] == 18.0
    assert captured["cache"].name == "face_suggestion.json"
    assert (edit_path.read_bytes() if edit_path.exists() else None) == original_edit


def test_face_crop_suggestion_reports_unavailable_detector(seeded_job, monkeypatch):
    client, job_id, _ = seeded_job

    def unavailable(*_args):
        raise client.app_module.vision.FaceSuggestionUnavailable("Install the vision extra")

    monkeypatch.setattr(client.app_module.vision, "suggest_face_crop", unavailable)
    resp = client.post(f"/api/job/{job_id}/clip/1/suggest-crop", json={})
    assert resp.status_code == 503
    assert "Install" in resp.json()["detail"]


def test_face_crop_suggestion_rejects_invalid_clip_and_window(seeded_job):
    client, job_id, _ = seeded_job
    assert client.post(f"/api/job/{job_id}/clip/99/suggest-crop", json={}).status_code == 404
    assert client.post(f"/api/job/{job_id}/clip/1/suggest-crop",
                       json={"start": -1, "end": 10}).status_code == 400


# --- FE-05: re-derive captions endpoint --------------------------------------
def test_api_clip_captions_returns_window_segments(seeded_job):
    """FE-05: re-derive caption events for an arbitrary [start,end] source window."""
    client, job_id, _ = seeded_job
    resp = client.get(f"/api/job/{job_id}/clip/1/captions", params={"start": 0, "end": 100})
    assert resp.status_code == 200
    body = resp.json()
    assert "segments" in body and isinstance(body["segments"], list)
    assert body["segments"], "a 0–100s window should overlap the seeded transcript"
    seg = body["segments"][0]
    assert {"start", "end", "text"} <= set(seg)
    assert seg["start"] >= 0  # clip-relative (re-timed to the window start)


def test_api_clip_captions_unknown_job_404(client):
    resp = client.get("/api/job/nope/clip/1/captions", params={"start": 0, "end": 5})
    assert resp.status_code == 404


def test_api_clip_captions_out_of_range_idx_404(seeded_job):
    client, job_id, _ = seeded_job
    resp = client.get(f"/api/job/{job_id}/clip/99/captions", params={"start": 0, "end": 5})
    assert resp.status_code == 404


# --- FE-08: media URLs carry a cache-busting ?v= -----------------------------
def test_api_job_outputs_carry_cache_bust(seeded_job):
    """FE-08: every output media URL now carries ?v=<int mtime> so rebuilt files bust."""
    client, job_id, _ = seeded_job
    body = client.get(f"/api/job/{job_id}").json()
    for clip in body["clips"]:
        for url in clip["outputs"].values():
            assert url.startswith("/media/") and "?v=" in url


def test_api_job_unknown_404(client):
    assert client.get("/api/job/nope/").status_code in (404, 307, 308)
    assert client.get("/api/job/nope").status_code == 404


def test_api_clip_out_of_range_404(seeded_job):
    client, job_id, _ = seeded_job
    resp = client.get(f"/api/job/{job_id}/clip/99")
    assert resp.status_code == 404


# --- API-04: edit + rerender-status ------------------------------------------
def test_edit_too_short_trim_400(seeded_job):
    client, job_id, _ = seeded_job
    resp = client.post(f"/api/job/{job_id}/clip/1/edit", json={"start": 0, "end": 0.2})
    assert resp.status_code == 400


def test_edit_valid_queues(seeded_job, monkeypatch):
    client, job_id, _ = seeded_job
    captured = {}

    def fake_enqueue(jid, idx, edit, aspects):
        captured["args"] = (jid, idx, edit, aspects)

    monkeypatch.setattr(client.app_module, "_enqueue_rerender", fake_enqueue)
    resp = client.post(
        f"/api/job/{job_id}/clip/1/edit",
        json={"start": 1.0, "end": 3.0, "aspects": ["9:16"]},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["queued"] is True
    assert body["index"] == 1
    assert body["aspects"] == ["9:16"]
    assert captured["args"][0] == job_id


def test_rerender_status_idle_shape(seeded_job, monkeypatch):
    client, job_id, _ = seeded_job
    # stub enqueue so no tracker is created -> idle
    monkeypatch.setattr(client.app_module, "_enqueue_rerender", lambda *a, **k: None)
    resp = client.get(f"/api/job/{job_id}/clip/1/rerender-status")
    assert resp.status_code == 200
    body = resp.json()
    assert body["active"] is False
    assert body["status"] == "idle"
    assert body["aspects"] == {}
    assert body["result"] is None


# --- API-05: download + /media mount + legacy reframe ------------------------
def test_download_ok(seeded_job):
    client, job_id, _ = seeded_job
    resp = client.get(f"/download/{job_id}/1/9:16")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "video/mp4"


def test_download_missing_aspect_404(seeded_job):
    client, job_id, _ = seeded_job
    # 4:5 is not a rendered aspect -> no such file
    resp = client.get(f"/download/{job_id}/1/4:5")
    assert resp.status_code == 404


def test_download_missing_clip_404(seeded_job):
    client, job_id, _ = seeded_job
    resp = client.get(f"/download/{job_id}/99/9:16")
    assert resp.status_code == 404


def test_media_mount_denies_job_manifest(seeded_job):
    """VAL-05: the /media mount now serves only an allowlist of media extensions,
    so job metadata (job.json / transcript.json) is no longer leaked — it 404s,
    while real media (source.mp4) still serves."""
    client, job_id, _ = seeded_job
    assert client.get(f"/media/{job_id}/job.json").status_code == 404
    assert client.get(f"/media/{job_id}/transcript.json").status_code == 404
    assert client.get(f"/media/{job_id}/source.mp4").status_code == 200


def test_legacy_reframe_with_render_stubbed(seeded_job, monkeypatch):
    client, job_id, job_dir = seeded_job
    out_path = str(job_dir / "clips" / "clip01" / "9x16.mp4")

    def fake_rerender(jid, idx, x_offset=0.0, caption_mode="overlay", **kw):
        return {"index": idx, "outputs": {"9:16": out_path}}

    monkeypatch.setattr(client.app_module.render, "rerender_one", fake_rerender)
    resp = client.post(f"/api/job/{job_id}/clip/1/reframe", data={"x_offset": 0.0})
    assert resp.status_code == 200
    body = resp.json()
    assert body["index"] == 1
    assert body["outputs"]["9:16"].startswith("/media/")
