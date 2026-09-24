"""Face crop proposals stay within the renderer's existing transform contract."""

import pytest

from content_machine import vision


def test_face_on_right_produces_rightward_9x16_crop():
    boxes = [{"x": 0.72, "y": 0.18, "w": 0.10, "h": 0.16}] * 5
    tf = vision.transform_for_faces(boxes, 1920, 1080)
    assert 1 <= tf["zoom"] <= 1.6
    assert tf["x"] > 0.5
    assert -1 <= tf["y"] <= 1


def test_no_faces_does_not_propose_crop():
    with pytest.raises(vision.NoFaceFound):
        vision.transform_for_faces([], 1920, 1080)


def test_suggestion_caches_same_trim_and_reanalyzes_changed_trim(tmp_path, monkeypatch):
    source = tmp_path / "source.mp4"
    source.write_bytes(b"video")
    model = tmp_path / "face.onnx"
    model.write_bytes(b"model")
    monkeypatch.setattr(vision, "model_path", lambda: model)
    cache = tmp_path / "face_suggestion.json"
    calls = []

    def fake_sample(_source, start, end):
        calls.append((start, end))
        return ([{"x": 0.7, "y": 0.2, "w": 0.1, "h": 0.15}] * 4, 12)

    monkeypatch.setattr(vision, "sample_face_boxes", fake_sample)
    first = vision.suggest_face_crop(source, 5, 20, cache, (1920, 1080))
    second = vision.suggest_face_crop(source, 5, 20, cache, (1920, 1080))
    third = vision.suggest_face_crop(source, 5, 21, cache, (1920, 1080))
    assert first["cached"] is False
    assert second["cached"] is True
    assert third["cached"] is False
    assert first["transform"] == second["transform"]
    assert calls == [(5, 20), (5, 21)]


def test_sparse_detections_leave_manual_framing_untouched(tmp_path, monkeypatch):
    source = tmp_path / "source.mp4"
    source.write_bytes(b"video")
    monkeypatch.setattr(vision, "sample_face_boxes", lambda *_: (
        [{"x": 0.7, "y": 0.2, "w": 0.1, "h": 0.15}], 12))
    with pytest.raises(vision.NoFaceFound):
        vision.suggest_face_crop(source, 5, 20, tmp_path / "cache.json", (1920, 1080))
    assert not (tmp_path / "cache.json").exists()
