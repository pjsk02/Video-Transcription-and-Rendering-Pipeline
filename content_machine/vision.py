"""Optional local face-aware, static 9:16 crop suggestions for the clip editor."""

from __future__ import annotations

import json
import math
import os
import statistics
import subprocess
import tempfile
from pathlib import Path

from . import config, render
from .jobs import atomic_write_text, read_json

MODEL_NAME = "face_detection_yunet_2026may.onnx"
DEFAULT_MODEL_PATH = config.PROJECT_ROOT / "vendor" / "models" / MODEL_NAME
SAMPLE_COUNT = 12
MAX_ANALYSIS_SECONDS = 180
ANALYSIS_VERSION = 2


class FaceSuggestionUnavailable(RuntimeError):
    """The optional detector or its model is not installed."""


class NoFaceFound(ValueError):
    """The sampled frames did not contain enough usable face detections."""


def model_path() -> Path:
    return Path(os.environ.get("CM_FACE_MODEL", DEFAULT_MODEL_PATH))


def sample_face_boxes(source: Path, start: float, end: float) -> tuple[list[dict], int]:
    """Decode a small, evenly spaced frame set and return normalized face boxes.

    OpenCV is imported here so transcription/rendering work without the optional
    vision extra. ffmpeg handles video decoding; OpenCV runs the local YuNet model.
    """
    try:
        import cv2
    except ImportError as e:
        raise FaceSuggestionUnavailable(
            'Face suggestions need OpenCV. Install with: pip install -e ".[vision]"'
        ) from e

    model = model_path()
    if not model.is_file():
        raise FaceSuggestionUnavailable(
            f"Face detector model missing: {model}. Run python scripts/setup_vision.py."
        )
    config.require_tool(config.FFMPEG, config.ffmpeg_hint())
    duration = end - start
    fps = SAMPLE_COUNT / duration
    boxes: list[dict] = []
    with tempfile.TemporaryDirectory(prefix="face-samples-") as tmp:
        out = Path(tmp) / "frame_%03d.png"
        cmd = [
            config.FFMPEG, "-hide_banner", "-loglevel", "error", "-y",
            "-ss", f"{start:.3f}", "-i", str(source), "-t", f"{duration:.3f}",
            "-vf", f"fps={fps:.6f},scale=640:640:force_original_aspect_ratio=decrease",
            "-frames:v", str(SAMPLE_COUNT), "-an", str(out),
        ]
        subprocess.run(cmd, capture_output=True, check=True, timeout=180)
        frames = sorted(Path(tmp).glob("frame_*.png"))
        detector = cv2.FaceDetectorYN.create(
            model=str(model), config="", input_size=(640, 640),
            score_threshold=0.7, nms_threshold=0.3,
        )
        for frame in frames:
            image = cv2.imread(str(frame))
            if image is None:
                continue
            height, width = image.shape[:2]
            detector.setInputSize((width, height))
            _, detections = detector.detect(image)
            if detections is None or len(detections) == 0:
                continue
            # A static crop follows the dominant visible face. The user can
            # inspect/adjust the proposal when several people are on screen.
            best = max(detections, key=lambda d: float(d[2] * d[3]))
            x = max(0.0, float(best[0]) / width)
            y = max(0.0, float(best[1]) / height)
            right = min(1.0, float(best[0] + best[2]) / width)
            bottom = min(1.0, float(best[1] + best[3]) / height)
            if right > x and bottom > y:
                boxes.append({"x": x, "y": y, "w": right - x, "h": bottom - y})
    return boxes, len(frames)


def transform_for_faces(boxes: list[dict], source_w: int, source_h: int) -> dict:
    """Choose a conservative crop containing the detected face positions."""
    if not boxes or source_w <= 0 or source_h <= 0:
        raise NoFaceFound("No clear face found in this clip. Adjust the crop manually.")

    base_w, base_h, _, _ = render.compute_crop(source_w, source_h, "9:16")
    center_x = statistics.median(b["x"] + b["w"] / 2 for b in boxes)
    center_y = statistics.median(b["y"] + b["h"] / 2 for b in boxes)
    face_w = statistics.median(b["w"] for b in boxes) * source_w
    face_h = statistics.median(b["h"] for b in boxes) * source_h
    span_w = (max(b["x"] + b["w"] for b in boxes) - min(b["x"] for b in boxes)) * source_w
    span_h = (max(b["y"] + b["h"] for b in boxes) - min(b["y"] for b in boxes)) * source_h

    # Aim for a face around one quarter of the crop width, while leaving room
    # for movement observed across frames. Limit zoom to retain body/context.
    needed_w = max(face_w / 0.24, span_w * 1.3)
    needed_h = max(face_h / 0.22, span_h * 1.3)
    zoom = max(1.0, min(1.6, base_w / max(needed_w, 1), base_h / max(needed_h, 1)))
    crop_w, crop_h, _, _ = render.compute_crop(source_w, source_h, "9:16", zoom=zoom)
    slack_x, slack_y = source_w - crop_w, source_h - crop_h

    # Keep the face near the upper third when vertical panning is possible.
    left = center_x * source_w - crop_w / 2
    top = center_y * source_h - crop_h * 0.36
    x = max(-1.0, min(1.0, 2 * left / slack_x - 1)) if slack_x > 0 else 0.0
    y = max(-1.0, min(1.0, 2 * top / slack_y - 1)) if slack_y > 0 else 0.0
    return {"zoom": round(zoom, 2), "x": round(x, 2), "y": round(y, 2)}


def suggest_face_crop(source: Path, start: float, end: float, cache_path: Path,
                      source_dims: tuple[int, int]) -> dict:
    """Return a cached or newly computed 9:16 suggestion without changing edits."""
    if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end - start < 0.5:
        raise ValueError("Choose a valid clip window of at least 0.5 seconds.")
    if end - start > MAX_ANALYSIS_SECONDS:
        raise ValueError("Face analysis is limited to clips of 180 seconds or less.")
    if not source.is_file():
        raise FileNotFoundError("Source video missing")

    model = model_path()
    source_stat = source.stat()
    model_stamp = [model.stat().st_size, model.stat().st_mtime_ns] if model.exists() else None
    key = [ANALYSIS_VERSION, str(source.resolve()), source_stat.st_size,
           source_stat.st_mtime_ns, round(start, 3), round(end, 3), model_stamp]
    if cache_path.exists():
        cached = read_json(cache_path, default={})
        if cached.get("key") == key:
            return {**cached["result"], "cached": True}

    boxes, sampled = sample_face_boxes(source, start, end)
    if len(boxes) < max(1, math.ceil(sampled / 4)):
        raise NoFaceFound("No clear face found across this clip. Adjust the crop manually.")
    transform = transform_for_faces(boxes, *source_dims)
    result = {"transform": transform, "faces_found": len(boxes),
              "frames_sampled": sampled, "aspect": "9:16"}
    atomic_write_text(cache_path, json.dumps({"key": key, "result": result}, indent=2))
    return {**result, "cached": False}
