"""Download OpenCV Zoo's optional YuNet model used for crop suggestions."""

from __future__ import annotations

import hashlib
import os
import tempfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODEL = ROOT / "vendor" / "models" / "face_detection_yunet_2026may.onnx"
MODEL_URL = (
    "https://github.com/opencv/opencv_zoo/raw/refs/heads/main/"
    "models/face_detection_yunet/face_detection_yunet_2026may.onnx"
)
MODEL_SHA256 = "ebafce4e3c118d6554634be5c27ab333b4c047a9a8c3faf1d7cf93101c22f0f0"


def valid_model(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size < 100_000:
        return False
    digest = hashlib.sha256()
    with path.open("rb") as model_file:
        for chunk in iter(lambda: model_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest() == MODEL_SHA256


def main() -> None:
    MODEL.parent.mkdir(parents=True, exist_ok=True)
    if valid_model(MODEL):
        print(f"Face detector model already present: {MODEL}")
        return
    fd, temp = tempfile.mkstemp(dir=MODEL.parent, suffix=".onnx.tmp")
    try:
        with os.fdopen(fd, "wb") as out, urllib.request.urlopen(MODEL_URL, timeout=60) as src:
            while chunk := src.read(1024 * 1024):
                out.write(chunk)
        if not valid_model(Path(temp)):
            raise RuntimeError("Downloaded face detector model is invalid")
        os.replace(temp, MODEL)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)
    print(f"Face detector model downloaded: {MODEL}")


if __name__ == "__main__":
    main()
