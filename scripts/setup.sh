#!/usr/bin/env bash
# One-shot local setup for Video Transcription and Rendering Pipeline (macOS / Apple Silicon).
# Idempotent: safe to re-run. Installs deps, builds whisper.cpp (Metal),
# downloads a model, and creates the Python venv.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PATH="/opt/homebrew/bin:$PATH"

MODEL="${CM_WHISPER_MODEL:-base.en}"

# GPU acceleration on macOS is automatic: Homebrew ffmpeg includes the
# h264_videotoolbox encoder (the app probes + uses it, else falls back to CPU
# libx264), and whisper.cpp is built with Metal below. Nothing extra to install.
echo "==> 1/4 System deps (ffmpeg [VideoToolbox], cmake)"
command -v ffmpeg >/dev/null 2>&1 || brew install ffmpeg
command -v cmake  >/dev/null 2>&1 || brew install cmake

echo "==> 2/4 whisper.cpp (clone + build with Metal)"
mkdir -p vendor
if [ ! -d vendor/whisper.cpp/.git ]; then
  git clone --depth 1 https://github.com/ggml-org/whisper.cpp vendor/whisper.cpp
fi
( cd vendor/whisper.cpp
  cmake -B build -DGGML_METAL=1 -DWHISPER_BUILD_TESTS=OFF -DWHISPER_BUILD_EXAMPLES=ON
  cmake --build build -j --config Release
  bash ./models/download-ggml-model.sh "$MODEL" )

echo "==> 3/4 Python venv + deps"
python3 -m venv .venv
./.venv/bin/pip install --quiet --upgrade pip
./.venv/bin/pip install --quiet -e ".[dev,vision]"
./.venv/bin/python scripts/setup_vision.py

echo "==> 4/4 Smoke test"
./.venv/bin/pytest -q || true

echo "✓ Setup complete. Activate with: source .venv/bin/activate"
echo "  Then: content-machine ingest <your-video.mp4>"
