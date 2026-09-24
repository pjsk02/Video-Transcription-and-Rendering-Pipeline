"""Benchmark and validate the selected hardware encoder against CPU x264.

Measures wall-time for rendering a clip in all 3 aspect ratios with the detected
GPU encoder (NVENC/VideoToolbox) versus forced CPU libx264, and validates every
output with ffprobe (codec, dimensions, audio stream present, not corrupt). Also
times CPU transcription for context. Proves the speedup AND that the CPU-fallback
path produces equally-valid output.

    python scripts/benchmark.py [SOURCE.mp4] [--seconds N] [--start S]

The GPU encoder is whatever hwaccel.select_encoder() picks on this machine; CPU is
forced via CM_FORCE_CPU. Nothing here touches the web app or any real job dir.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from content_machine import config, hwaccel, render  # noqa: E402

ASPECTS = ("9:16", "1:1", "16:9")


def _slice(source: Path, start: float, seconds: float, out: Path) -> Path:
    """Cut a short test clip (re-encoded, with audio) for repeatable timing."""
    subprocess.run([config.FFMPEG, "-hide_banner", "-y", "-ss", f"{start}",
                    "-i", str(source), "-t", f"{seconds}", "-c:v", "libx264",
                    "-preset", "veryfast", "-crf", "23", "-c:a", "aac", str(out)],
                   check=True, capture_output=True)
    return out


def _probe_valid(path: Path) -> dict:
    """ffprobe a rendered file: returns codec/dims/has_audio/ok."""
    out = subprocess.run([config.FFPROBE, "-v", "error", "-show_entries",
                          "stream=codec_type,codec_name,width,height",
                          "-of", "default=nw=1", str(path)],
                         capture_output=True, text=True).stdout
    has_v = "codec_type=video" in out
    has_a = "codec_type=audio" in out
    w = next((line.split("=")[1] for line in out.splitlines()
              if line.startswith("width=")), "?")
    h = next((line.split("=")[1] for line in out.splitlines()
              if line.startswith("height=")), "?")
    codec = next((line.split("=")[1] for line in out.splitlines()
                  if line.startswith("codec_name=")), "?")
    return {"ok": has_v and has_a and path.stat().st_size > 1000,
            "codec": codec, "dims": f"{w}x{h}", "audio": has_a,
            "size_kb": path.stat().st_size // 1024}


def _render_all(src: Path, encoder: dict, outdir: Path, seconds: float) -> tuple[float, list[dict]]:
    """Render the clip in all 3 aspects with `encoder`; return (wall_s, [probes])."""
    t0 = time.time()
    probes = []
    for a in ASPECTS:
        out = outdir / f"{encoder['key']}_{a.replace(':', 'x')}.mp4"
        cmd = render.build_render_cmd(src, 0.0, seconds, a, 0.0, out, 1920, 1080,
                                      png_events=None, zoom=1.2, y_offset=0.0,
                                      mute=False, volume=1.0, encoder=encoder)
        subprocess.run(cmd, check=True, capture_output=True)
        probes.append({"aspect": a, **_probe_valid(out)})
    return time.time() - t0, probes


def _time_transcribe_cpu(src: Path, work: Path, seconds: float) -> float | None:
    """Time CPU whisper transcription of the clip's audio (informational)."""
    try:
        from content_machine import transcribe as tr
        wav = work / "a.wav"
        tr.extract_audio(src, wav)
        model = config.model_path()
        if not model.exists():
            return None
        t0 = time.time()
        tr.run_whisper(wav, model, work / "w")
        return time.time() - t0
    except Exception as e:
        print(f"  (transcribe timing skipped: {e})")
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("source")
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--start", type=float, default=90.0)
    args = ap.parse_args()

    source = Path(args.source)
    if not source.exists():
        print(f"source not found: {source}")
        return 2

    work = Path(tempfile.mkdtemp(prefix="cm-bench-"))
    print(f"== Video Transcription and Rendering Pipeline render benchmark ==\nsource: {source} "
          f"({args.seconds:.0f}s slice @ {args.start:.0f}s)\nwork: {work}\n")

    clip = _slice(source, args.start, args.seconds, work / "clip.mp4")

    gpu = hwaccel.select_encoder()
    cpu = hwaccel.X264
    print(f"GPU encoder: {gpu['key']} ({gpu['codec']})   CPU encoder: {cpu['codec']}\n")

    # warm both once (NVENC session setup / first-run), then measure
    _render_all(clip, gpu, work, args.seconds)
    _render_all(clip, cpu, work, args.seconds)
    gpu_s, gpu_probes = _render_all(clip, gpu, work, args.seconds)
    cpu_s, cpu_probes = _render_all(clip, cpu, work, args.seconds)

    all_valid = all(p["ok"] for p in gpu_probes + cpu_probes)
    speedup = cpu_s / gpu_s if gpu_s > 0 else 0.0

    print("RENDER (3 aspects each):")
    print(f"  {gpu['key']:<14} {gpu_s:6.2f}s   outputs: " +
          ", ".join(f"{p['aspect']} {p['codec']} {p['dims']} "
                    f"{'aud✓' if p['audio'] else 'NO-AUDIO'}" for p in gpu_probes))
    print(f"  {'x264 (CPU)':<14} {cpu_s:6.2f}s   outputs: " +
          ", ".join(f"{p['aspect']} {p['codec']} {p['dims']} "
                    f"{'aud✓' if p['audio'] else 'NO-AUDIO'}" for p in cpu_probes))
    print(f"  speedup: {speedup:.2f}x   all outputs valid: {all_valid}\n")

    tr_s = _time_transcribe_cpu(clip, work, args.seconds)
    if tr_s is not None:
        print(f"TRANSCRIBE (CPU whisper): {tr_s:.2f}s for {args.seconds:.0f}s audio "
              f"(realtime x{args.seconds / tr_s:.1f})\n")

    print("RESULT:", "PASS — GPU faster, all outputs valid" if (speedup > 1.0 and all_valid)
          else ("OK — all outputs valid (GPU not faster on this clip length)"
                if all_valid else "FAIL — an output was invalid"))
    return 0 if all_valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
