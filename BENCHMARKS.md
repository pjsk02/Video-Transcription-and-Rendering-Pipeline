# Benchmarking

Use the benchmark script to compare the encoder selected by the application with
CPU `libx264` on your own video and hardware:

```bash
python scripts/benchmark.py path/to/video.mp4 --seconds 30 --start 90
```

Choose a start time and duration that fit within the source. The script cuts a
temporary sample, warms both encoding paths, renders 9:16, 1:1, and 16:9 outputs,
then reports elapsed time and checks each output with ffprobe. It also times local
transcription when a whisper.cpp model is available.

Results depend on the source video, ffmpeg build, encoder, drivers, and machine.
The selected encoder may be `libx264` when hardware encoding is unavailable; in
that case the two render paths use the same encoder. The script writes to a
temporary directory and prints its location so you can inspect the outputs.
