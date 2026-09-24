"""Video Transcription and Rendering Pipeline CLI.

  content-machine ingest <video>   # phase 1: transcribe
  content-machine select <job_id>  # phase 2: pick clips (added in phase 2)
  content-machine render <job_id>  # phase 3: cut + caption (added in phase 3)
  content-machine serve            # phase 4: localhost UI (added in phase 4)
"""

from __future__ import annotations

from pathlib import Path

import typer

app = typer.Typer(add_completion=False, help="Video Transcription and Rendering Pipeline.")


@app.callback()
def main() -> None:
    """Transcribe, select, and render captioned video clips."""


@app.command()
def ingest(
    video: Path = typer.Argument(..., exists=True, dir_okay=False, help="Local video file"),
    model: str = typer.Option(None, "--model", "-m", help="whisper model (default: base.en)"),
    no_vad: bool = typer.Option(False, "--no-vad", help="Disable silence/hallucination filter"),
    force: bool = typer.Option(False, "--force", help="Re-transcribe even if cached"),
    language: str = typer.Option(None, "--language", "-l", help="Force language (e.g. en)"),
):
    """Transcribe a video into data/<job_id>/transcript.json."""
    from . import transcribe as t

    typer.echo(f"→ Ingesting {video.name} ...")
    job = t.transcribe(video, model=model, vad=not no_vad, force=force, language=language)
    transcript = job.transcript_path
    typer.echo(f"✓ Job {job.job_id}")
    typer.echo(f"  data dir:   {job.data_dir}")
    typer.echo(f"  transcript: {transcript}")
    if transcript.exists():
        import json
        data = json.loads(transcript.read_text())
        typer.echo(f"  language:   {data.get('language')}")
        typer.echo(f"  duration:   {data.get('duration', 0):.1f}s")
        typer.echo(f"  segments:   {len(data.get('segments', []))}"
                   + (f" (VAD dropped {data['vad_dropped']})" if data.get('vad_dropped') else ""))


@app.command()
def select(
    job_id: str = typer.Argument(..., help="Job id from `ingest`"),
    max_clips: int = typer.Option(6, "--max-clips", "-n", help="Max clips to select"),
    force: bool = typer.Option(False, "--force", help="Re-select even if cached"),
):
    """Pick the best clip-worthy moments via Claude (subscription) → clips.json."""
    from . import select as s

    typer.echo(f"→ Selecting clips for {job_id} (via claude -p) ...")
    path = s.select_clips(job_id, max_clips=max_clips, force=force)
    import json
    data = json.loads(path.read_text())
    typer.echo(f"✓ {len(data['clips'])} clips → {path}")
    for i, c in enumerate(data["clips"], 1):
        typer.echo(f"  {i}. [{c['start']:.0f}-{c['end']:.0f}s] score {c['score']:.1f}  {c['title']}")


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", help="Bind host (localhost only by default)"),
    port: int = typer.Option(8000, help="Port"),
):
    """Launch the localhost web UI (upload → review → download)."""
    import uvicorn
    typer.echo(f"→ Video Transcription and Rendering Pipeline UI at http://{host}:{port}")
    uvicorn.run("content_machine.app:app", host=host, port=port)


@app.command()
def render(
    job_id: str = typer.Argument(..., help="Job id from `ingest`"),
    x_offset: float = typer.Option(0.0, "--x-offset", help="Crop offset -1..1 for off-center speakers"),
    captions_mode: str = typer.Option("overlay", "--captions", help="Caption renderer: overlay | hyperframes | none"),
):
    """Cut + reframe (9:16/1:1/16:9) + captions for each selected clip."""
    from . import render as r

    typer.echo(f"→ Rendering clips for {job_id} (captions={captions_mode}) ...")
    path = r.render_job(job_id, x_offset=x_offset, caption_mode=captions_mode)
    import json
    data = json.loads(path.read_text())
    typer.echo(f"✓ Rendered {len(data['clips'])} clips → {path.parent}")
    for c in data["clips"]:
        typer.echo(f"  clip{c['index']:02d}: {', '.join(c['outputs'].keys())}  ({c['captions']} captions)")


if __name__ == "__main__":
    app()
