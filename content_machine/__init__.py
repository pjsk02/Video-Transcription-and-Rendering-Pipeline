"""Video Transcription and Rendering Pipeline — local video clipping tools."""

__version__ = "0.1.0"

# The app logs and prints Unicode glyphs (→ ✓ ✗ ▶ ★ 📜). On Windows the console
# and file handlers default to cp1252, which raises UnicodeEncodeError on those
# characters. Force UTF-8 on stdio at import time so every entrypoint (CLI, web
# app, tests) is safe regardless of the active code page.
import sys as _sys

for _stream in (_sys.stdout, _sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    except (AttributeError, ValueError):
        pass  # non-reconfigurable stream (e.g. already wrapped / redirected)
