"""Watch a folder: drop an audio or video file in, get a transcript folder next to it."""

from __future__ import annotations

import time
import traceback
from pathlib import Path

from . import audio, pipeline


def _pending(folder: Path) -> list[Path]:
    files = []
    for p in sorted(folder.iterdir()):
        if p.is_file() and p.suffix.lower() in audio.AUDIO_EXTS and not p.name.startswith("."):
            done = p.with_name(p.stem + ".unmuffle") / "transcript.txt"
            failed = p.with_name(p.stem + ".unmuffle") / "error.txt"
            if not done.exists() and not failed.exists():
                files.append(p)
    return files


def watch(folder: str | Path, opts: pipeline.Options, interval: float = 5.0, once: bool = False) -> None:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    print(f"watching {folder.resolve()} (Ctrl+C to stop)")
    sizes: dict[Path, int] = {}
    while True:
        for p in _pending(folder):
            size = p.stat().st_size
            if sizes.get(p) != size:  # still copying; wait for the size to settle
                sizes[p] = size
                continue
            try:
                pipeline.run(p, opts=opts)
            except Exception:  # keep watching; leave the error next to the file
                out = p.with_name(p.stem + ".unmuffle")
                out.mkdir(exist_ok=True)
                (out / "error.txt").write_text(traceback.format_exc())
                print(f"failed on {p.name}; see {out / 'error.txt'}")
            sizes.pop(p, None)
        if once and not _pending(folder):
            return
        time.sleep(interval)
