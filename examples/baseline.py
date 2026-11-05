"""Plain Whisper baseline: mono mix, 25 s chunks, no cleanup. What you get without unmuffle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from unmuffle import audio
from unmuffle.asr import Whisper


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("file", type=Path)
    ap.add_argument("--model", default="turbo")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    sr = audio.SAMPLE_RATE
    x = audio.load(a.file, sr).mean(axis=1)
    asr = Whisper(a.model)
    lines, step = [], 25.0
    t = 0.0
    while t < len(x) / sr:
        lines.append({"start": t, "text": asr(x[int(t * sr):int((t + step) * sr)], sr)})
        t += step
    a.out.write_text(json.dumps(lines, indent=2))
    for l in lines:
        print(f"[{int(l['start'] // 60)}:{int(l['start'] % 60):02d}] {l['text']}")


if __name__ == "__main__":
    main()
