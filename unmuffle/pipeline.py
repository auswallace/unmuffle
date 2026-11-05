"""End-to-end pipeline: any audio or video file in, cleaned audio and a checked transcript out.

    read file ─► build method tracks ─► find clean and noisy parts ─► find speakers
             ─► decide per speaker turn ─► check with a second model ─► write files

Each stage lives in its own module; every expensive step is cached in
<out>/cache, so an interrupted run resumes.
"""

from __future__ import annotations

import time
from pathlib import Path

from . import audio, cancel, decide, outputs, segment, speakers
from .analysis import analyze, noise_floor  # noqa: F401  (re-exported)
from .context import Context
from .data import METHODS, Line, Options, Piece  # noqa: F401  (re-exported)
from .log import log
from .tracks import build_tracks, enhance_chunked  # noqa: F401  (re-exported)


def run(src: str | Path, out_dir: str | Path | None = None, opts: Options | None = None) -> Path:
    opts = opts or Options()
    src = Path(src)
    out = Path(out_dir) if out_dir else src.with_name(src.stem + ".unmuffle")
    work = out / "cache"
    work.mkdir(parents=True, exist_ok=True)
    sr = audio.SAMPLE_RATE
    t0 = time.time()

    # 1. read
    log(f"reading {src.name}")
    x = audio.load(src, sr)
    report: dict = {"source": src.name, **analyze(x, sr)}
    stereo = x.shape[1] == 2 and not cancel.channel_report(x[:, 0], x[:, 1])["identical"]
    if stereo:
        report["channels"] = cancel.channel_report(x[:, 0], x[:, 1])
    methods = [m for m in opts.methods if m != "cancel" or stereo]
    if "cancel" in opts.methods and not stereo:
        log("mono input: the two-mic canceller is skipped")

    # 2. one track per cleaning method
    tracks = build_tracks(x, methods, sr, work, opts.threads)
    ctx = Context(opts, sr, work, tracks, methods, report)

    # 3. clean and noisy parts, judged on the raw audio in chunks cut at pauses
    dur = len(tracks["raw"]) / sr
    chunks = segment.pause_cuts([tracks[m] for m in methods if m != "raw"] or [tracks["raw"]], sr, 0.0, dur)
    log(f"{len(chunks)} chunks; scoring the raw audio to find clean and noisy parts")
    regs = speakers.regions(ctx, chunks)

    # 4. speakers: turns per condition, background voices folded, noisy voices matched to clean ones
    spk = speakers.Speakers(ctx, regs)
    pieces = spk.fold_small(spk.pieces())
    spk.match(pieces)

    # 5. per turn: pick the method and the line(s)
    lines, decisions = decide.decide(ctx, pieces, spk)
    del spk
    ctx.release()

    # 6. second-model check
    if opts.check_model:
        lines = decide.check(ctx, lines)
    lines.sort(key=lambda ln: ln.start)
    speakers.name_speakers(lines, opts.names)

    # 7. files
    outputs.write_audio(ctx, out, lines, decisions)
    outputs.write_text(out, lines, decisions, report)
    report["elapsed_s"] = round(time.time() - t0, 1)
    outputs.write_report(out, report, decisions, methods)
    ctx.release()
    log(f"done in {report['elapsed_s']} s -> {out}")
    return out
