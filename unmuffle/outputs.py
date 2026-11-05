"""Write the run's files: cleaned audio, per-speaker audio, transcripts, decisions, report."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from . import audio, render
from .context import Context
from .data import Line
from .log import log, stamp


def speaker_filename(who: str) -> str:
    slug = who.lower().replace(" ", "_")
    return f"{slug if slug.startswith('speaker') else 'speaker_' + slug}.wav"


def write_audio(ctx: Context, out: Path, lines: list[Line], decisions: list[dict]) -> None:
    log("writing cleaned audio")
    sr = ctx.sr
    audio.save_wav(out / "cleaned.wav", render.finish(render.cleaned(ctx.tracks, decisions, sr), sr), sr)
    for who, sig in render.speaker_files(ctx.tracks, lines, sr).items():
        audio.save_wav(out / speaker_filename(who), render.finish(sig, sr), sr)
    if ctx.opts.write_tracks:
        for m, sig in ctx.tracks.items():
            audio.save_m4a(out / f"track_{m}.m4a", sig, sr)


def write_text(out: Path, lines: list[Line], decisions: list[dict], report: dict) -> None:
    txt, md = [], [f"# Transcript: {report['source']}", ""]
    low = 0
    for ln in lines:
        who = ln.speaker + ("" if ln.speaker_sure else " [?]")
        flag = ""
        if ln.confidence is not None and ln.confidence < 0.5:
            flag, low = " [?]", low + 1
        txt.append(f"[{stamp(ln.start)}] {who.upper()}: {ln.text}{flag}")
        md.append(f"**[{stamp(ln.start)}] {who}:** {ln.text}{flag}  ")
        if ln.alt:
            md.append(f"<sub>second model heard: {ln.alt}</sub>  ")
    key = ("\n\nKEY\n[?] after a line = the two models disagree; listen to that spot.\n"
           "[?] after a name = the speaker match is uncertain.\n")
    (out / "transcript.txt").write_text("\n".join(txt) + key)
    (out / "transcript.md").write_text("\n".join(md) + "\n")
    (out / "transcript.json").write_text(json.dumps([asdict(ln) for ln in lines], indent=2))
    methods = list(decisions[0]["mos"]) if decisions else []
    dec = ["# Decisions", "",
           "DNSMOS overall quality per method (1 to 5, higher is cleaner), the method used, and why.", "",
           "| Time | " + " | ".join(methods) + " | Used | Why |", "|---" * (len(methods) + 3) + "|"]
    for d in decisions:
        cells = [f"{d['mos'][m]['ovrl']:.2f}" for m in methods]
        dec.append(f"| {stamp(d['start'])} to {stamp(d['end'])} | " + " | ".join(cells) +
                   f" | {' + '.join(d['use'])} | {d['why']} |")
    (out / "decisions.md").write_text("\n".join(dec) + "\n")
    report["lines"] = len(lines)
    report["low_confidence_lines"] = low


def write_report(out: Path, report: dict, decisions: list[dict], methods: list[str]) -> None:
    report["methods_used"] = {m: sum(1 for d in decisions if m in d["use"]) for m in methods}
    report["decisions_for_scoring"] = [
        {k: d[k] for k in ("start", "end", "use", "mos")} | {"clean": d.get("clean", False)} for d in decisions]
    (out / "report.json").write_text(json.dumps(report, indent=2))
