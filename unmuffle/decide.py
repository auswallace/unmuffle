"""For each speaker turn, pick the cleaning method and the transcript line(s).

Clean turns keep the original audio. Noisy turns:

1. Score every method with DNSMOS.
2. Transcribe the best `shortlist` methods, plus the canceller (stereo files).
3. Judge = DNSMOS + words heard − loops (see judge.py), with hysteresis per speaker.
4. Use the canceller only if it heard words voice isolation missed.
5. Keep a second line only if another method heard different words in a different voice.

Then a second model re-transcribes every line on the same audio and flags disagreement.
"""

from __future__ import annotations

import numpy as np

from . import judge, segment, text
from .context import Context
from .data import Line, Piece
from .log import log
from .speakers import Speakers


def second_voice(win: judge.Candidate, cands: list[judge.Candidate]) -> judge.Candidate | None:
    """Another method that heard different, credible words: usually a second speaker on another track."""
    best = None
    for c in cands:
        if c is win or len(text.tokens(c.text)) < 4 or c.score < win.score - 1.0:
            continue
        different = text.containment(c.text, win.text) < 0.6 and text.similarity(c.text, win.text) < 0.5
        if different and (best is None or c.score > best.score):
            best = c
    return best


def chunk_lines(ctx: Context, a: float, b: float, win: judge.Candidate,
                second: judge.Candidate | None) -> list[Line]:
    """Lines for one turn; with two voices, order them by where each track's voice energy sits."""
    if not win.text:
        return []
    meta = {"span": (a, b)}
    lw = Line(a, b, win.text, win.name, meta={**meta, "source": win.name})
    if not second:
        return [lw]
    ls = Line(a, b, second.text, second.name, meta={**meta, "source": second.name})
    e1 = segment.speech_envelope(ctx.audio(win.name, a, b), ctx.sr)
    e2 = segment.speech_envelope(ctx.audio(second.name, a, b), ctx.sr)
    r = np.log10((e2 + 1e-12) / (e1 + 1e-12))
    idx = np.arange(r.size)
    hi, lo = r > np.median(r), r <= np.median(r)
    t2 = idx[hi].mean() if hi.any() else 0
    t1 = idx[lo].mean() if lo.any() else 0
    mid = (a + b) / 2
    if t2 < t1:
        ls.start, lw.start = a, mid
        return [ls, lw]
    lw.start, ls.start = a, mid
    return [lw, ls]


def decide(ctx: Context, pieces: list[Piece], spk: Speakers) -> tuple[list[Line], list[dict]]:
    log(f"{len(pieces)} speaker turns to transcribe")
    opts = ctx.opts
    lines: list[Line] = []
    decisions: list[dict] = []
    previous: dict[str, str] = {}
    for n, (a, b, label, clean) in enumerate(pieces):
        row = {m: ctx.mos(m, a, b) for m in ctx.methods}
        who = spk.who(label)
        if clean:
            t = text.clean(ctx.transcribe(opts.model, "raw", a, b))
            if t:
                lines.append(Line(a, b, t, "raw", speaker=who, speaker_sure=not label.endswith("-1"),
                                  meta={"source": "raw", "span": (a, b)}))
            decisions.append({"start": a, "end": b, "mos": row, "use": ["raw"], "speaker": who, "clean": True,
                              "why": "clean part: original audio kept"})
            continue

        short = sorted(ctx.methods, key=lambda m: -row[m]["ovrl"])[: opts.shortlist]
        if "cancel" in ctx.methods and "cancel" not in short:
            short.append("cancel")
        cands = []
        for m in short:
            raw_t = ctx.transcribe(opts.model, m, a, b)
            cands.append(judge.Candidate(m, row[m], raw_t, text.clean(raw_t)))
        judge.score(cands)
        win = judge.pick(cands, previous.get(who))
        notes = []
        # the canceller can dull a voice that reaches both mics equally; use it only when it hears new words
        iso = next((c for c in cands if c.name == "dpdfnet"), None)
        if win.name == "cancel" and iso and iso.text and (
                text.similarity(win.text, iso.text) >= 0.6 or text.containment(win.text, iso.text) >= 0.7):
            win = iso
            notes.append("canceller heard nothing new, kept voice isolation")
        second = second_voice(win, cands)
        if second and text.containment(win.text, second.text) >= 0.6:
            win, second = second, None
        other = None
        if second:
            # keep a second line only if it is a different person, not a garbled copy of the same one
            other = spk.voice_of(ctx.audio(second.name, a, b), who)
            if other is None:
                second = None
        use = [win.name] + ([second.name] if second else [])
        new = chunk_lines(ctx, a, b, win, second)
        for ln in new:
            ln.speaker, ln.speaker_sure = who, spk.sure.get(label, True)
            if second and len(new) == 2 and ln.track == second.name:
                ln.speaker, ln.speaker_sure = other, True
        lines += new
        why = ", ".join(f"{c.name} {c.score}" + (f" ({'; '.join(c.notes)})" if c.notes else "") for c in cands)
        decisions.append({"start": a, "end": b, "mos": row, "use": use, "speaker": who,
                          "scores": {c.name: c.score for c in cands},
                          "why": f"scored {why}; used {' + '.join(use)}" + "".join(f"; {x}" for x in notes)})
        previous[who] = win.name
        if (n + 1) % 10 == 0:
            log(f"  {n + 1}/{len(pieces)}")
    return lines, decisions


def check(ctx: Context, lines: list[Line]) -> list[Line]:
    """Re-transcribe every line with a second model; flag disagreement; drop tiny disputed fragments."""
    model = ctx.opts.check_model
    log(f"checking {len(lines)} lines with {model}")
    for i, ln in enumerate(lines):
        a, b = ln.meta["span"]
        alt = text.clean(ctx.transcribe(model, ln.meta.get("text_source", ln.meta["source"]), a, b))
        ln.confidence = round(max(text.similarity(ln.text, alt), text.containment(ln.text, alt)), 2) if alt else 0.0
        if ln.confidence < 0.5:
            ln.alt = alt
        if (i + 1) % 20 == 0:
            log(f"  check {i + 1}/{len(lines)}")
    dropped = [ln for ln in lines if ln.confidence < 0.3 and len(text.tokens(ln.text)) <= 3]
    ctx.report["dropped_fragments"] = [ln.text for ln in dropped]
    return [ln for ln in lines if ln not in dropped]
