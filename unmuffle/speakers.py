"""Who is speaking when.

A muffled voice no longer sounds like the same person clean, so speakers are
found separately in the clean part and the noisy part, then paired up:

1. Split the file into clean and noisy regions (from how the raw audio scores).
2. Diarize each condition on its own. The clean part also gives the speaker count.
3. Cut both into pieces, one speaker each, short enough for Whisper.
4. Fold "speakers" with only a few seconds (usually background voices) into real ones.
5. Pair each noisy-part voice with a different clean-part voice (best total match).
"""

from __future__ import annotations

import json

import numpy as np

from . import judge, segment
from .context import Context
from .data import Line, Piece
from .log import log


def regions(ctx: Context, chunks: list[tuple[float, float]]) -> list[tuple[float, float, bool]]:
    """Merge chunks into runs of clean or noisy audio, judged on the raw track."""
    out: list[list] = []
    for a, b in chunks:
        c = judge.is_clean(ctx.mos("raw", a, b))
        if out and out[-1][2] == c:
            out[-1][1] = b
        else:
            out.append([a, b, c])
    ctx.report["clean_seconds"] = round(sum(b - a for a, b, c in out if c), 1)
    ctx.report["regions"] = [{"start": round(a, 1), "end": round(b, 1), "clean": c} for a, b, c in out]
    return [tuple(r) for r in out]


def turns_in(turns, a: float, b: float, min_len: float = 1.0) -> list[tuple[float, float, int]]:
    """Speaker turns clipped to [a, b]; tiny pieces merge into a neighbor."""
    parts = [(max(a, s), min(b, e), k) for s, e, k in turns if e > a and s < b]
    parts = [p for p in parts if p[1] - p[0] > 0.05]
    if not parts:
        return [(a, b, -1)]
    merged: list[list] = []
    for s, e, k in parts:
        if merged and (e - s < min_len or merged[-1][2] == k):
            merged[-1][1] = e
        else:
            merged.append([s, e, k])
    merged[0][0], merged[-1][1] = a, b
    return [tuple(m) for m in merged]


class Speakers:
    """Diarization and voice matching for one run."""

    def __init__(self, ctx: Context, regs: list[tuple[float, float, bool]]):
        self.ctx, self.regs = ctx, regs
        self.diar = segment.Diarizer(ctx.opts.speakers, ctx.opts.threads)
        self.noisy_src = "dpdfnet" if "dpdfnet" in ctx.tracks else "raw"
        self.mapping: dict[str, str] = {}
        self.sure: dict[str, bool] = {}
        self.prints: dict[str, np.ndarray] = {}

    def src_for(self, clean: bool) -> str:
        return "raw" if clean else self.noisy_src

    # --- diarization

    def _diarize(self, tag: str, sig: np.ndarray, keep: bool, k: int | None = None) -> list:
        ctx = self.ctx
        k_eff = k if k is not None else ctx.opts.speakers
        tpath = ctx.work / f"turns_{tag}{'' if k_eff is None else f'_k{k_eff}'}.json"
        if tpath.exists():
            return [tuple(t) for t in json.loads(tpath.read_text())]
        masked = sig.copy()
        for a, b, c in self.regs:
            if c != keep:
                masked[int(a * ctx.sr):int(b * ctx.sr)] = 0
        log(f"speaker diarization ({tag} part)")
        d = self.diar if k is None else segment.Diarizer(k, ctx.opts.threads)
        t = d.turns(masked, ctx.sr)
        tpath.write_text(json.dumps(t))
        return t

    def pieces(self) -> list[Piece]:
        ctx = self.ctx
        dur = len(ctx.tracks["raw"]) / ctx.sr
        clean_s = sum(b - a for a, b, c in self.regs if c)
        turns_c = self._diarize("clean", ctx.tracks["raw"], True) if clean_s >= 3 else []
        # the clean part tells us how many people there are; use that count in the noisy part
        k_clean = len({k for *_, k in turns_c}) if turns_c else 0
        k_noisy = ctx.opts.speakers or (k_clean if k_clean >= 1 else None)
        turns_n = self._diarize("noisy", ctx.tracks[self.noisy_src], False, k_noisy) if dur - clean_s >= 3 else []

        out: list[Piece] = []
        for a, b, c in self.regs:
            for sa, sb, k in turns_in(turns_c if c else turns_n, a, b):
                label = f"{'C' if c else 'N'}{k}"
                if not c and sb - sa > 9 and "cancel" in ctx.tracks:
                    # with a canceller track, long noisy turns can hide short replies from a quieter voice;
                    # shorter windows catch them. Without one, splitting only costs Whisper context.
                    sigs = [ctx.tracks[m] for m in ("cancel", self.noisy_src) if m in ctx.tracks]
                    out += [Piece(x0, x1, label, c) for x0, x1 in segment.pause_cuts(sigs, ctx.sr, sa, sb, 4.0, 8.0)]
                elif sb - sa > 25:  # Whisper takes at most 30 s at a time
                    sig = ctx.tracks[self.src_for(c)]
                    out += [Piece(x0, x1, label, c) for x0, x1 in segment.pause_cuts([sig], ctx.sr, sa, sb, 8.0, 20.0)]
                else:
                    out.append(Piece(sa, sb, label, c))
        return out

    # --- voice prints

    def _voiceprints(self, pieces: list[Piece]) -> dict[str, np.ndarray]:
        prints: dict[str, np.ndarray] = {}
        for label in sorted({p.label for p in pieces if not p.label.endswith("-1")}):
            segs = [self.ctx.audio(self.src_for(p.clean), p.start, p.end)
                    for p in pieces if p.label == label and p.end - p.start >= 1.0]
            if segs:
                prints[label] = self.diar.embedding(np.concatenate(segs)[: 60 * self.ctx.sr], self.ctx.sr)
        return prints

    def fold_small(self, pieces: list[Piece], min_s: float = 5.0, min_share: float = 0.08) -> list[Piece]:
        """Relabel pieces of 'speakers' with little total speech to the nearest speaker with plenty of it."""
        for part in (True, False):  # clean part, then noisy part
            sel = [p for p in pieces if p.clean == part and not p.label.endswith("-1")]
            total = sum(p.end - p.start for p in sel)
            dur: dict[str, float] = {}
            for p in sel:
                dur[p.label] = dur.get(p.label, 0.0) + p.end - p.start
            big = [k for k, d in dur.items() if d >= max(min_s, min_share * total)]
            small = [k for k in dur if k not in big]
            if not small or not big:
                continue
            prints = self._voiceprints([p for p in sel if p.label in big])
            for p in pieces:
                if p.clean == part and p.label in small and prints:
                    v = self.diar.embedding(self.ctx.audio(self.src_for(part), p.start, p.end), self.ctx.sr)
                    p.label = max(prints, key=lambda k: float(v @ prints[k]))
            self.ctx.report.setdefault("folded_speakers", []).extend(small)
        return pieces

    def match(self, pieces: list[Piece]) -> None:
        """Pair each noisy-part voice with a different clean-part voice, maximizing total similarity."""
        from scipy.optimize import linear_sum_assignment

        self.prints = self._voiceprints(pieces)
        cl = [k for k in self.prints if k.startswith("C")]
        ny = [k for k in self.prints if k.startswith("N")]
        self.ctx.report["speakers_found"] = {"clean_part": len(cl), "noisy_part": len(ny)}
        if not (cl and ny):
            return
        sim = np.array([[float(self.prints[n] @ self.prints[c]) for c in cl] for n in ny])
        rows, cols = linear_sum_assignment(-sim)
        match = {}
        for r, c in zip(rows, cols):
            self.mapping[ny[r]] = cl[c]
            others = np.delete(sim[r], c)
            margin = float(sim[r, c] - (others.max() if others.size else -1.0))
            self.sure[ny[r]] = margin >= 0.1
            match[ny[r]] = {"to": cl[c], "similarity": round(float(sim[r, c]), 3), "margin": round(margin, 3)}
        self.ctx.report["voice_match"] = match

    def who(self, label: str) -> str:
        return self.mapping.get(label, label)

    def voice_of(self, x: np.ndarray, spk: str, margin: float = 0.05) -> str | None:
        """The known speaker this audio sounds like, if it is clearly someone other than `spk`."""
        if not self.prints or len(x) < self.ctx.sr:
            return None
        v = self.diar.embedding(x, self.ctx.sr)
        sims: dict[str, float] = {}
        for label, p in self.prints.items():
            w = self.who(label)
            sims[w] = max(sims.get(w, -1.0), float(v @ p))
        ranked = sorted(sims.items(), key=lambda kv: -kv[1])
        best = ranked[0][0]
        if best == spk:
            return None
        return best if ranked[0][1] - sims.get(spk, -1.0) >= margin else None


def name_speakers(lines: list[Line], names) -> None:
    """Speaker 1 = whoever talks first. --names sets the names in that order."""
    order: list[str] = []
    for ln in lines:
        if ln.speaker not in order:
            order.append(ln.speaker)
    labels = list(names) if names else []
    mapping = {s: labels[i] if i < len(labels) else f"Speaker {i + 1}" for i, s in enumerate(order)}
    for ln in lines:
        ln.speaker = mapping.get(ln.speaker, ln.speaker)
