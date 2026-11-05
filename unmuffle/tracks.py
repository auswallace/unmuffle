"""Build one full-length track per cleaning method.

  raw      the original mix, untouched
  gtcrn    light voice isolation (GTCRN)
  dpdfnet  strong voice isolation (DPDFNet)
  cancel   two-mic noise canceller, then DPDFNet (stereo files only)
"""

from __future__ import annotations

import gc
from pathlib import Path

import numpy as np

from . import audio, cancel
from .log import log


def build_tracks(x: np.ndarray, methods: list[str], sr: int, work: Path, threads: int) -> dict[str, np.ndarray]:
    mix = x.mean(axis=1).astype(np.float32)
    tracks = {"raw": mix}
    for m in [m for m in methods if m != "raw"]:
        p = work / f"{m}.wav"
        if p.exists():
            tracks[m] = audio.read_wav(p)
            continue
        if m == "cancel":
            cpath = work / "cancelled_raw.wav"
            if cpath.exists():
                base = audio.read_wav(cpath)
            else:
                log("two-mic canceller")
                base = audio.normalize(cancel.adaptive_cancel(x[:, 0], x[:, 1], sr))
                audio.save_wav(cpath, base, sr)
            kind = "dpdfnet"
        else:
            base, kind = mix, m
        from .enhance import Denoiser
        den = Denoiser(kind, threads)
        log(f"building {m} track ({kind})")
        y = audio.level_match(audio.normalize(enhance_chunked(den, base, sr, work, m)), sr)
        del den
        gc.collect()
        audio.save_wav(p, y, sr)
        tracks[m] = y
    return tracks


def enhance_chunked(denoiser, sig: np.ndarray, sr: int, work: Path, name: str,
                    chunk_s: float = 60.0, overlap_s: float = 1.0) -> np.ndarray:
    """Denoise in chunks with a short crossfade. Each chunk is cached, so an interrupted run resumes."""
    n, chunk, ov = len(sig), int(chunk_s * sr), int(overlap_s * sr)
    out = np.zeros(n, dtype=np.float32)
    weight = np.zeros(n, dtype=np.float32)
    # the previous chunk already runs `ov` past its end, so drop a start that leaves nothing new
    starts = [a for a in range(0, n, chunk) if a == 0 or a + ov < n]
    for i, a in enumerate(starts):
        last = i == len(starts) - 1
        b = n if last else min(n, a + chunk + ov)
        cache = work / f"{name}_enh_{i:04d}.npy"
        if cache.exists():
            y = np.load(cache)
        else:
            y = denoiser(sig[a:b], sr)
            np.save(cache, y)
        w = np.ones(b - a, dtype=np.float32)
        fade = min(ov, b - a)
        if i > 0:
            w[:fade] = np.linspace(0, 1, fade)
        if not last:
            w[-fade:] = np.linspace(1, 0, fade)
        out[a:b] += y[: b - a] * w
        weight[a:b] += w
    return out / np.maximum(weight, 1e-6)
