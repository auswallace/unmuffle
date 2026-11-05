"""Build listening files: one cleaned file with everyone, and one file per speaker.

Each chunk uses the audio of the method that won it. Where a chunk held two
voices on two different tracks, both tracks are mixed in. Joins are crossfaded
so method switches do not click, and the result is loudness-normalized.
"""

from __future__ import annotations

import numpy as np

from . import audio

FADE_S = 0.04


def _window(n: int, sr: int, a: float, b: float, fade_s: float = FADE_S) -> tuple[int, int, np.ndarray]:
    """A 0..1 weight over [a, b] with raised-cosine ramps that overlap the neighbors by `fade_s`."""
    f = int(fade_s * sr)
    i0, i1 = max(0, int(a * sr) - f // 2), min(n, int(b * sr) + f // 2)
    w = np.ones(i1 - i0, dtype=np.float32)
    k = min(f, len(w) // 2)
    if k > 1:
        ramp = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, k, dtype=np.float32))
        if i0 > 0:
            w[:k] = ramp
        if i1 < n:
            w[-k:] = ramp[::-1]
    return i0, i1, w


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x ** 2))) if x.size else 0.0


def cleaned(tracks: dict[str, np.ndarray], chunks: list[dict], sr: int) -> np.ndarray:
    """chunks: [{"start", "end", "use": [track names]}], contiguous and in order."""
    n = len(next(iter(tracks.values())))
    out = np.zeros(n, dtype=np.float32)
    # level each track to a common speech level so switching tracks does not jump in volume
    ref = np.median([_rms(t[t != 0]) for t in tracks.values() if _rms(t) > 0]) or 0.1
    gain = {k: (ref / _rms(v) if _rms(v) > 0 else 1.0) for k, v in tracks.items()}
    for c in chunks:
        i0, i1, w = _window(n, sr, c["start"], c["end"])
        use = c["use"]
        for name in use:
            out[i0:i1] += tracks[name][i0:i1] * gain[name] * w / (1.0 if len(use) == 1 else 0.8 * len(use))
    return out


def speaker_files(tracks: dict[str, np.ndarray], lines: list, sr: int) -> dict[str, np.ndarray]:
    """One full-length file per speaker: that speaker's lines from their source track, silence elsewhere."""
    n = len(next(iter(tracks.values())))
    ref = np.median([_rms(t) for t in tracks.values() if _rms(t) > 0]) or 0.1
    out: dict[str, np.ndarray] = {}
    for ln in lines:
        who = ln.speaker.replace(" [?]", "")
        src = tracks[ln.meta.get("source", "raw")]
        a, b = ln.meta.get("span", (ln.start, ln.end))
        i0, i1, w = _window(n, sr, a - 0.15, b + 0.15, fade_s=0.1)
        g = ref / (_rms(src) or ref)
        buf = out.setdefault(who, np.zeros(n, dtype=np.float32))
        wts = out.setdefault(who + "\0w", np.zeros(n, dtype=np.float32))
        # max of weights so overlapping spans of the same speaker do not double the level
        new_w = np.maximum(wts[i0:i1], w)
        buf[i0:i1] = src[i0:i1] * g * new_w
        wts[i0:i1] = new_w
    return {k: v for k, v in out.items() if not k.endswith("\0w")}


def finish(x: np.ndarray, sr: int) -> np.ndarray:
    """Loudness-normalize for listening (ffmpeg loudnorm, -18 LUFS) and keep peaks under 0 dBFS."""
    try:
        y = audio.ffmpeg_filter(x, sr, "loudnorm=I=-18:TP=-1.5:LRA=11")
    except RuntimeError:
        y = x
    return audio.normalize(y, 0.95) if np.max(np.abs(y)) > 0.95 else y.astype(np.float32)
