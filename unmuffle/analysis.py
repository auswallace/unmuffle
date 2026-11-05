"""File-level facts: background noise floor, and where it jumps (a covered mic)."""

from __future__ import annotations

import numpy as np


def noise_floor(mono: np.ndarray, sr: int, frame_s: float = 0.1, span_s: float = 1.0) -> np.ndarray:
    """Low-band (60 to 600 Hz) noise floor per 100 ms, in dB: the quietest frames around each moment."""
    from scipy.signal import butter, sosfilt
    low = sosfilt(butter(4, [60, 600], "bandpass", fs=sr, output="sos"), mono) ** 2
    hop = int(frame_s * sr)
    n = max(1, len(low) // hop)
    e = 10 * np.log10(low[: n * hop].reshape(n, hop).mean(axis=1) + 1e-12)
    half = max(1, int(span_s / frame_s))
    return np.array([np.percentile(e[max(0, i - half):i + half + 1], 10) for i in range(n)])


def analyze(x: np.ndarray, sr: int) -> dict:
    """Duration, plus the biggest lasting jump in background noise, if any (6 dB or more over 5 s)."""
    mono = x.mean(axis=1)
    fl = noise_floor(mono, sr)
    w = 50
    jump = None
    if len(fl) > 2 * w + 1:
        rises = np.array([np.median(fl[i:i + w]) - np.median(fl[i - w:i]) for i in range(w, len(fl) - w)])
        k = int(np.argmax(rises))
        best = float(rises[k])
        if best >= 6:
            at = k + w
            after = np.median(fl[at:at + w])
            # first moment the floor reaches within 10 dB of its new level and stays there for 2 s
            first = next((j for j in range(at - w, at + w) if np.min(fl[j:j + 20]) > after - 10), at)
            jump = {"at_seconds": round(first * 0.1, 1), "rise_db": round(best, 1)}
    return {"duration_s": round(len(mono) / sr, 2), "loudness_jump": jump}
