"""Two-microphone adaptive noise cancelling.

Phones record with two mics. When a mic is covered, the rumble (handling noise,
fabric, body contact) reaches both mics in almost the same way, while the voices
reach them differently. For each frequency bin, and for each short block of
time, we learn the complex gain that best predicts the left channel from the
right channel. Because the rumble carries most of the energy, that prediction is
mostly rumble. Subtracting it cancels the rumble and leaves the part of the
voice that differs between the two mics.

    out(f, t) = L(f, t) - a(f, block) * R(f, t)
    a(f, block) = sum(L * conj(R)) / sum(|R|^2)   over the block and its neighbors
"""

from __future__ import annotations

import numpy as np
from scipy.signal import istft, stft


def adaptive_cancel(left: np.ndarray, right: np.ndarray, sr: int, nfft: int = 1024,
                    hop: int = 256, block_s: float = 2.0) -> np.ndarray:
    """Return the left channel with the part predictable from the right channel removed."""
    if left.shape != right.shape:
        raise ValueError("left and right must have the same length")
    noverlap = nfft - hop
    _, _, sl = stft(left, sr, nperseg=nfft, noverlap=noverlap)
    _, _, sr_ = stft(right, sr, nperseg=nfft, noverlap=noverlap)
    frames = sl.shape[1]
    block = max(1, int(round(block_s * sr / hop)))
    out = np.empty_like(sl)
    for s in range(0, frames, block):
        e = min(frames, s + block)
        a0, a1 = max(0, s - block), min(frames, e + block)  # learn from the block and its neighbors
        num = np.sum(sl[:, a0:a1] * np.conj(sr_[:, a0:a1]), axis=1)
        den = np.sum(np.abs(sr_[:, a0:a1]) ** 2, axis=1) + 1e-12
        gain = num / den
        out[:, s:e] = sl[:, s:e] - gain[:, None] * sr_[:, s:e]
    _, y = istft(out, sr, nperseg=nfft, noverlap=noverlap)
    y = y[: len(left)]
    if len(y) < len(left):
        y = np.pad(y, (0, len(left) - len(y)))
    return y.astype(np.float32)


def channel_report(left: np.ndarray, right: np.ndarray) -> dict:
    """How similar the two channels are. Near 1.0 with a real stereo file means the trick can help."""
    if left.size == 0:
        return {"correlation": 0.0, "identical": True}
    corr = float(np.corrcoef(left, right)[0, 1]) if np.std(left) > 0 and np.std(right) > 0 else 1.0
    identical = bool(np.max(np.abs(left - right)) < 1e-6)
    return {"correlation": round(corr, 4), "identical": identical}
