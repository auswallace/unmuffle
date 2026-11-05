"""Decide which processing came out best for a chunk, without knowing the right answer.

Two kinds of evidence:

1. How the audio sounds. DNSMOS (Microsoft, from the DNS Challenge) predicts
   the ratings a listening panel would give, with no clean reference needed:
     SIG  how natural and undistorted the voice is (1 to 5)
     BAK  how quiet the background is (1 to 5)
     OVRL overall quality (1 to 5)
2. What Whisper made of it. Noise makes Whisper loop, say filler, or return
   few words. A candidate that yields far fewer words than the best one, or
   that loops, loses points.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import models, text

DNSMOS_LEN = 144160  # 9.01 s at 16 kHz, the model's fixed input
_P_SIG = np.poly1d([-0.08397278, 1.22083953, 0.0052439])
_P_BAK = np.poly1d([-0.13166888, 1.60915514, -0.39604546])
_P_OVR = np.poly1d([-0.06766283, 1.11546468, 0.04602535])

CLEAN_OVRL = 3.0  # raw audio at or above these is treated as clean: no processing needed
CLEAN_BAK = 3.5


class DNSMOS:
    def __init__(self, threads: int = 4):
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        self._s = ort.InferenceSession(str(models.dnsmos_file()), opts, providers=["CPUExecutionProvider"])
        self._in = self._s.get_inputs()[0].name

    def __call__(self, x: np.ndarray) -> dict[str, float]:
        """Score 16 kHz mono audio. Short audio is repeated to fill the model's 9 s window."""
        x = np.asarray(x, dtype=np.float32)
        if x.size == 0 or float(np.max(np.abs(x))) < 1e-6:
            return {"sig": 1.0, "bak": 1.0, "ovrl": 1.0}
        while len(x) < DNSMOS_LEN:
            x = np.concatenate([x, x])
        starts = range(0, max(1, len(x) - DNSMOS_LEN + 1), 16000) if len(x) > DNSMOS_LEN else [0]
        raw = np.array([self._s.run(None, {self._in: x[a:a + DNSMOS_LEN][None, :]})[0][0] for a in starts])
        sig, bak, ovr = raw.mean(axis=0)
        return {"sig": round(float(_P_SIG(sig)), 3), "bak": round(float(_P_BAK(bak)), 3),
                "ovrl": round(float(_P_OVR(ovr)), 3)}


@dataclass
class Candidate:
    name: str
    mos: dict
    raw_text: str = ""
    text: str = ""
    score: float = 0.0
    notes: list[str] = field(default_factory=list)


def is_clean(mos: dict) -> bool:
    return mos["ovrl"] >= CLEAN_OVRL and mos["bak"] >= CLEAN_BAK


def score(cands: list[Candidate]) -> None:
    """Final score = DNSMOS overall + up to 1 point for word count relative to the best candidate − penalties."""
    words = {c.name: len(text.tokens(c.text)) for c in cands}
    most = max(words.values()) if words else 0
    for c in cands:
        s = c.mos["ovrl"]
        if most:
            s += words[c.name] / most
        raw_n, clean_n = len(c.raw_text.split()), len(c.text.split())
        if raw_n >= 6 and clean_n < 0.6 * raw_n:
            s -= 0.5
            c.notes.append("looped")
        if most >= 4 and words[c.name] == 0:
            c.notes.append("heard nothing")
        c.score = round(s, 3)


def pick(cands: list[Candidate], previous: str | None, hold: float = 0.15) -> Candidate:
    """Best score wins, but keep the previous chunk's choice if it is within `hold` (avoids audible flip-flopping)."""
    best = max(cands, key=lambda c: c.score)
    if previous:
        prev = next((c for c in cands if c.name == previous), None)
        if prev and best.score - prev.score <= hold and prev.text:
            return prev
    return best
