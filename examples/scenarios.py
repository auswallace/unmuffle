"""Build a set of noisy test recordings with known transcripts and clean references.

Every scenario uses the same two voices and script as make_demo.py, placed at a
table in front of a two-mic phone, plus one kind of noise:

  pocket   the phone goes into Sam's pocket at 0:20 (same as demo.m4a)
  cafe     other people talking all around (babble), the whole time
  wind     gusty low rumble, the whole time
  hum      electrical hum plus a fan, the whole time
  traffic  cars passing every few seconds
  music    a music bed under the conversation

For each scenario this writes <name>.wav (stereo, 16 kHz), <name>_clean.wav
(the voices alone, mono) and <name>_script.json.

  python examples/scenarios.py --voices /path/to/voices --out scenarios/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import butter, lfilter, sosfilt

sys.path.insert(0, str(Path(__file__).parent))
from make_demo import DIALOGUE, SR, lowpass, tts  # noqa: E402
from make_demo import build as build_pocket

BABBLE = [
    "I think the train was late again this morning.",
    "Can you pass me the sugar, please?",
    "We should try the new place on Fifth Street.",
    "She said the meeting moved to Thursday afternoon.",
    "My brother is visiting next weekend with the kids.",
    "Do you know if they still take cash here?",
    "The game went into overtime last night.",
    "I left my umbrella at the office again.",
]


def _band(x, lo, hi, order=4):
    return sosfilt(butter(order, [lo, hi], "bandpass", fs=SR, output="sos"), x)


def _voices(voices: Path, rng) -> tuple[list, float]:
    dirs = {"Sam": voices / "vits-piper-en_US-kristin-medium", "Alex": voices / "vits-piper-en_US-norman-medium"}
    clips, t = [], 0.6
    for who, line in DIALOGUE:
        y = tts(dirs[who], line, speed=1.05 if who == "Sam" else 1.0)
        y = y / (np.max(np.abs(y)) + 1e-9) * 0.5
        clips.append((who, t, y, line))
        t += len(y) / SR + float(rng.uniform(0.35, 0.8))
    return clips, t + 1.0


def _place(n, clips):
    """Both speakers at the table: Sam near mic A, Alex across, slightly nearer mic B."""
    left, right, ref = np.zeros(n), np.zeros(n), np.zeros(n)
    for who, start, y, _ in clips:
        a = int(start * SR)
        gl, gr, d = (1.0, 0.8, 2) if who == "Sam" else (0.6, 0.66, 1)
        m = min(len(y), n - a - d)
        left[a:a + m] += gl * y[:m]
        right[a + d:a + d + m] += gr * y[:m]
        ref[a:a + m] += (gl + gr) / 2 * y[:m]
    return left, right, ref


def _noise(kind: str, n: int, rng, voices: Path) -> tuple[np.ndarray, np.ndarray]:
    t = np.arange(n) / SR
    if kind == "cafe":
        # babble: many voices from different directions, so the two mics get different mixtures
        nl, nr = np.zeros(n), np.zeros(n)
        dirs = [voices / "vits-piper-en_US-kristin-medium", voices / "vits-piper-en_US-norman-medium"]
        for i in range(14):
            y = tts(dirs[i % 2], BABBLE[i % len(BABBLE)], speed=float(rng.uniform(0.9, 1.15)))
            y = lowpass(y, float(rng.uniform(2500, 4500))) * float(rng.uniform(0.15, 0.35))
            for _ in range(2):
                a = int(rng.uniform(0, max(1, n - len(y))))
                m = min(len(y), n - a)
                gl, gr = rng.uniform(0.3, 1.0, 2)
                d = int(rng.integers(0, 12))
                nl[a:a + m] += gl * y[:m]
                nr[min(n, a + d):min(n, a + d + m)] += gr * y[: min(m, n - a - d)]
        nl += rng.standard_normal(n) * 0.004
        nr += rng.standard_normal(n) * 0.004
        return nl, nr
    if kind == "wind":
        base = lfilter([1.0], [1.0, -0.995], rng.standard_normal(n))
        gust = 0.4 + 0.6 * np.clip(lowpass(rng.standard_normal(n), 0.5, 2) * 40 + 0.5, 0, 1.5)
        w = lowpass(base, 700, 2) * gust
        other = lowpass(lfilter([1.0], [1.0, -0.995], rng.standard_normal(n)), 700, 2) * gust
        return w, 0.8 * w + 0.35 * other  # mostly shared, partly different at each mic
    if kind == "hum":
        hum = sum(np.sin(2 * np.pi * 60 * h * t + rng.uniform(0, 6)) / h for h in range(1, 10))
        fan = lowpass(lfilter([1.0], [1.0, -0.9], rng.standard_normal(n)), 2000, 2)
        x = hum * 0.6 + fan * 2.0
        return x, lfilter([0.93, 0.05], [1.0], x)
    if kind == "traffic":
        x = np.zeros(n)
        start = 1.0
        while start < n / SR - 2:
            dur = float(rng.uniform(2.5, 5.0))
            k = int(dur * SR)
            a = int(start * SR)
            env = np.sin(np.linspace(0, np.pi, k)) ** 2
            car = lowpass(lfilter([1.0], [1.0, -0.98], rng.standard_normal(k)), float(rng.uniform(600, 1500)), 2)
            x[a:a + k] += car[: n - a] * env[: n - a] * float(rng.uniform(0.6, 1.4))
            start += dur + float(rng.uniform(1.0, 5.0))
        return x, lfilter([0.9, 0.08], [1.0], np.roll(x, 3))
    if kind == "music":
        x = np.zeros(n)
        chords = [(220.0, 277.2, 329.6), (196.0, 246.9, 293.7), (174.6, 220.0, 261.6), (196.0, 246.9, 329.6)]
        for i, start in enumerate(np.arange(0, n / SR, 2.0)):
            a, k = int(start * SR), int(2.0 * SR)
            seg = np.arange(min(k, n - a)) / SR
            env = np.exp(-seg * 0.8)
            for f in chords[i % 4]:
                x[a:a + len(seg)] += env * (np.sin(2 * np.pi * f * seg) + 0.3 * np.sin(4 * np.pi * f * seg))
            for beat in (0.0, 0.5, 1.0, 1.5):  # a soft kick on each beat
                b = a + int(beat * SR)
                kk = min(int(0.12 * SR), n - b)
                if kk > 0:
                    x[b:b + kk] += np.sin(2 * np.pi * 60 * np.arange(kk) / SR) * np.exp(-np.arange(kk) / SR * 30) * 2
        return x, lfilter([0.85, 0.1], [1.0], x)
    raise ValueError(kind)


SNR_DB = {"cafe": 3.0, "wind": 0.0, "hum": 3.0, "traffic": 0.0, "music": 3.0}


def build_all(voices: Path, out: Path, only: list[str] | None = None) -> None:
    out.mkdir(parents=True, exist_ok=True)
    names = only or ["pocket", *SNR_DB]
    if "pocket" in names and not (out / "pocket.wav").exists():
        build_pocket(voices, out, keep_wav=True, name="pocket")
    rest = [k for k in names if k != "pocket" and not (out / f"{k}.wav").exists()]
    if not rest:
        return
    rng = np.random.default_rng(11)
    clips, total = _voices(voices, rng)
    n = int(total * SR)
    left, right, ref = _place(n, clips)
    script = {"pocket_at": None, "lines": [{"speaker": w, "start": round(s, 2), "end": round(s + len(y) / SR, 2),
                                            "text": line} for w, s, y, line in clips]}
    speech_rms = np.sqrt(np.mean(ref[ref != 0] ** 2))
    for kind in rest:
        nl, nr = _noise(kind, n, np.random.default_rng(hash(kind) % 2**32), voices)
        g = speech_rms / (np.sqrt(np.mean(nl ** 2)) + 1e-12) / (10 ** (SNR_DB[kind] / 20))
        st = np.stack([left + g * nl, right + g * nr], axis=1)
        peak = np.max(np.abs(st))
        sf.write(out / f"{kind}.wav", (st / peak * 0.9).astype(np.float32), SR)
        sf.write(out / f"{kind}_clean.wav", (ref / peak * 0.9).astype(np.float32), SR)
        meta = {**script, "noise": kind, "snr_db": SNR_DB[kind]}
        (out / f"{kind}_script.json").write_text(json.dumps(meta, indent=2))
        print(f"wrote {kind}.wav (SNR {SNR_DB[kind]} dB)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--voices", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("scenarios"))
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    build_all(a.voices, a.out, a.only)
