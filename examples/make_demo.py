"""Build a synthetic covered-mic recording with a known transcript.

Two text-to-speech voices read a made-up conversation. At POCKET_AT seconds
the phone goes into Sam's pocket:

  - a loud, steady rumble starts. It reaches both mics almost the same way.
  - Sam (the phone's owner) is now heard through fabric: muffled, quiet, and
    much stronger on one mic than the other.
  - Alex (across the table) gets duller but still reaches both mics about equally.

That is the situation unmuffle was built for. Output: demo.m4a (stereo) and
demo_script.json (who said what, when).

Needs two Piper voices from the sherpa-onnx tts-models release:
  vits-piper-en_US-kristin-medium  (Sam)
  vits-piper-en_US-norman-medium   (Alex)

  python examples/make_demo.py --voices /path/to/voices --out examples/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.signal import butter, lfilter, resample_poly, sosfilt

SR = 16000
POCKET_AT = 20.0
SEED = 4

DIALOGUE = [
    ("Sam", "Hey Alex, do you have a minute to talk about the market booth?"),
    ("Alex", "Sure. What's going on with it?"),
    ("Sam", "The city moved us from the corner spot to the back row by the parking lot."),
    ("Alex", "Since when? Nobody told me about that."),
    ("Sam", "The email came on Tuesday. I think it only went to the board."),
    ("Alex", "Okay. So what do you want to do about it?"),
    ("Sam", "I want to ask for the corner back, because most of our sales come from people walking past."),
    ("Alex", "That makes sense, but the permit office closes at four on Fridays."),
    ("Sam", "Then I can call them tomorrow morning before my shift starts."),
    ("Alex", "Ask them about the power outlet too. The freezer needs one."),
    ("Sam", "Good point. The ice cream sold out by noon last time because the freezer kept tripping."),
    ("Alex", "Right. And if they say no, we could share the corner with the bakery."),
    ("Sam", "Maria already told me she would split the space and the cost with us."),
    ("Alex", "Great. Send me what they say, and I will update the schedule."),
    ("Sam", "Will do. Thanks for sorting this out with me."),
]


def tts(model_dir: Path, text: str, sid: int = 0, speed: float = 1.0) -> np.ndarray:
    import sherpa_onnx as so

    onnx = next(model_dir.glob("*.onnx"))
    cfg = so.OfflineTtsConfig(model=so.OfflineTtsModelConfig(
        vits=so.OfflineTtsVitsModelConfig(model=str(onnx), tokens=str(model_dir / "tokens.txt"),
                                          data_dir=str(model_dir / "espeak-ng-data")), num_threads=4))
    audio = so.OfflineTts(cfg).generate(text, sid=sid, speed=speed)
    y = np.asarray(audio.samples, dtype=np.float32)
    return resample_poly(y, SR, audio.sample_rate).astype(np.float32) if audio.sample_rate != SR else y


def lowpass(x, hz, order=4):
    if len(x) == 0:
        return x
    return sosfilt(butter(order, hz, "lowpass", fs=SR, output="sos"), x)


def build(voices: Path, out: Path, keep_wav: bool = False, name: str = "demo") -> None:
    rng = np.random.default_rng(SEED)
    dirs = {"Sam": voices / "vits-piper-en_US-kristin-medium", "Alex": voices / "vits-piper-en_US-norman-medium"}
    clips, script, t = [], [], 0.6
    for who, line in DIALOGUE:
        y = tts(dirs[who], line, speed=1.05 if who == "Sam" else 1.0)
        y = y / (np.max(np.abs(y)) + 1e-9) * 0.5
        clips.append((who, t, y))
        script.append({"speaker": who, "start": round(t, 2), "end": round(t + len(y) / SR, 2), "text": line})
        t += len(y) / SR + float(rng.uniform(0.35, 0.8))
    n = int((t + 1.0) * SR)
    left, right = np.zeros(n), np.zeros(n)
    cover = int(POCKET_AT * SR)

    def place(sig, start, gl, gr, delay_r=0):
        a = int(start * SR)
        left[a:a + len(sig)] += gl * sig[: n - a]
        b = a + delay_r
        right[b:b + len(sig)] += gr * sig[: max(0, n - b)]

    ref = np.zeros(n)  # the voices alone, as they sounded before the pocket: the target for cleaned audio
    for who, start, y in clips:
        a0 = int(start * SR)
        ref[a0:a0 + len(y)] += (0.925 if who == "Sam" else 0.6) * y[: n - a0]
        # split each line at the pocket moment
        cut = max(0, min(len(y), cover - int(start * SR)))
        before, after = y[:cut], y[cut:]
        if who == "Sam":   # near voice, on the phone's side
            place(before, start, 1.0, 0.85, 2)
            muff = lowpass(after, 850) * 0.45
            place(muff, start + cut / SR, 1.0, 0.0)                         # fabric side: mic A
            place(lowpass(muff, 400) * 0.12, start + cut / SR, 0.0, 1.0, 9)  # far side: mic B, faint
        else:              # far voice, across the table
            place(before, start, 0.6, 0.6, 1)
            dull = lowpass(after, 2400) * 0.75
            place(dull, start + cut / SR, 0.55, 0.53, 1)

    # rumble from the covered mic: brown noise below ~1.2 kHz, plus slow handling swells
    m = n - cover
    rumble = lfilter([1.0], [1.0, -0.985], rng.standard_normal(m))
    rumble = lowpass(rumble, 1200, 2)
    rumble *= 1.0 + 0.35 * np.sin(2 * np.pi * 0.23 * np.arange(m) / SR)
    sam_rms = np.sqrt(np.mean(left[cover:] ** 2)) + 1e-9
    rumble *= 3.2 * sam_rms / (np.sqrt(np.mean(rumble ** 2)) + 1e-9)
    left[cover:] += rumble
    right[cover:] += lfilter([0.91, 0.07, -0.025], [1.0], rumble)   # slightly different path to mic B

    # a little independent hiss on each mic so it is not too clean
    left += rng.standard_normal(n) * 0.0008
    right += rng.standard_normal(n) * 0.0008
    stereo = np.stack([left, right], axis=1)
    peak = np.max(np.abs(stereo))
    stereo = (stereo / peak * 0.9).astype(np.float32)

    out.mkdir(parents=True, exist_ok=True)
    import subprocess

    import soundfile as sf
    wav = out / f"{name}.wav"
    sf.write(wav, stereo, SR)
    sf.write(out / f"{name}_clean.wav", (ref / peak * 0.9).astype(np.float32), SR)
    (out / f"{name}_script.json").write_text(json.dumps({"pocket_at": POCKET_AT, "lines": script}, indent=2))
    if not keep_wav:
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(wav), "-c:a", "aac", "-b:a", "128k",
                        str(out / f"{name}.m4a")], check=True)
        wav.unlink()
    print(f"wrote {name} ({n / SR:.1f} s, pocket at {POCKET_AT} s)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--voices", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path(__file__).parent)
    a = ap.parse_args()
    build(a.voices, a.out)
