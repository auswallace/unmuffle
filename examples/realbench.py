"""Benchmark on real recordings: real speech mixed with real recorded noise.

Data: Microsoft MS-SNSD (https://github.com/microsoft/MS-SNSD, MIT license),
downloaded straight from GitHub. For each of its 9 test noise types (air
conditioner, airport announcements, babble, copy machine, munching, neighbor,
shutting door, typing, vacuum cleaner), this builds one file: 4 real speech
clips from clean_test with short pauses, plus one real noise recording from
noise_test at 0 dB SNR (noise as loud as speech).

MS-SNSD has no transcripts, so the reference is Whisper turbo on the clean
speech, clip by clip. WER then measures how much the noise costs relative to
clean audio, with the same yardstick for plain Whisper and unmuffle.

The second-model check is skipped (--check none): it only flags lines and does
not change words, and it doubles the run time.

  python examples/realbench.py bench/          # download, mix, run, score
"""

from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent))
from score import wer  # noqa: E402
from scorecard import align, si_sdr  # noqa: E402

from unmuffle import audio, judge, pipeline, text  # noqa: E402
from unmuffle.asr import Whisper  # noqa: E402

RAW = "https://raw.githubusercontent.com/microsoft/MS-SNSD/master"
SR = 16000
NOISES = ["AirConditioner", "AirportAnnouncements", "Babble", "CopyMachine", "Munching",
          "Neighbor", "ShuttingDoor", "Typing", "VacuumCleaner"]
SNR_DB = 0.0
CLIPS_PER_FILE = 4


def fetch(url: str, dest: Path) -> Path:
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url) as r:
            dest.write_bytes(r.read())
    return dest


def build(folder: Path) -> list[str]:
    data = folder / "ms-snsd"
    names = []
    for i, kind in enumerate(NOISES):
        name = f"{kind.lower()}"
        names.append(name)
        if (folder / f"{name}.wav").exists():
            continue
        clips = [audio.read_wav(fetch(f"{RAW}/clean_test/clnsp{10 * i + k}.wav", data / f"clnsp{10 * i + k}.wav"))
                 for k in range(CLIPS_PER_FILE)]
        noise = audio.read_wav(fetch(f"{RAW}/noise_test/{kind}_1.wav", data / f"{kind}_1.wav"))
        gap = np.zeros(int(0.7 * SR), dtype=np.float32)
        parts, spans, t = [gap], [], len(gap) / SR
        for c in clips:
            spans.append((round(t, 2), round(t + len(c) / SR, 2)))
            parts += [c, gap]
            t += (len(c) + len(gap)) / SR
        clean = np.concatenate(parts)
        reps = int(np.ceil(len(clean) / len(noise)))
        n = np.tile(noise, reps)[: len(clean)]
        speech_rms = np.sqrt(np.mean(clean[np.abs(clean) > 1e-4] ** 2))
        n = n * speech_rms / (np.sqrt(np.mean(n ** 2)) + 1e-12) / (10 ** (SNR_DB / 20))
        noisy = clean + n
        peak = np.max(np.abs(noisy))
        sf.write(folder / f"{name}.wav", (noisy / peak * 0.9).astype(np.float32), SR)
        sf.write(folder / f"{name}_clean.wav", (clean / peak * 0.9).astype(np.float32), SR)
        (folder / f"{name}_spans.json").write_text(json.dumps(spans))
    return names


def reference(folder: Path, name: str, asr) -> str:
    path = folder / f"{name}_ref.txt"
    if not path.exists():
        clean = audio.read_wav(folder / f"{name}_clean.wav")
        spans = json.loads((folder / f"{name}_spans.json").read_text())
        path.write_text(" ".join(asr()(clean[int(a * SR):int(b * SR)], SR) for a, b in spans))
    return path.read_text()


def baseline(folder: Path, name: str, asr) -> str:
    path = folder / f"{name}_plain.txt"
    if not path.exists():
        x = audio.read_wav(folder / f"{name}.wav")
        path.write_text(" ".join(asr()(x[int(t * SR):int((t + 25) * SR)], SR) for t in np.arange(0, len(x) / SR, 25.0)))
    return path.read_text()


def main() -> None:
    folder = Path(sys.argv[1])
    folder.mkdir(parents=True, exist_ok=True)
    names = build(folder)
    cache: dict = {}

    def asr():
        if "w" not in cache:
            cache["w"] = Whisper("turbo")
        return cache["w"]

    mos = judge.DNSMOS()
    res_path = folder / "realbench.json"
    rows = {r["noise"]: r for r in json.loads(res_path.read_text())} if res_path.exists() else {}
    for name in names:
        ref = text.tokens(reference(folder, name, asr))
        plain = text.tokens(baseline(folder, name, asr))
        out = folder / f"{name}.unmuffle"
        if not (out / "report.json").exists():
            cache.clear()  # free Whisper before the pipeline loads its own
            pipeline.run(folder / f"{name}.wav", out, pipeline.Options(check_model=None))
        hyp = text.tokens(" ".join(ln["text"] for ln in json.loads((out / "transcript.json").read_text())))
        clean = audio.read_wav(folder / f"{name}_clean.wav")
        raw = audio.read_wav(folder / f"{name}.wav")
        cleaned = align(audio.read_wav(out / "cleaned.wav"), clean)
        rep = json.loads((out / "report.json").read_text())
        rows[name] = {
            "noise": name,
            "wer_plain": round(wer(ref, plain), 3), "wer_unmuffle": round(wer(ref, hyp), 3),
            "si_sdr_raw": si_sdr(raw, clean), "si_sdr_cleaned": si_sdr(cleaned, clean),
            "dnsmos_raw": mos(raw)["ovrl"], "dnsmos_cleaned": mos(cleaned)["ovrl"],
            "methods_used": rep.get("methods_used"),
        }
        res_path.write_text(json.dumps([rows[n] for n in names if n in rows], indent=2))
        print(json.dumps(rows[name]), flush=True)
    write_md([rows[n] for n in names if n in rows], folder / "realbench.md")


def write_md(rows: list[dict], path: Path) -> None:
    lines = ["# Real-recording benchmark (MS-SNSD)", "",
             f"Real speech + real recorded noise at {SNR_DB:g} dB SNR, {CLIPS_PER_FILE} clips per file. "
             "Reference transcript = Whisper turbo on the clean speech.", "",
             "| Noise | WER plain | WER unmuffle | SI-SDR raw | SI-SDR cleaned | DNSMOS raw | DNSMOS cleaned |",
             "|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['noise']} | {r['wer_plain']:.2f} | {r['wer_unmuffle']:.2f} | {r['si_sdr_raw']:.1f} dB "
                     f"| {r['si_sdr_cleaned']:.1f} dB | {r['dnsmos_raw']:.2f} | {r['dnsmos_cleaned']:.2f} |")
    if rows:
        m = {k: float(np.mean([r[k] for r in rows])) for k in rows[0] if k not in ("noise", "methods_used")}
        lines.append(f"| **mean** | {m['wer_plain']:.2f} | {m['wer_unmuffle']:.2f} | {m['si_sdr_raw']:.1f} dB "
                     f"| {m['si_sdr_cleaned']:.1f} dB | {m['dnsmos_raw']:.2f} | {m['dnsmos_cleaned']:.2f} |")
    path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
