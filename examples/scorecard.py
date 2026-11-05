"""Run plain Whisper and unmuffle on every scenario and write a scorecard.

Measures, per scenario:
  wer            word error rate of the transcript (lower is better)
  words found    share of each speaker's words that appear, in order
  speakers       share of script lines whose matching transcript line has the right speaker
  si_sdr_db      how close the audio is to the clean voices (higher is better), raw vs cleaned
  dnsmos         predicted listener rating of the audio (1 to 5), raw vs cleaned
  judge          for noisy turns: how often the judge picked the method that the clean
                 reference says was best, and how much quality its picks gave up

Runs resume: run it again after an interruption and finished steps are reused.

  python examples/scorecard.py scenarios/            # all scenarios
  python examples/scorecard.py scenarios/ cafe wind  # some
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from score import score  # noqa: E402

from unmuffle import audio, judge, pipeline, text  # noqa: E402
from unmuffle.asr import Whisper  # noqa: E402

SR = 16000
ORDER = ["pocket", "cafe", "wind", "hum", "traffic", "music"]


def si_sdr(est: np.ndarray, ref: np.ndarray) -> float:
    n = min(len(est), len(ref))
    est, ref = est[:n] - est[:n].mean(), ref[:n] - ref[:n].mean()
    a = float(est @ ref) / (float(ref @ ref) + 1e-12)
    t = a * ref
    e = est - t
    return round(float(10 * np.log10((t @ t + 1e-12) / (e @ e + 1e-12))), 2)


def align(est: np.ndarray, ref: np.ndarray, max_lag: int = 8000) -> np.ndarray:
    """Undo any small delay processing added, by cross-correlation over the first 20 s."""
    n = min(len(est), len(ref), 20 * SR)
    c = np.fft.irfft(np.fft.rfft(est[:n], 2 * n) * np.conj(np.fft.rfft(ref[:n], 2 * n)))
    c = np.concatenate([c[-max_lag:], c[: max_lag + 1]])
    lag = int(np.argmax(c)) - max_lag
    return np.roll(est, -lag) if lag else est


def baseline(path: Path, out: Path) -> Path:
    if out.exists():
        return out
    x = audio.load(path).mean(axis=1)
    asr = Whisper("turbo")
    lines = [{"start": t, "text": asr(x[int(t * SR):int((t + 25) * SR)], SR)} for t in np.arange(0, len(x) / SR, 25.0)]
    out.write_text(json.dumps(lines, indent=2))
    return out


def speaker_accuracy(script: dict, hyp_path: Path) -> float:
    """Share of script lines whose matching transcript line has the right speaker.

    Labels are matched up to renaming (a consistent swap counts as right): a file with
    no clean stretch has no way to know which voice is called what.
    """
    import itertools
    hyp = json.loads(hyp_path.read_text())
    pairs = []
    for r in script["lines"]:
        best, best_s = None, 0.0
        for h in hyp:
            overlap = min(r["end"], h["end"]) - max(r["start"], h["start"])
            s = text.containment(h["text"], r["text"]) if overlap > -1.0 else 0.0
            if s > best_s:
                best, best_s = h, s
        if best is not None and best_s >= 0.5:
            pairs.append((r["speaker"], best["speaker"].replace(" [?]", "")))
    if not pairs:
        return 0.0
    truth = sorted({t for t, _ in pairs})
    said = sorted({h for _, h in pairs})
    best = 0
    for perm in itertools.permutations(said, min(len(said), len(truth))):
        m = dict(zip(perm, truth))
        best = max(best, sum(m.get(h) == t for t, h in pairs))
    return round(best / len(pairs), 3)


def judge_vs_oracle(out: Path, ref: np.ndarray, raw: np.ndarray) -> dict:
    """For each noisy turn, compare the judge's pick with the method closest to the clean voices."""
    decisions = json.loads((out / "report.json").read_text()).get("decisions_for_scoring", [])
    tracks = {"raw": raw}
    for p in (out / "cache").glob("*.wav"):
        if p.stem in pipeline.METHODS:
            tracks[p.stem] = audio.read_wav(p)
    agree = n = 0
    lost = []
    for d in decisions:
        if d["use"] == ["raw"] and d.get("clean"):
            continue
        a, b = int(d["start"] * SR), int(d["end"] * SR)
        if b - a < SR:
            continue
        q = {m: si_sdr(t[a:b], ref[a:b]) for m, t in tracks.items() if m in d["mos"]}
        oracle = max(q, key=q.get)
        n += 1
        agree += d["use"][0] == oracle
        lost.append(q[oracle] - q[d["use"][0]])
    return {"turns": n, "agreement": round(agree / n, 3) if n else None,
            "median_db_given_up": round(float(np.median(lost)), 2) if lost else None}


def run_one(folder: Path, name: str, mos_model) -> dict:
    wav = folder / f"{name}.wav"
    script = json.loads((folder / f"{name}_script.json").read_text())
    ref = audio.read_wav(folder / f"{name}_clean.wav")
    x = audio.load(wav)
    raw = x.mean(axis=1)
    base = baseline(wav, folder / f"{name}_baseline.json")
    out = folder / f"{name}.unmuffle"
    if not (out / "report.json").exists():
        pipeline.run(wav, out, pipeline.Options(names=("Sam", "Alex")))
    cleaned = align(audio.read_wav(out / "cleaned.wav"), ref)
    rep = json.loads((out / "report.json").read_text())
    return {
        "scenario": name,
        "baseline": score(folder / f"{name}_script.json", base),
        "unmuffle": score(folder / f"{name}_script.json", out / "transcript.json"),
        "speakers": speaker_accuracy(script, out / "transcript.json"),
        "si_sdr_db": {"raw": si_sdr(raw, ref), "cleaned": si_sdr(cleaned, ref)},
        "dnsmos": {"raw": mos_model(raw)["ovrl"], "cleaned": mos_model(cleaned)["ovrl"]},
        "judge": judge_vs_oracle(out, ref, raw),
        "methods_used": rep.get("methods_used"),
        "voice_match": rep.get("voice_match"),
    }


def write_md(rows: list[dict], path: Path) -> None:
    def pct(v):
        return "" if v is None else f"{round(100 * v)}%"

    lines = ["# Scorecard", "",
             "Same script and voices in every scenario; only the noise changes. "
             "Plain Whisper = Whisper turbo on the mono mix.",
             "", "| Scenario | WER plain | WER unmuffle | Words found plain | Words found unmuffle | Speakers right "
             "| SI-SDR raw | SI-SDR cleaned | DNSMOS raw | DNSMOS cleaned | Judge agrees |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        b, u = r["baseline"], r["unmuffle"]
        bf = [v for k, v in b.items() if "found" in k and "before" not in k]
        uf = [v for k, v in u.items() if "found" in k and "before" not in k]
        lines.append(f"| {r['scenario']} | {b['wer']:.2f} | {u['wer']:.2f} | {pct(np.mean(bf))} | {pct(np.mean(uf))} "
                     f"| {pct(r['speakers'])} | {r['si_sdr_db']['raw']:.1f} dB | {r['si_sdr_db']['cleaned']:.1f} dB "
                     f"| {r['dnsmos']['raw']:.2f} | {r['dnsmos']['cleaned']:.2f} | {pct(r['judge']['agreement'])} |")
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    folder = Path(sys.argv[1])
    names = sys.argv[2:] or [n for n in ORDER if (folder / f"{n}.wav").exists()]
    mos_model = judge.DNSMOS()
    res_path = folder / "scorecard.json"
    rows = {r["scenario"]: r for r in json.loads(res_path.read_text())} if res_path.exists() else {}
    for name in names:
        rows[name] = run_one(folder, name, mos_model)
        res_path.write_text(json.dumps([rows[k] for k in ORDER if k in rows], indent=2))
        print(json.dumps(rows[name]), flush=True)
    write_md([rows[k] for k in ORDER if k in rows], folder / "scorecard.md")


if __name__ == "__main__":
    main()
