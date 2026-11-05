"""Score a transcript against demo_script.json.

Reports word error rate (WER) for the clean part and the covered part, and how
many of the muffled speaker's words came back after the pocket moment.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from unmuffle.text import containment, tokens


def wer(ref: list[str], hyp: list[str]) -> float:
    """Word error rate: (substitutions + deletions + insertions) / reference words."""
    d = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        prev, d[0] = d[0], i
        for j, h in enumerate(hyp, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (r != h))
    return d[len(hyp)] / max(1, len(ref))


def load_hyp(path: Path) -> list[dict]:
    data = json.loads(path.read_text())
    return [{"start": float(x["start"]), "text": x["text"], "speaker": x.get("speaker", "")} for x in data]


def score(script: Path, hyp_path: Path) -> dict:
    s = json.loads(script.read_text())
    cut = s.get("pocket_at")
    if cut is None:  # noise the whole time
        ref, hyp = s["lines"], load_hyp(hyp_path)
        hyp_text = " ".join(tokens(" ".join(l["text"] for l in sorted(hyp, key=lambda l: l["start"]))))
        out = {"wer": round(wer(tokens(" ".join(l["text"] for l in ref)), hyp_text.split()), 3)}
        for who in sorted({l["speaker"] for l in ref}):
            out[f"{who.lower()}_words_found"] = round(
                containment(" ".join(l["text"] for l in ref if l["speaker"] == who), hyp_text), 3)
        return out
    ref = s["lines"]
    hyp = load_hyp(hyp_path)

    def words(lines, lo, hi, who=None):
        out = []
        for l in lines:
            mid = l["start"]
            if lo <= mid < hi and (who is None or l.get("speaker") == who):
                out += tokens(l["text"])
        return out

    # Score against the whole transcript, so chunking that crosses the pocket moment is not penalized.
    hyp_all = words(sorted(hyp, key=lambda l: l["start"]), -1, 1e9)
    hyp_text = " ".join(hyp_all)
    sam_clean = " ".join(words(ref, 0, cut, "Sam"))
    sam_noisy = " ".join(words(ref, cut, 1e9, "Sam"))
    alex_noisy = " ".join(words(ref, cut, 1e9, "Alex"))
    return {
        "wer": round(wer(words(ref, 0, 1e9), hyp_all), 3),
        "sam_words_found_before_pocket": round(containment(sam_clean, hyp_text), 3),
        "sam_words_found_in_pocket": round(containment(sam_noisy, hyp_text), 3),
        "alex_words_found_in_pocket": round(containment(alex_noisy, hyp_text), 3),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("script", type=Path)
    ap.add_argument("transcript", type=Path, help="baseline JSON or unmuffle transcript.json")
    a = ap.parse_args()
    print(json.dumps(score(a.script, a.transcript), indent=2))
