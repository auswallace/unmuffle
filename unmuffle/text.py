"""Clean Whisper output and compare transcripts."""

from __future__ import annotations

import re
from difflib import SequenceMatcher

# What Whisper says when it hears only noise or silence.
FILLER = {"", "so", "you", "and", "the", "a", "um", "uh", "okay so", "thank you", "thanks", "thanks for watching",
          "thank you for watching", "bye", "yeah so", "hello", "music", "applause", "silence", "oh", "off"}
_ANNOTATION = re.compile(r"\[[^\]]*\]|\([^)]*\)|\*[^*]*\*")
_ELLIPSIS = re.compile(r"(\.\s*){2,}")


def norm(word: str) -> str:
    return re.sub(r"[^a-z0-9']", "", word.lower())


def tokens(text: str) -> list[str]:
    return [t for t in (norm(w) for w in text.split()) if t]


def collapse_loops(words: list[str], max_n: int = 6, min_repeats: int = 3) -> list[str]:
    """Remove the 'I was like, I was like, I was like...' loops Whisper makes on noise."""
    out = list(words)
    changed = True
    while changed:
        changed = False
        for n in range(1, max_n + 1):
            i = 0
            while i + n * min_repeats <= len(out):
                gram = [norm(w) for w in out[i:i + n]]
                reps = 1
                while out[i + reps * n:i + (reps + 1) * n] and \
                        [norm(w) for w in out[i + reps * n:i + (reps + 1) * n]] == gram:
                    reps += 1
                if reps >= min_repeats:
                    del out[i + n:i + reps * n]
                    changed = True
                i += 1
    return out


def clean(text: str) -> str:
    text = _ANNOTATION.sub(" ", text)
    text = _ELLIPSIS.sub(" ", text)
    text = " ".join(collapse_loops(text.split())).strip(" -,")
    return "" if " ".join(tokens(text)) in FILLER else text


def similarity(a: str, b: str) -> float:
    ta, tb = tokens(a), tokens(b)
    if not ta or not tb:
        return 0.0
    return SequenceMatcher(a=ta, b=tb, autojunk=False).ratio()


def containment(small: str, big: str) -> float:
    """Share of the words in `small` that also appear, in order, in `big`."""
    ts, tb = tokens(small), tokens(big)
    if not ts:
        return 1.0
    matched = sum(b.size for b in SequenceMatcher(a=ts, b=tb, autojunk=False).get_matching_blocks())
    return matched / len(ts)
