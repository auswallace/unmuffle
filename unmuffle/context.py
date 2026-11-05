"""State shared by every stage of one run, with cached transcription and scoring."""

from __future__ import annotations

import gc
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import judge
from .data import Options
from .log import log


class ResultCache:
    """One JSON line per keyed result (transcripts, scores), so interrupted runs resume."""

    def __init__(self, path: Path):
        self.path, self.data = path, {}
        if path.exists():
            for ln in path.read_text().splitlines():
                if ln.strip():
                    d = json.loads(ln)
                    self.data[d["k"]] = d["v"]

    def get(self, key):
        return self.data.get(json.dumps(key))

    def put(self, key, value):
        k = json.dumps(key)
        self.data[k] = value
        with open(self.path, "a") as f:
            f.write(json.dumps({"k": k, "v": value}) + "\n")


@dataclass
class Context:
    opts: Options
    sr: int
    work: Path
    tracks: dict[str, np.ndarray]
    methods: list[str]
    report: dict
    cache: ResultCache = field(init=False)
    _asr: dict = field(default_factory=dict, init=False)
    _mos: object = field(default=None, init=False)

    def __post_init__(self):
        self.cache = ResultCache(self.work / "results.jsonl")

    def audio(self, method: str, a: float, b: float) -> np.ndarray:
        return self.tracks[method][int(a * self.sr):int(b * self.sr)]

    def transcribe(self, model: str, method: str, a: float, b: float) -> str:
        """Whisper text for one method's audio over [a, b], cached. One large model is loaded at a time."""
        key = ["asr", model, method, round(a, 2), round(b, 2)]
        hit = self.cache.get(key)
        if hit is None:
            if model not in self._asr:
                from .asr import Whisper
                self._asr.clear()
                gc.collect()
                log(f"loading Whisper {model}")
                self._asr[model] = Whisper(model, self.opts.threads)
            hit = self._asr[model](self.audio(method, a, b), self.sr)
            self.cache.put(key, hit)
        return hit

    def mos(self, method: str, a: float, b: float) -> dict:
        """DNSMOS scores for one method's audio over [a, b], cached."""
        key = ["mos", method, round(a, 2), round(b, 2)]
        hit = self.cache.get(key)
        if hit is None:
            if self._mos is None:
                self._mos = judge.DNSMOS(self.opts.threads)
            hit = self._mos(self.audio(method, a, b))
            self.cache.put(key, hit)
        return hit

    def release(self) -> None:
        """Free the loaded models."""
        self._asr.clear()
        self._mos = None
        gc.collect()
