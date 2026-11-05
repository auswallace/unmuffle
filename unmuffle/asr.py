"""Whisper speech recognition through sherpa-onnx (CPU, int8)."""

from __future__ import annotations

import numpy as np

from . import models


class Whisper:
    def __init__(self, name: str = "turbo", threads: int = 4):
        import sherpa_onnx as so

        f = models.whisper_files(name)
        self.name = name
        self._r = so.OfflineRecognizer.from_whisper(
            encoder=str(f["encoder"]), decoder=str(f["decoder"]), tokens=str(f["tokens"]),
            language="en", task="transcribe", num_threads=threads, tail_paddings=800)

    def __call__(self, x: np.ndarray, sr: int) -> str:
        s = self._r.create_stream()
        s.accept_waveform(sr, x.astype(np.float32))
        self._r.decode_stream(s)
        return s.result.text.strip()
