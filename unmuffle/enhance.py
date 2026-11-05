"""Voice isolation with sherpa-onnx speech denoisers (DPDFNet or GTCRN)."""

from __future__ import annotations

import numpy as np

from . import models


class Denoiser:
    def __init__(self, kind: str = "dpdfnet", threads: int = 4):
        import sherpa_onnx as so

        self.kind = kind
        if kind == "none":
            self._d = None
            return
        path = str(models.denoiser_file(kind))
        if kind == "dpdfnet":
            mc = so.OfflineSpeechDenoiserModelConfig(
                dpdfnet=so.OfflineSpeechDenoiserDpdfNetModelConfig(model=path), num_threads=threads)
        else:
            mc = so.OfflineSpeechDenoiserModelConfig(
                gtcrn=so.OfflineSpeechDenoiserGtcrnModelConfig(model=path), num_threads=threads)
        self._d = so.OfflineSpeechDenoiser(so.OfflineSpeechDenoiserConfig(model=mc))

    def __call__(self, x: np.ndarray, sr: int) -> np.ndarray:
        if self._d is None:
            return x.astype(np.float32)
        out = self._d(x.astype(np.float32), sr)
        y = np.asarray(out.samples, dtype=np.float32)
        if out.sample_rate != sr:
            from scipy.signal import resample_poly
            y = resample_poly(y, sr, out.sample_rate).astype(np.float32)
        return y[: len(x)] if len(y) >= len(x) else np.pad(y, (0, len(x) - len(y)))
