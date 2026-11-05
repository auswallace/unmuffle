"""Where to cut the audio: at pauses (noisy part) or at speaker turns (clean part)."""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfilt

from . import models

FRAME = 0.05  # seconds


def speech_envelope(x: np.ndarray, sr: int) -> np.ndarray:
    """Speech-band energy per 50 ms frame, lightly smoothed."""
    sos = butter(4, [250, 3400], "bandpass", fs=sr, output="sos")
    b = sosfilt(sos, x) ** 2
    hop = int(FRAME * sr)
    n = max(1, len(b) // hop)
    e = b[: n * hop].reshape(n, hop).mean(axis=1)
    return np.convolve(e, np.ones(8) / 8, mode="same")


def pause_cuts(signals: list[np.ndarray], sr: int, start: float, end: float,
               min_len: float = 6.0, max_len: float = 14.0) -> list[tuple[float, float]]:
    """Split [start, end) into chunks of min_len..max_len seconds, cutting at the quietest moment."""
    env = sum(speech_envelope(s, sr) / (np.median(speech_envelope(s, sr)) + 1e-12) for s in signals)
    cuts = [start]
    while cuts[-1] < end - max_len:
        lo, hi = int((cuts[-1] + min_len) / FRAME), int((cuts[-1] + max_len) / FRAME)
        cuts.append((lo + int(np.argmin(env[lo:hi]))) * FRAME)
    cuts.append(end)
    return [(a, b) for a, b in zip(cuts[:-1], cuts[1:]) if b - a > 0.5]


class Diarizer:
    """pyannote segmentation + speaker embeddings via sherpa-onnx."""

    def __init__(self, num_speakers: int | None = None, threads: int = 4, threshold: float = 0.5):
        import sherpa_onnx as so

        self._so = so
        seg, emb = models.diarization_files()
        self.emb_path = str(emb)
        cfg = so.OfflineSpeakerDiarizationConfig(
            segmentation=so.OfflineSpeakerSegmentationModelConfig(
                pyannote=so.OfflineSpeakerSegmentationPyannoteModelConfig(model=str(seg)), num_threads=threads),
            embedding=so.SpeakerEmbeddingExtractorConfig(model=str(emb), num_threads=threads),
            clustering=so.FastClusteringConfig(num_clusters=num_speakers or -1, threshold=threshold),
            min_duration_on=0.3, min_duration_off=0.5)
        self._d = so.OfflineSpeakerDiarization(cfg)
        self._emb = so.SpeakerEmbeddingExtractor(
            so.SpeakerEmbeddingExtractorConfig(model=str(emb), num_threads=threads))

    def turns(self, x: np.ndarray, sr: int, offset: float = 0.0, max_len: float = 25.0,
              join_gap: float = 1.0) -> list[tuple[float, float, int]]:
        """Speaker turns as (start, end, speaker). Joins short gaps; splits very long turns."""
        if not np.any(x):
            return []
        segs = [(s.start, s.end, s.speaker) for s in self._d.process(x.astype(np.float32)).sort_by_start_time()]
        joined: list[list] = []
        for a, b, spk in segs:
            if joined and joined[-1][2] == spk and a - joined[-1][1] <= join_gap and b - joined[-1][0] <= max_len:
                joined[-1][1] = max(joined[-1][1], b)
            else:
                joined.append([a, b, spk])
        # make turns contiguous so no speech falls between them
        for i in range(len(joined) - 1):
            mid = (joined[i][1] + joined[i + 1][0]) / 2
            if joined[i + 1][0] > joined[i][1]:
                joined[i][1] = joined[i + 1][0] = mid
        if joined:
            joined[0][0] = 0.0
            joined[-1][1] = len(x) / sr
        return [(offset + a, offset + b, int(s)) for a, b, s in joined]

    def embedding(self, x: np.ndarray, sr: int) -> np.ndarray:
        st = self._emb.create_stream()
        st.accept_waveform(sr, x.astype(np.float32))
        st.input_finished()
        v = np.asarray(self._emb.compute(st), dtype=np.float32)
        return v / (np.linalg.norm(v) + 1e-9)
