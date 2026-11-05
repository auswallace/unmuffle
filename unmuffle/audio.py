"""Audio input and output through ffmpeg."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

SAMPLE_RATE = 16000

AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg", ".opus",
              ".mov", ".mp4", ".m4v", ".mkv", ".webm", ".3gp"}


def require_ffmpeg() -> str:
    path = shutil.which("ffmpeg")
    if not path:
        raise RuntimeError("ffmpeg is not installed. Install it (for example: brew install ffmpeg) and try again.")
    return path


def load(path: str | Path, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Decode any audio or video file to float32, shape (samples, channels).

    Keeps up to two channels. Mono files come back with one column.
    """
    ffmpeg = require_ffmpeg()
    probe = subprocess.run(
        [ffmpeg, "-v", "error", "-i", str(path), "-map", "0:a:0", "-ar", str(sr),
         "-f", "f32le", "-acodec", "pcm_f32le", "-"],
        capture_output=True, check=False,
    )
    if probe.returncode != 0:
        raise RuntimeError(f"ffmpeg could not read {path}: {probe.stderr.decode(errors='ignore').strip()}")
    channels = _channel_count(path)
    data = np.frombuffer(probe.stdout, dtype=np.float32)
    if channels > 1:
        data = data[: len(data) // channels * channels].reshape(-1, channels)[:, :2]
    else:
        data = data.reshape(-1, 1)
    return np.ascontiguousarray(data)


def _channel_count(path: str | Path) -> int:
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return 1
    out = subprocess.run(
        [ffprobe, "-v", "error", "-select_streams", "a:0", "-show_entries", "stream=channels",
         "-of", "default=nw=1:nk=1", str(path)],
        capture_output=True, text=True, check=False,
    ).stdout.strip()
    try:
        return max(1, int(out.splitlines()[0]))
    except (ValueError, IndexError):
        return 1


def normalize(x: np.ndarray, peak: float = 0.9) -> np.ndarray:
    m = float(np.max(np.abs(x))) if x.size else 0.0
    return (x / m * peak).astype(np.float32) if m > 0 else x.astype(np.float32)


def ffmpeg_filter(x: np.ndarray, sr: int, af: str) -> np.ndarray:
    """Run a mono float32 signal through an ffmpeg audio filter chain."""
    ffmpeg = require_ffmpeg()
    res = subprocess.run(
        [ffmpeg, "-v", "error", "-f", "f32le", "-ar", str(sr), "-ac", "1", "-i", "-",
         "-af", af, "-ar", str(sr), "-f", "f32le", "-acodec", "pcm_f32le", "-"],  # some filters (loudnorm) resample
        input=x.astype(np.float32).tobytes(), capture_output=True, check=False,
    )
    if res.returncode != 0:
        raise RuntimeError(f"ffmpeg filter failed: {res.stderr.decode(errors='ignore').strip()}")
    y = np.frombuffer(res.stdout, dtype=np.float32)
    return np.ascontiguousarray(y[: len(x)])


def level_match(x: np.ndarray, sr: int) -> np.ndarray:
    """Slow automatic gain (ffmpeg dynaudnorm) so quiet speech reaches the recognizer at a usable level."""
    return ffmpeg_filter(x, sr, "dynaudnorm=f=150:g=21:m=40")


def save_wav(path: str | Path, x: np.ndarray, sr: int = SAMPLE_RATE) -> None:
    sf.write(str(path), x.astype(np.float32), sr)


def save_m4a(path: str | Path, x: np.ndarray, sr: int = SAMPLE_RATE) -> None:
    """Write a small AAC file for listening. Falls back to WAV if encoding fails."""
    ffmpeg = require_ffmpeg()
    tmp = Path(path).with_suffix(".tmp.wav")
    save_wav(tmp, x, sr)
    res = subprocess.run([ffmpeg, "-y", "-v", "error", "-i", str(tmp), "-c:a", "aac", "-b:a", "96k", str(path)],
                         capture_output=True, check=False)
    if res.returncode == 0:
        tmp.unlink(missing_ok=True)
    else:
        tmp.rename(Path(path).with_suffix(".wav"))


def read_wav(path: str | Path) -> np.ndarray:
    x, _ = sf.read(str(path), dtype="float32")
    return x
