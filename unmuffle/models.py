"""Model registry and downloader.

All models come from the sherpa-onnx GitHub releases, which stay reachable on
networks that block Hugging Face. Files are cached in ~/.cache/unmuffle
(override with UNMUFFLE_MODELS).
"""

from __future__ import annotations

import os
import shutil
import tarfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

BASE = "https://github.com/k2-fsa/sherpa-onnx/releases/download"


@dataclass(frozen=True)
class WhisperModel:
    name: str
    archive: str
    folder: str
    prefix: str

    def files(self, root: Path) -> dict[str, Path]:
        d = root / self.folder
        return {
            "encoder": d / f"{self.prefix}-encoder.int8.onnx",
            "decoder": d / f"{self.prefix}-decoder.int8.onnx",
            "tokens": d / f"{self.prefix}-tokens.txt",
        }


WHISPER = {
    "turbo": WhisperModel("turbo", "asr-models/sherpa-onnx-whisper-turbo.tar.bz2",
                          "sherpa-onnx-whisper-turbo", "turbo"),
    "medium.en": WhisperModel("medium.en", "asr-models/sherpa-onnx-whisper-medium.en.tar.bz2",
                              "sherpa-onnx-whisper-medium.en", "medium.en"),
    "small.en": WhisperModel("small.en", "asr-models/sherpa-onnx-whisper-small.en.tar.bz2",
                             "sherpa-onnx-whisper-small.en", "small.en"),
}

DENOISERS = {
    "dpdfnet": "speech-enhancement-models/dpdfnet8.onnx",
    "gtcrn": "speech-enhancement-models/gtcrn_simple.onnx",
}


DIARIZATION = {
    "segmentation": ("speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2",
                     "sherpa-onnx-pyannote-segmentation-3-0/model.onnx"),
    "embedding": ("speaker-recongition-models/3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx",
                  "3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx"),
}


def models_dir() -> Path:
    root = Path(os.environ.get("UNMUFFLE_MODELS", Path.home() / ".cache" / "unmuffle"))
    root.mkdir(parents=True, exist_ok=True)
    return root


def _download(url: str, dest: Path) -> None:
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"  downloading {url}")
    with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f, length=1 << 20)
    tmp.rename(dest)


def whisper_files(name: str) -> dict[str, Path]:
    if name not in WHISPER:
        raise ValueError(f"unknown Whisper model {name!r}; choose from {', '.join(WHISPER)}")
    model = WHISPER[name]
    root = models_dir()
    files = model.files(root)
    if all(p.exists() for p in files.values()):
        return files
    archive = root / Path(model.archive).name
    if not archive.exists():
        _download(f"{BASE}/{model.archive}", archive)
    wanted = {f"{model.folder}/{p.name}" for p in files.values()}
    with tarfile.open(archive, "r:bz2") as tar:
        for member in tar:
            if member.name in wanted:
                try:
                    tar.extract(member, root, filter="data")
                except TypeError:  # Python without tar extraction filters
                    tar.extract(member, root)
    missing = [str(p) for p in files.values() if not p.exists()]
    if missing:
        raise RuntimeError(f"model archive is incomplete, delete {archive} and retry: missing {missing}")
    archive.unlink(missing_ok=True)
    return files


def diarization_files() -> tuple[Path, Path]:
    root = models_dir()
    seg_url, seg_rel = DIARIZATION["segmentation"]
    seg = root / seg_rel
    if not seg.exists():
        archive = root / Path(seg_url).name
        _download(f"{BASE}/{seg_url}", archive)
        with tarfile.open(archive, "r:bz2") as tar:
            member = tar.getmember(seg_rel)
            try:
                tar.extract(member, root, filter="data")
            except TypeError:
                tar.extract(member, root)
        archive.unlink(missing_ok=True)
    emb_url, emb_rel = DIARIZATION["embedding"]
    emb = root / emb_rel
    if not emb.exists():
        _download(f"{BASE}/{emb_url}", emb)
    return seg, emb


DNSMOS_URL = "https://github.com/microsoft/DNS-Challenge/raw/master/DNSMOS/DNSMOS/sig_bak_ovr.onnx"


def dnsmos_file() -> Path:
    dest = models_dir() / "dnsmos_sig_bak_ovr.onnx"
    if not dest.exists():
        _download(DNSMOS_URL, dest)
    return dest


def denoiser_file(name: str) -> Path:
    if name not in DENOISERS:
        raise ValueError(f"unknown denoiser {name!r}; choose from {', '.join(DENOISERS)}")
    dest = models_dir() / Path(DENOISERS[name]).name
    if not dest.exists():
        _download(f"{BASE}/{DENOISERS[name]}", dest)
    return dest
