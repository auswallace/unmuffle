import numpy as np
import pytest

from unmuffle.tracks import enhance_chunked as _enhance_chunked

SR = 16000


@pytest.mark.parametrize("seconds", [0.5, 30, 60, 60.5, 60.9, 61.0, 61.5, 125.3])
def test_chunked_identity_denoiser_returns_input(tmp_path, seconds):
    x = np.random.default_rng(0).standard_normal(int(seconds * SR)).astype(np.float32)
    y = _enhance_chunked(lambda s, sr: s.copy(), x, SR, tmp_path, "t")
    assert y.shape == x.shape
    assert np.allclose(y, x, atol=1e-5)


def test_ffmpeg_filter_keeps_sample_rate_and_length():
    from unmuffle import audio
    x = (np.sin(np.arange(SR * 3) / SR * 2 * np.pi * 440) * 0.1).astype(np.float32)
    y = audio.ffmpeg_filter(x, SR, "loudnorm=I=-18:TP=-1.5:LRA=11")
    assert len(y) == len(x)
    # same pitch after filtering: the dominant frequency stays at 440 Hz
    f = np.argmax(np.abs(np.fft.rfft(y))) * SR / len(y)
    assert abs(f - 440) < 2
