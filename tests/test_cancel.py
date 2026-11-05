import numpy as np
from scipy.signal import lfilter

from unmuffle.cancel import adaptive_cancel, channel_report

SR = 16000


def _db(x):
    return 10 * np.log10(np.mean(x ** 2) + 1e-20)


def _scene(seconds=20, seed=0):
    rng = np.random.default_rng(seed)
    n = SR * seconds
    rumble = lfilter([1.0], [1.0, -0.97], rng.standard_normal(n)) * 0.02  # low, steady, loud
    # the rumble reaches mic B through a slightly different path
    rumble_b = lfilter([0.9, 0.08, -0.03], [1.0], rumble)
    t = np.arange(n) / SR
    voice = 0.05 * np.sin(2 * np.pi * 220 * t) * (np.sin(2 * np.pi * 3 * t) > 0)  # syllable-like bursts
    return rumble, rumble_b, voice


def test_cancels_rumble_by_at_least_20_db():
    rumble, rumble_b, _ = _scene()
    out = adaptive_cancel(rumble.astype(np.float32), rumble_b.astype(np.float32), SR)
    assert _db(rumble) - _db(out) > 20


def test_keeps_voice_that_differs_between_mics():
    rumble, rumble_b, voice = _scene()
    left = rumble + voice          # voice mostly on mic A
    right = rumble_b + 0.1 * voice
    out = adaptive_cancel(left.astype(np.float32), right.astype(np.float32), SR)
    noise_only = adaptive_cancel(rumble.astype(np.float32), rumble_b.astype(np.float32), SR)
    snr_in = _db(voice) - _db(rumble)
    snr_out = _db(out - noise_only) - _db(noise_only)
    assert snr_out - snr_in > 15


def test_plain_subtraction_is_worse_than_adaptive():
    rumble, rumble_b, _ = _scene()
    plain = rumble - rumble_b
    adaptive = adaptive_cancel(rumble.astype(np.float32), rumble_b.astype(np.float32), SR)
    assert _db(adaptive) < _db(plain) - 5


def test_channel_report_detects_fake_stereo():
    x = np.random.default_rng(1).standard_normal(SR).astype(np.float32)
    assert channel_report(x, x)["identical"]
    assert not channel_report(x, x * 0.5 + 0.01)["identical"]
