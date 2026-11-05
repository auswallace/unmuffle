import numpy as np

from unmuffle.analysis import analyze
from unmuffle.segment import pause_cuts
from unmuffle.text import clean, collapse_loops, containment, similarity

SR = 16000


def test_clean_drops_filler_and_annotations():
    assert clean("so") == ""
    assert clean("Thank you.") == ""
    assert clean("[wind blowing]") == ""
    assert clean("*Burps* Okay, let's start.") == "Okay, let's start."


def test_collapse_loops():
    words = ["I", "was", "like,", "I", "was", "like,", "I", "was", "like,", "I", "was", "like,", "done"]
    assert " ".join(collapse_loops(words)) == "I was like, done"


def test_containment_and_similarity():
    assert containment("the back row", "they moved us to the back row by the lot") == 1.0
    assert similarity("The freezer needs one outlet.", "The freezer needs one outlet") == 1.0
    assert similarity("Hello there", "Completely different words") == 0.0


def test_pause_cuts_land_in_silence():
    t = np.arange(SR * 40) / SR
    x = np.sin(2 * np.pi * 800 * t).astype(np.float32) * 0.3
    for gap in (9.0, 19.0, 29.0):  # silent half-second gaps
        x[int(gap * SR):int((gap + 0.5) * SR)] = 0
    cuts = pause_cuts([x], SR, 0.0, 40.0)
    inner = [b for _, b in cuts[:-1]]
    assert all(any(abs(c - (g + 0.25)) < 0.4 for g in (9.0, 19.0, 29.0)) for c in inner)


def test_analyze_finds_loudness_jump():
    rng = np.random.default_rng(2)
    x = np.concatenate([rng.standard_normal(SR * 30) * 0.01, rng.standard_normal(SR * 30) * 0.05])
    rep = analyze(np.stack([x, x], axis=1), SR)
    assert rep["loudness_jump"] and abs(rep["loudness_jump"]["at_seconds"] - 30) <= 1


def test_analyze_ignores_loud_speech_and_finds_rumble():
    from scipy.signal import lfilter
    rng = np.random.default_rng(3)
    n = SR * 60
    t = np.arange(n) / SR
    # bursty speech-like signal all the way through, loudest just before the cover
    speech = np.sin(2 * np.pi * 300 * t) * (np.sin(2 * np.pi * 2.5 * t) > 0.3) * 0.2
    speech[int(14 * SR):int(19 * SR)] *= 3
    rumble = lfilter([1.0], [1.0, -0.985], rng.standard_normal(n)) * 0.01
    rumble[: 20 * SR] = 0
    x = speech + rumble + rng.standard_normal(n) * 0.0005
    rep = analyze(np.stack([x, x], axis=1), SR)
    assert rep["loudness_jump"] and abs(rep["loudness_jump"]["at_seconds"] - 20) <= 1
