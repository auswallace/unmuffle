import numpy as np

from unmuffle import judge, render
from unmuffle.decide import second_voice as _second_voice
from unmuffle.speakers import turns_in as _turns_in

SR = 16000


def _cand(name, ovrl, txt, raw=None):
    return judge.Candidate(name, {"sig": ovrl, "bak": ovrl, "ovrl": ovrl}, raw or txt, txt)


def test_more_words_beats_slightly_better_sound():
    a = _cand("dpdfnet", 2.6, "so")
    b = _cand("cancel", 2.4, "Can you send me the new schedule for the booth before Friday?")
    judge.score([a, b])
    assert judge.pick([a, b], None).name == "cancel"


def test_loop_is_penalized():
    a = _cand("gtcrn", 2.5, "I was like, done", raw="I was like, I was like, I was like, I was like, done")
    b = _cand("dpdfnet", 2.5, "I was looking at the calendar and saw the meeting.")
    judge.score([a, b])
    assert "looped" in a.notes and b.score > a.score


def test_pick_holds_previous_choice_when_close():
    a, b = _cand("dpdfnet", 2.50, "one two three four"), _cand("cancel", 2.55, "one two three four")
    judge.score([a, b])
    assert judge.pick([a, b], "dpdfnet").name == "dpdfnet"


def test_second_voice_found_only_for_different_words():
    win = _cand("dpdfnet", 2.6, "The freezer needs its own outlet. Ask the permit office.")
    other = _cand("cancel", 2.3, "Maria said she would split the corner space with us.")
    dup = _cand("gtcrn", 2.4, "The freezer needs its own outlet.")
    judge.score([win, other, dup])
    assert _second_voice(win, [win, other, dup]) is other


def test_clean_threshold():
    assert judge.is_clean({"sig": 3.8, "bak": 4.0, "ovrl": 3.3})
    assert not judge.is_clean({"sig": 3.0, "bak": 2.1, "ovrl": 2.4})


def test_turns_clipped_and_tiny_pieces_merged():
    turns = [(0.0, 5.0, 0), (5.0, 5.4, 1), (5.4, 12.0, 0), (12.0, 20.0, 1)]
    parts = _turns_in(turns, 3.0, 15.0)
    assert parts[0][0] == 3.0 and parts[-1][1] == 15.0
    assert [p[2] for p in parts] == [0, 1]


def test_cleaned_render_switches_tracks_without_gaps():
    n = SR * 4
    tracks = {"raw": np.ones(n, dtype=np.float32) * 0.1, "dpdfnet": np.ones(n, dtype=np.float32) * 0.1}
    chunks = [{"start": 0.0, "end": 2.0, "use": ["raw"]}, {"start": 2.0, "end": 4.0, "use": ["dpdfnet"]}]
    y = render.cleaned(tracks, chunks, SR)
    assert y.shape == (n,)
    assert np.min(np.abs(y[100:-100])) > 0.05  # no hole at the join
