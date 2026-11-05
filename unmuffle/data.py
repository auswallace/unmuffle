"""Data shared across stages."""

from __future__ import annotations

from dataclasses import dataclass, field

METHODS = ("raw", "gtcrn", "dpdfnet", "cancel")


@dataclass
class Options:
    model: str = "turbo"
    check_model: str | None = "medium.en"
    threads: int = 4
    names: tuple[str, ...] | None = None
    speakers: int | None = None          # None = detect
    methods: tuple[str, ...] = METHODS
    shortlist: int = 2
    write_tracks: bool = False           # also write every method's full track, for comparison


@dataclass
class Line:
    """One transcript line. meta["source"] is the track its audio came from; meta["span"] the audio it covers."""
    start: float
    end: float
    text: str
    track: str
    speaker: str = ""
    speaker_sure: bool = True
    confidence: float | None = None
    alt: str = ""
    meta: dict = field(default_factory=dict)


@dataclass
class Piece:
    """A stretch of one speaker's audio: the unit that gets judged and transcribed."""
    start: float
    end: float
    label: str      # C<k> for a clean-part speaker, N<k> for a noisy-part speaker, -1 if unknown
    clean: bool

    def __iter__(self):  # allows `a, b, label, clean = piece`
        return iter((self.start, self.end, self.label, self.clean))
