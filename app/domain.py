from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class CluePolarity(StrEnum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    UNCERTAIN = "uncertain"


class AnswerValue(StrEnum):
    YES = "yes"
    NO = "no"
    UNCERTAIN = "uncertain"


@dataclass(frozen=True, slots=True)
class Album:
    id: str
    title: str
    artist: str
    language: str
    first_release_year: int | None = None
    primary_type: str = "Album"
    source_uri: str | None = None
    genre_score: float = 0.0


@dataclass(frozen=True, slots=True)
class Release:
    id: str
    album_id: str
    release_date: str | None = None
    country: str | None = None
    status: str | None = None
    barcode: str | None = None


@dataclass(frozen=True, slots=True)
class Cover:
    id: str
    release_id: str
    image_uri: str
    image_path: str | None = None
    is_front: bool = True
    source_uri: str | None = None
    sha256: str | None = None
    perceptual_hash: str | None = None
    width: int | None = None
    height: int | None = None


@dataclass(frozen=True, slots=True)
class Track:
    id: str
    release_id: str
    position: int
    title: str
    artist: str | None = None
    length_ms: int | None = None
    recording_id: str | None = None


@dataclass(frozen=True, slots=True)
class MemoryClue:
    raw_text: str
    normalized_text: str
    confidence: float
    polarity: CluePolarity = CluePolarity.POSITIVE

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence 必须位于 0 到 1 之间")


@dataclass(frozen=True, slots=True)
class Candidate:
    album: Album
    release: Release
    cover: Cover
    score: float
    matching_clues: tuple[str, ...]
    conflicting_clues: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Question:
    id: str
    text: str
    yes_prompt: str
    no_prompt: str


@dataclass(frozen=True, slots=True)
class QuestionAnswer:
    question_id: str
    value: AnswerValue


@dataclass(frozen=True, slots=True)
class SearchResult:
    clue: MemoryClue
    candidates: tuple[Candidate, ...]
    question: Question | None
