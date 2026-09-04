from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Literal

import gradio as gr
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from app.domain import AnswerValue, QuestionAnswer
from app.errors import SeekError
from app.runtime import LocalRuntime, StaticRuntime
from app.settings import Settings
from app.ui import build_ui


class SearchAnswerRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question_id: str = Field(min_length=1)
    value: Literal["yes", "no", "uncertain"]


class SearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    description: str = Field(min_length=1, max_length=1000)
    answers: list[SearchAnswerRequest] = Field(default_factory=list)


class AlbumResponse(BaseModel):
    id: str
    title: str
    artist: str
    language: str
    first_release_year: int | None


class ReleaseResponse(BaseModel):
    id: str
    release_date: str | None
    country: str | None


class CoverResponse(BaseModel):
    id: str
    image_uri: str


class CandidateResponse(BaseModel):
    album: AlbumResponse
    release: ReleaseResponse
    cover: CoverResponse
    score: float
    matching_clues: list[str]
    conflicting_clues: list[str]


class QuestionResponse(BaseModel):
    id: str
    text: str
    answers: tuple[str, str, str] = ("yes", "no", "uncertain")


class SearchResponse(BaseModel):
    raw_query: str
    normalized_query: str
    candidates: list[CandidateResponse]
    question: QuestionResponse | None


def create_app(runtime=None, *, mount_ui: bool = True) -> FastAPI:
    settings = Settings.from_environment()
    selected_runtime = runtime or LocalRuntime(settings)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        selected_runtime.initialize()
        yield

    application = FastAPI(title="Seek API", version="0.2.0", lifespan=lifespan)

    @application.get("/health")
    def health() -> dict[str, str | int | bool | None]:
        status = selected_runtime.status
        return {
            "status": status.state,
            "message": status.message,
            "index_count": status.index_count,
            "index_reused": status.index_reused,
        }

    @application.post("/search", response_model=SearchResponse)
    def search(request: SearchRequest) -> SearchResponse:
        answers = tuple(
            QuestionAnswer(question_id=answer.question_id, value=AnswerValue(answer.value))
            for answer in request.answers
        )
        try:
            result = selected_runtime.get_service().search(request.description, answers=answers)
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        except SeekError as error:
            raise HTTPException(status_code=503, detail=str(error)) from error

        candidates = [
            CandidateResponse(
                album=AlbumResponse(
                    id=candidate.album.id,
                    title=candidate.album.title,
                    artist=candidate.album.artist,
                    language=candidate.album.language,
                    first_release_year=candidate.album.first_release_year,
                ),
                release=ReleaseResponse(
                    id=candidate.release.id,
                    release_date=candidate.release.release_date,
                    country=candidate.release.country,
                ),
                cover=CoverResponse(id=candidate.cover.id, image_uri=candidate.cover.image_uri),
                score=candidate.score,
                matching_clues=list(candidate.matching_clues),
                conflicting_clues=list(candidate.conflicting_clues),
            )
            for candidate in result.candidates
        ]
        question = (
            QuestionResponse(id=result.question.id, text=result.question.text)
            if result.question
            else None
        )
        return SearchResponse(
            raw_query=result.clue.raw_text,
            normalized_query=result.clue.normalized_text,
            candidates=candidates,
            question=question,
        )

    if settings.cover_dir.is_dir():
        application.mount(
            "/covers", StaticFiles(directory=settings.cover_dir), name="album-covers"
        )
    if mount_ui:
        application = gr.mount_gradio_app(
            application,
            build_ui(selected_runtime),
            path="/",
            allowed_paths=[str(settings.cover_dir)],
            run_history=False,
        )
    return application


app = create_app()


def build_test_app(search_service) -> FastAPI:
    return create_app(StaticRuntime(search_service), mount_ui=False)
