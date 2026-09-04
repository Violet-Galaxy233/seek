from __future__ import annotations

import math
from collections.abc import Sequence

from app.domain import (
    AnswerValue,
    Candidate,
    CluePolarity,
    MemoryClue,
    Question,
    QuestionAnswer,
    SearchResult,
)
from app.ports import AlbumRepository, CoverEmbeddingProvider, Embedding

DEFAULT_QUESTIONS = (
    Question("many_shoes", "封面上是否有很多只鞋？", "很多只鞋", "一只鞋"),
    Question("person_visible", "封面上是否能看到人物或人的一部分？", "有人腿或人物", "没有人物"),
    Question("red_background", "封面是否以红色为主？", "红色背景", "不是红色"),
    Question("photo_style", "封面是否更像摄影照片？", "摄影照片", "插画"),
)


def cosine_similarity(left: Embedding, right: Embedding) -> float:
    if len(left) != len(right):
        raise ValueError("向量维数不一致")
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return sum(a * b for a, b in zip(left, right, strict=True)) / (left_norm * right_norm)


class SearchService:
    def __init__(
        self,
        repository: AlbumRepository,
        embeddings: CoverEmbeddingProvider,
        questions: Sequence[Question] = DEFAULT_QUESTIONS,
    ) -> None:
        self._repository = repository
        self._embeddings = embeddings
        self._questions = tuple(questions)

    def search(
        self,
        description: str,
        answers: Sequence[QuestionAnswer] = (),
        limit: int = 20,
    ) -> SearchResult:
        normalized = " ".join(description.split())
        if not normalized:
            raise ValueError("描述不能为空")
        if not 1 <= limit <= 20:
            raise ValueError("limit 必须位于 1 到 20 之间")

        clue = MemoryClue(
            raw_text=description,
            normalized_text=normalized,
            confidence=1.0,
            polarity=CluePolarity.POSITIVE,
        )
        question_by_id = {question.id: question for question in self._questions}
        unknown_ids = {answer.question_id for answer in answers} - question_by_id.keys()
        if unknown_ids:
            raise ValueError(f"未知问题：{', '.join(sorted(unknown_ids))}")

        query_embedding = self._embeddings.embed_text(normalized)
        candidates: list[Candidate] = []
        cover_embeddings: dict[str, Embedding] = {}

        for cover in self._repository.list_covers():
            cover_embedding = self._embeddings.embed_cover(cover)
            cover_embeddings[cover.id] = cover_embedding
            score = cosine_similarity(query_embedding, cover_embedding)
            matches = [f"封面语义与描述“{normalized}”的相似度为 {score:.3f}"]
            conflicts: list[str] = []

            for answer in answers:
                if answer.value is AnswerValue.UNCERTAIN:
                    continue
                question = question_by_id[answer.question_id]
                yes_score = cosine_similarity(
                    cover_embedding, self._embeddings.embed_text(question.yes_prompt)
                )
                no_score = cosine_similarity(
                    cover_embedding, self._embeddings.embed_text(question.no_prompt)
                )
                alignment = yes_score - no_score
                if answer.value is AnswerValue.NO:
                    alignment = -alignment
                score += 0.2 * alignment
                if alignment >= 0:
                    matches.append(f"与回答“{question.text}”一致")
                else:
                    conflicts.append(f"与回答“{question.text}”存在冲突")

            if not conflicts:
                conflicts.append("当前假模型未识别到明确冲突")

            release = self._repository.get_release(cover.release_id)
            album = self._repository.get_album(release.album_id)
            candidates.append(
                Candidate(
                    album=album,
                    release=release,
                    cover=cover,
                    score=score,
                    matching_clues=tuple(matches),
                    conflicting_clues=tuple(conflicts),
                )
            )

        candidates.sort(key=lambda candidate: (-candidate.score, candidate.album.id))
        selected = tuple(candidates[:limit])
        answered_ids = {answer.question_id for answer in answers}
        next_question = self._select_question(selected, cover_embeddings, answered_ids)
        return SearchResult(clue=clue, candidates=selected, question=next_question)

    def _select_question(
        self,
        candidates: Sequence[Candidate],
        cover_embeddings: dict[str, Embedding],
        answered_ids: set[str],
    ) -> Question | None:
        if len(candidates) < 2:
            return None

        best: tuple[float, Question] | None = None
        for question in self._questions:
            if question.id in answered_ids:
                continue
            yes_embedding = self._embeddings.embed_text(question.yes_prompt)
            no_embedding = self._embeddings.embed_text(question.no_prompt)
            sides = []
            margins = []
            for candidate in candidates:
                cover_embedding = cover_embeddings[candidate.cover.id]
                margin = cosine_similarity(cover_embedding, yes_embedding) - cosine_similarity(
                    cover_embedding, no_embedding
                )
                if abs(margin) > 1e-9:
                    sides.append(margin >= 0)
                    margins.append(abs(margin))
            if not sides or all(sides) or not any(sides):
                continue
            balance = 1.0 - abs(sum(sides) - (len(sides) / 2)) / (len(sides) / 2)
            quality = balance + (sum(margins) / len(margins)) * 0.01
            if best is None or quality > best[0]:
                best = (quality, question)
        return best[1] if best else None

