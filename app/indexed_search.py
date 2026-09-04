from __future__ import annotations

from collections.abc import Sequence

from app.adapters.faiss_index import FaissCoverIndex, IndexStatus
from app.domain import Candidate, CluePolarity, MemoryClue, QuestionAnswer, SearchResult
from app.ports import AlbumRepository, CoverEmbeddingProvider


class IndexedSearchService:
    def __init__(
        self,
        repository: AlbumRepository,
        embeddings: CoverEmbeddingProvider,
        index: FaissCoverIndex,
    ) -> None:
        self._repository = repository
        self._embeddings = embeddings
        self._index = index
        self._cover_by_id = {cover.id: cover for cover in repository.list_covers()}
        self.index_status: IndexStatus | None = None

    def prepare(self) -> IndexStatus:
        self.index_status = self._index.prepare(self._repository, self._embeddings)
        return self.index_status

    def search(
        self,
        description: str,
        answers: Sequence[QuestionAnswer] = (),
        limit: int = 20,
    ) -> SearchResult:
        normalized = " ".join(description.split())
        if not normalized:
            raise ValueError("描述不能为空")
        if answers:
            raise ValueError("当前端到端闭环尚未启用追问反馈")
        if not 1 <= limit <= 20:
            raise ValueError("limit 必须位于 1 到 20 之间")
        if self.index_status is None:
            self.prepare()

        clue = MemoryClue(
            raw_text=description,
            normalized_text=normalized,
            confidence=1.0,
            polarity=CluePolarity.POSITIVE,
        )
        hits = self._index.search(self._embeddings.embed_text(normalized), limit)
        candidates = []
        for cover_id, score in hits:
            cover = self._cover_by_id[cover_id]
            release = self._repository.get_release(cover.release_id)
            album = self._repository.get_album(release.album_id)
            candidates.append(
                Candidate(
                    album=album,
                    release=release,
                    cover=cover,
                    score=score,
                    matching_clues=(f"本地图文向量相似度为 {score:.3f}",),
                    conflicting_clues=("当前闭环尚未启用结构化冲突识别",),
                )
            )
        return SearchResult(clue=clue, candidates=tuple(candidates), question=None)
