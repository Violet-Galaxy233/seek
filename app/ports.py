from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from app.domain import Album, Cover, Release, Track

Embedding = tuple[float, ...]


class CoverEmbeddingProvider(Protocol):
    """将中文描述和封面编码到同一个向量空间。"""

    @property
    def dimension(self) -> int: ...

    @property
    def model_id(self) -> str: ...

    def embed_text(self, text: str) -> Embedding: ...

    def embed_cover(self, cover: Cover) -> Embedding: ...

    def embed_covers(self, covers: Sequence[Cover]) -> Sequence[Embedding]: ...


class AlbumRepository(Protocol):
    """提供搜索所需的专辑、发行版本和封面。"""

    def list_covers(self) -> Sequence[Cover]: ...

    def get_release(self, release_id: str) -> Release: ...

    def get_album(self, album_id: str) -> Album: ...

    def list_tracks(self, release_id: str) -> Sequence[Track]: ...
