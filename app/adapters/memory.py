from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

from app.domain import Album, Cover, Release, Track
from app.ports import Embedding


class InMemoryAlbumRepository:
    def __init__(
        self,
        albums: Iterable[Album],
        releases: Iterable[Release],
        covers: Iterable[Cover],
        tracks: Iterable[Track] = (),
    ) -> None:
        self._albums = {album.id: album for album in albums}
        self._releases = {release.id: release for release in releases}
        self._covers = tuple(covers)
        self._tracks = tuple(tracks)
        self._validate_references()

    def _validate_references(self) -> None:
        for release in self._releases.values():
            if release.album_id not in self._albums:
                raise ValueError(f"发行版本 {release.id} 引用了不存在的专辑")
        for cover in self._covers:
            if cover.release_id not in self._releases:
                raise ValueError(f"封面 {cover.id} 引用了不存在的发行版本")

    def list_covers(self) -> Sequence[Cover]:
        return self._covers

    def get_release(self, release_id: str) -> Release:
        try:
            return self._releases[release_id]
        except KeyError as error:
            raise LookupError(f"未找到发行版本：{release_id}") from error

    def get_album(self, album_id: str) -> Album:
        try:
            return self._albums[album_id]
        except KeyError as error:
            raise LookupError(f"未找到专辑：{album_id}") from error

    def list_tracks(self, release_id: str) -> Sequence[Track]:
        return tuple(track for track in self._tracks if track.release_id == release_id)


class FakeCoverEmbeddingProvider:
    """测试骨架用的确定性向量提供器，不读取图片，也不调用外部服务。"""

    FEATURES = (
        "red",
        "not_red",
        "shoes",
        "many",
        "single",
        "person",
        "no_person",
        "photo",
        "illustration",
        "dark",
        "white",
        "blue",
    )

    def __init__(self, cover_features: Mapping[str, set[str]]) -> None:
        self._cover_features = cover_features

    @property
    def dimension(self) -> int:
        return len(self.FEATURES)

    @property
    def model_id(self) -> str:
        return "fake-cover-embedding-v1"

    def embed_cover(self, cover: Cover) -> Embedding:
        features = self._cover_features.get(cover.id, set())
        return tuple(1.0 if feature in features else 0.0 for feature in self.FEATURES)

    def embed_covers(self, covers: Sequence[Cover]) -> Sequence[Embedding]:
        return tuple(self.embed_cover(cover) for cover in covers)

    def embed_text(self, text: str) -> Embedding:
        normalized = text.lower()
        detected: set[str] = set()

        phrase_rules = (
            ("not_red", ("不是红色", "非红色", "not red")),
            ("no_person", ("没有人物", "没有人", "无人", "no person")),
            ("many", ("很多", "多只", "散落", "杂乱", "many")),
            ("single", ("一只", "单只", "single")),
            ("red", ("红", "赤", "red")),
            ("shoes", ("鞋", "shoe")),
            ("person", ("人物", "人腿", "腿", "袜", "person")),
            ("photo", ("摄影", "照片", "photo")),
            ("illustration", ("插画", "绘画", "illustration")),
            ("dark", ("暗色", "黑色", "阴暗", "dark")),
            ("white", ("白色", "浅色", "white")),
            ("blue", ("蓝色", "blue")),
        )
        for feature, phrases in phrase_rules:
            if any(phrase in normalized for phrase in phrases):
                detected.add(feature)

        if "not_red" in detected:
            detected.discard("red")
        if "no_person" in detected:
            detected.discard("person")

        return tuple(1.0 if feature in detected else 0.0 for feature in self.FEATURES)
