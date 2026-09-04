from app.adapters.faiss_index import FaissCoverIndex
from app.adapters.memory import FakeCoverEmbeddingProvider, InMemoryAlbumRepository
from app.domain import Album, Cover, Release


class CountingProvider(FakeCoverEmbeddingProvider):
    def __init__(self, cover_features):
        super().__init__(cover_features)
        self.cover_calls = 0

    def embed_cover(self, cover):
        self.cover_calls += 1
        return super().embed_cover(cover)


def build_repository(tmp_path):
    first_path = tmp_path / "first.png"
    second_path = tmp_path / "second.png"
    first_path.write_bytes(b"first")
    second_path.write_bytes(b"second")
    albums = [
        Album("album-shoes", "鞋", "艺人 A", "ja"),
        Album("album-moon", "月", "艺人 B", "ja"),
    ]
    releases = [
        Release("release-shoes", "album-shoes"),
        Release("release-moon", "album-moon"),
    ]
    covers = [
        Cover("cover-shoes", "release-shoes", "/covers/first.png", str(first_path)),
        Cover("cover-moon", "release-moon", "/covers/second.png", str(second_path)),
    ]
    return InMemoryAlbumRepository(albums, releases, covers)


def test_faiss_index_builds_and_finds_matching_cover(tmp_path) -> None:
    repository = build_repository(tmp_path)
    provider = CountingProvider(
        {"cover-shoes": {"shoes", "many"}, "cover-moon": {"blue", "dark"}}
    )
    index = FaissCoverIndex(tmp_path / "index")

    status = index.prepare(repository, provider)
    hits = index.search(provider.embed_text("很多鞋"), 2)

    assert status.reused is False
    assert status.count == 2
    assert hits[0][0] == "cover-shoes"


def test_faiss_index_is_reused_after_restart(tmp_path) -> None:
    repository = build_repository(tmp_path)
    features = {"cover-shoes": {"shoes", "many"}, "cover-moon": {"blue", "dark"}}
    first_provider = CountingProvider(features)
    first = FaissCoverIndex(tmp_path / "index")
    assert first.prepare(repository, first_provider).reused is False
    assert first_provider.cover_calls == 2

    second_provider = CountingProvider(features)
    restarted = FaissCoverIndex(tmp_path / "index")
    status = restarted.prepare(repository, second_provider)

    assert status.reused is True
    assert second_provider.cover_calls == 0
    assert restarted.search(second_provider.embed_text("很多鞋"), 1)[0][0] == "cover-shoes"


def test_faiss_index_rebuilds_when_cover_changes(tmp_path) -> None:
    repository = build_repository(tmp_path)
    features = {"cover-shoes": {"shoes"}, "cover-moon": {"blue"}}
    first = FaissCoverIndex(tmp_path / "index")
    first.prepare(repository, CountingProvider(features))
    cover_path = tmp_path / "first.png"
    cover_path.write_bytes(b"first changed")

    provider = CountingProvider(features)
    restarted = FaissCoverIndex(tmp_path / "index")

    assert restarted.prepare(repository, provider).reused is False
    assert provider.cover_calls == 1


def test_faiss_index_reuses_unchanged_vectors_when_cover_is_added(tmp_path) -> None:
    first_repository = build_repository(tmp_path)
    features = {
        "cover-shoes": {"shoes"},
        "cover-moon": {"blue"},
        "cover-new": {"red"},
    }
    FaissCoverIndex(tmp_path / "index").prepare(
        first_repository, CountingProvider(features)
    )

    third_path = tmp_path / "third.png"
    third_path.write_bytes(b"third")
    expanded = InMemoryAlbumRepository(
        [
            first_repository.get_album("album-shoes"),
            first_repository.get_album("album-moon"),
            Album("album-new", "红", "艺人 C", "ja"),
        ],
        [
            first_repository.get_release("release-shoes"),
            first_repository.get_release("release-moon"),
            Release("release-new", "album-new"),
        ],
        [
            *first_repository.list_covers(),
            Cover("cover-new", "release-new", "/covers/third.png", str(third_path)),
        ],
    )
    provider = CountingProvider(features)

    status = FaissCoverIndex(tmp_path / "index").prepare(expanded, provider)

    assert status.count == 3
    assert status.reused is False
    assert provider.cover_calls == 1
