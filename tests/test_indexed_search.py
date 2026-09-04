from app.adapters.faiss_index import FaissCoverIndex
from app.adapters.memory import FakeCoverEmbeddingProvider, InMemoryAlbumRepository
from app.domain import Album, Cover, Release
from app.indexed_search import IndexedSearchService


def test_indexed_search_returns_album_metadata_and_similarity(tmp_path) -> None:
    image_path = tmp_path / "shoes.png"
    image_path.write_bytes(b"fixture")
    repository = InMemoryAlbumRepository(
        [Album("album-1", "鞋のアルバム", "テスト", "ja", 2024)],
        [Release("release-1", "album-1", "2024-01-01", "JP")],
        [Cover("cover-1", "release-1", "/covers/shoes.png", str(image_path))],
    )
    provider = FakeCoverEmbeddingProvider({"cover-1": {"shoes", "many"}})
    service = IndexedSearchService(repository, provider, FaissCoverIndex(tmp_path / "index"))

    result = service.search("很多鞋")

    assert result.candidates[0].album.title == "鞋のアルバム"
    assert result.candidates[0].score > 0
    assert result.candidates[0].matching_clues

