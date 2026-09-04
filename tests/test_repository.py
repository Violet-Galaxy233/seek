import pytest

from app.adapters.memory import InMemoryAlbumRepository
from app.domain import Album, Cover, Release


def test_repository_rejects_dangling_cover() -> None:
    albums = [Album("album-1", "示例", "示例艺人", "ja")]
    releases = [Release("release-1", "album-1")]
    covers = [Cover("cover-1", "missing-release", "memory://missing.jpg")]

    with pytest.raises(ValueError, match="不存在的发行版本"):
        InMemoryAlbumRepository(albums, releases, covers)

