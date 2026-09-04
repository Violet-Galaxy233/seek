from __future__ import annotations

from app.adapters.memory import FakeCoverEmbeddingProvider, InMemoryAlbumRepository
from app.domain import Album, Cover, Release
from app.search import SearchService

_SEEDS = (
    ("健全な社会", {"shoes", "many", "white", "no_person", "photo"}),
    ("赤い靴の夜", {"shoes", "single", "red", "no_person", "photo"}),
    ("散らかった午後", {"shoes", "many", "red", "no_person", "illustration"}),
    ("歩道", {"shoes", "single", "not_red", "person", "photo"}),
    ("赤い部屋", {"red", "person", "photo"}),
    ("白い標本", {"white", "many", "no_person", "photo"}),
    ("夜の靴音", {"shoes", "single", "dark", "person", "illustration"}),
    ("青い窓", {"blue", "no_person", "illustration"}),
    ("赤の断片", {"red", "many", "no_person", "illustration"}),
    ("足跡", {"shoes", "many", "dark", "person", "photo"}),
    ("白昼夢", {"white", "person", "photo"}),
    ("小さな庭", {"not_red", "many", "no_person", "photo"}),
    ("境界線", {"dark", "person", "photo"}),
    ("透明な朝", {"white", "no_person", "photo"}),
    ("群像", {"many", "person", "illustration"}),
    ("片隅", {"single", "no_person", "photo"}),
    ("赤い記憶", {"red", "single", "person", "illustration"}),
    ("雨の標識", {"blue", "person", "photo"}),
    ("静かな棚", {"white", "many", "no_person", "illustration"}),
    ("影の中", {"dark", "single", "no_person", "photo"}),
    ("余白", {"white", "single", "no_person", "illustration"}),
    ("街角", {"not_red", "person", "photo"}),
    ("反復", {"many", "red", "no_person", "photo"}),
    ("遠い青", {"blue", "single", "no_person", "illustration"}),
)


def build_demo_service() -> SearchService:
    albums: list[Album] = []
    releases: list[Release] = []
    covers: list[Cover] = []
    cover_features: dict[str, set[str]] = {}

    for index, (title, features) in enumerate(_SEEDS, start=1):
        suffix = f"{index:03d}"
        album_id = f"album-{suffix}"
        release_id = f"release-{suffix}"
        cover_id = f"cover-{suffix}"
        albums.append(
            Album(
                id=album_id,
                title=title,
                artist=f"示例艺人 {index}",
                language="ja",
                first_release_year=2000 + index,
            )
        )
        releases.append(
            Release(
                id=release_id,
                album_id=album_id,
                release_date=f"{2000 + index}-01-01",
                country="JP",
            )
        )
        covers.append(
            Cover(
                id=cover_id,
                release_id=release_id,
                image_uri=f"memory://covers/{cover_id}.jpg",
            )
        )
        cover_features[cover_id] = set(features)

    repository = InMemoryAlbumRepository(albums, releases, covers)
    embeddings = FakeCoverEmbeddingProvider(cover_features)
    return SearchService(repository, embeddings)

