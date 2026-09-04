from app.adapters.sqlite_repository import SQLiteAlbumRepository
from app.domain import Album, Cover, Release, Track


def make_bundle(tmp_path, suffix="1", sha="a" * 64):
    image = tmp_path / f"cover-{suffix}.jpg"
    image.write_bytes(f"image-{suffix}".encode())
    album = Album(
        f"album-{suffix}",
        f"专辑 {suffix}",
        "测试艺人",
        "ja",
        2020,
        source_uri=f"https://musicbrainz.org/release-group/album-{suffix}",
        genre_score=0.8,
    )
    release = Release(f"release-{suffix}", album.id, "2020-01-01", "JP", "Official")
    cover = Cover(
        f"cover-{suffix}",
        release.id,
        f"/covers/cover-{suffix}.jpg",
        str(image),
        source_uri="https://coverartarchive.org/example",
        sha256=sha,
        perceptual_hash="0123456789abcdef",
        width=500,
        height=500,
    )
    tracks = (Track(f"track-{suffix}", release.id, 1, "第一首歌", "测试艺人", 123000),)
    return album, release, cover, tracks


def test_sqlite_repository_round_trips_real_bundle(tmp_path) -> None:
    repository = SQLiteAlbumRepository(tmp_path / "seek.sqlite3", require_data=False)
    album, release, cover, tracks = make_bundle(tmp_path)

    repository.save_bundle(album, release, cover, tracks)

    assert repository.get_album(album.id) == album
    assert repository.get_release(release.id) == release
    assert repository.list_covers() == (cover,)
    assert repository.list_tracks(release.id) == tracks
    assert repository.counts() == {"albums": 1, "releases": 1, "covers": 1, "tracks": 1}


def test_sqlite_repository_excludes_exact_duplicate_cover_from_index(tmp_path) -> None:
    repository = SQLiteAlbumRepository(tmp_path / "seek.sqlite3", require_data=False)
    first = make_bundle(tmp_path, "1", sha="b" * 64)
    second = make_bundle(tmp_path, "2", sha="b" * 64)

    repository.save_bundle(*first)
    repository.save_bundle(*second)

    assert [cover.id for cover in repository.list_covers()] == ["cover-1"]
    assert repository.counts()["albums"] == 2
