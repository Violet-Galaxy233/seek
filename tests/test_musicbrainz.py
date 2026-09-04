from io import BytesIO

from PIL import Image

from app.adapters.musicbrainz import MusicBrainzImporter, RateLimitedHttpClient
from app.adapters.sqlite_repository import SQLiteAlbumRepository


def image_bytes() -> bytes:
    stream = BytesIO()
    Image.new("RGB", (256, 256), "red").save(stream, format="JPEG")
    return stream.getvalue()


class FakeMusicClient:
    def __init__(self):
        self.cover_calls = 0

    def search_release_groups(self, query, limit, offset):
        if "releasegroup" in query:
            return {"release-groups": []}
        if offset:
            return {"release-group-count": 1, "release-groups": []}
        return {
            "release-group-count": 1,
            "release-groups": [
                {
                    "id": "album-1",
                    "title": "健全な社会",
                    "primary-type": "Album",
                    "first-release-date": "2020-03-01",
                    "artist-credit": [{"name": "测试乐队"}],
                }
            ],
        }

    def get_release_group(self, release_group_id):
        return {
            "id": release_group_id,
            "artist-credit": [{"name": "测试乐队"}],
            "tags": [{"name": "j-rock", "count": 8}],
            "releases": [
                {
                    "id": "release-us",
                    "status": "Official",
                    "country": "US",
                    "date": "2020-01-01",
                    "cover-art-archive": {"front": True},
                },
                {
                    "id": "release-jp",
                    "status": "Official",
                    "country": "JP",
                    "date": "2020-03-01",
                    "cover-art-archive": {"front": True},
                },
            ],
        }

    def get_release(self, release_id):
        assert release_id == "release-jp"
        return {
            "id": release_id,
            "date": "2020-03-01",
            "country": "JP",
            "status": "Official",
            "text-representation": {"language": "jpn"},
            "media": [
                {
                    "tracks": [
                        {
                            "id": "track-1",
                            "title": "曲名",
                            "length": 180000,
                            "recording": {"id": "recording-1"},
                        }
                    ]
                }
            ],
        }

    def download_front_cover(self, release_id):
        self.cover_calls += 1
        return image_bytes()


def test_importer_prefers_japanese_release_and_saves_tracks(tmp_path) -> None:
    repository = SQLiteAlbumRepository(tmp_path / "seek.sqlite3", require_data=False)
    client = FakeMusicClient()
    importer = MusicBrainzImporter(client, repository, tmp_path / "covers")

    summary = importer.import_collection(limit=1, force_titles=())

    assert summary.imported == 1
    cover = repository.list_covers()[0]
    release = repository.get_release(cover.release_id)
    assert release.id == "release-jp"
    assert repository.get_album("album-1").language == "jpn"
    assert repository.list_tracks("release-jp")[0].title == "曲名"
    assert cover.width == 256


def test_importer_is_restart_safe_and_skips_existing_album(tmp_path) -> None:
    repository = SQLiteAlbumRepository(tmp_path / "seek.sqlite3", require_data=False)
    client = FakeMusicClient()
    importer = MusicBrainzImporter(client, repository, tmp_path / "covers")
    importer.import_collection(limit=1, force_titles=())

    summary = importer.import_collection(limit=1, force_titles=())

    assert summary.imported == 0
    assert summary.examined == 0
    assert client.cover_calls == 1


def test_rate_limited_client_requires_user_agent() -> None:
    try:
        RateLimitedHttpClient("  ")
    except ValueError as error:
        assert "User-Agent" in str(error)
    else:
        raise AssertionError("空 User-Agent 应被拒绝")


class FakeResponse:
    def __init__(self, content=b"", status_code=200, location=None):
        self.content = content
        self.status_code = status_code
        self.headers = {} if location is None else {"Location": location}
        self.is_redirect = status_code in {301, 302, 303, 307, 308}
        self.is_permanent_redirect = status_code in {301, 308}

    def raise_for_status(self):
        return None


class RedirectSession:
    def __init__(self):
        self.headers = {}
        self.calls = []

    def get(self, url, *, timeout, allow_redirects):
        self.calls.append((url, allow_redirects))
        if url.startswith("https://coverartarchive.org/"):
            return FakeResponse(
                status_code=307,
                location="https://archive.org/download/example/cover.jpg",
            )
        return FakeResponse(content=b"cover")


def test_cover_download_uses_archive_mirror_compatibility_redirect() -> None:
    client = RateLimitedHttpClient("seek-tests/1.0", minimum_interval=0)
    session = RedirectSession()
    client._session = session

    result = client.get_bytes_or_none(
        "https://coverartarchive.org/release/release-1/front-500"
    )

    assert result == b"cover"
    assert session.calls == [
        ("https://coverartarchive.org/release/release-1/front-500", False),
        ("http://archive.org/download/example/cover.jpg", True),
    ]


def test_release_selection_prefers_official_japanese_front_cover() -> None:
    releases = [
        {"id": "bootleg-jp", "country": "JP", "status": "Bootleg"},
        {"id": "official-us", "country": "US", "status": "Official"},
        {
            "id": "official-jp",
            "country": "JP",
            "status": "Official",
            "cover-art-archive": {"front": True},
        },
    ]

    ordered = MusicBrainzImporter._ordered_releases(releases)

    assert [item["id"] for item in ordered] == ["official-jp"]


def test_release_selection_does_not_probe_known_coverless_releases() -> None:
    releases = [
        {
            "id": "without-cover",
            "country": "JP",
            "status": "Official",
            "cover-art-archive": {"front": False},
        }
    ]

    assert MusicBrainzImporter._ordered_releases(releases) == []


def test_compilation_is_out_of_scope() -> None:
    assert not MusicBrainzImporter._is_in_scope(
        {"primary-type": "Album", "secondary-types": ["Compilation"]}
    )
    assert MusicBrainzImporter._is_in_scope(
        {"primary-type": "Album", "secondary-types": []}
    )
