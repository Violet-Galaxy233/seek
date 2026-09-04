import json

import pytest

from app.adapters.file_catalog import FileAlbumRepository
from app.errors import CatalogError


def test_file_catalog_loads_only_listed_covers(tmp_path) -> None:
    cover_dir = tmp_path / "covers"
    cover_dir.mkdir()
    (cover_dir / "listed.png").write_bytes(b"fixture")
    (cover_dir / "ignored.png").write_bytes(b"fixture")
    catalog = tmp_path / "catalog.json"
    catalog.write_text(
        json.dumps(
            {
                "albums": [
                    {
                        "id": "album-1",
                        "title": "测试专辑",
                        "artist": "测试艺人",
                        "release": {
                            "id": "release-1",
                            "cover": {"id": "cover-1", "filename": "listed.png"},
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    repository = FileAlbumRepository(catalog, cover_dir)

    assert [cover.id for cover in repository.list_covers()] == ["cover-1"]
    assert repository.list_covers()[0].image_path == str(cover_dir / "listed.png")


def test_file_catalog_rejects_path_outside_test_directory(tmp_path) -> None:
    cover_dir = tmp_path / "covers"
    cover_dir.mkdir()
    outside = tmp_path / "outside.png"
    outside.write_bytes(b"fixture")
    catalog = tmp_path / "catalog.json"
    catalog.write_text(
        json.dumps(
            {
                "albums": [
                    {
                        "id": "album-1",
                        "title": "测试专辑",
                        "artist": "测试艺人",
                        "release": {
                            "id": "release-1",
                            "cover": {"id": "cover-1", "filename": "../outside.png"},
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(CatalogError, match="越过测试数据目录"):
        FileAlbumRepository(catalog, cover_dir)

