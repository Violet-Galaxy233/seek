from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.domain import Album, Cover, Release, Track
from app.errors import CatalogError


class FileAlbumRepository:
    """从受控测试目录的 JSON 清单读取专辑和封面。"""

    def __init__(self, catalog_path: Path, cover_dir: Path) -> None:
        self.catalog_path = catalog_path.resolve()
        self.cover_dir = cover_dir.resolve()
        payload = self._load_payload()
        self._albums: dict[str, Album] = {}
        self._releases: dict[str, Release] = {}
        self._covers: list[Cover] = []
        self._parse(payload)

    def _load_payload(self) -> dict[str, Any]:
        if not self.catalog_path.is_file():
            raise CatalogError(f"测试数据清单不存在：{self.catalog_path}")
        if not self.cover_dir.is_dir():
            raise CatalogError(f"测试封面目录不存在：{self.cover_dir}")
        try:
            return json.loads(self.catalog_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise CatalogError(f"无法读取测试数据清单：{self.catalog_path}: {error}") from error

    def _parse(self, payload: dict[str, Any]) -> None:
        records = payload.get("albums")
        if not isinstance(records, list) or not records:
            raise CatalogError("测试数据清单必须包含非空 albums 数组")

        try:
            for record in records:
                album = Album(
                    id=record["id"],
                    title=record["title"],
                    artist=record["artist"],
                    language=record.get("language", "ja"),
                    first_release_year=record.get("first_release_year"),
                )
                release_data = record["release"]
                release = Release(
                    id=release_data["id"],
                    album_id=album.id,
                    release_date=release_data.get("release_date"),
                    country=release_data.get("country"),
                )
                cover_data = release_data["cover"]
                image_path = (self.cover_dir / cover_data["filename"]).resolve()
                if not image_path.is_relative_to(self.cover_dir):
                    raise CatalogError(f"封面路径越过测试数据目录：{image_path}")
                if not image_path.is_file():
                    raise CatalogError(f"测试封面不存在：{image_path}")
                cover = Cover(
                    id=cover_data["id"],
                    release_id=release.id,
                    image_uri=f"/covers/{image_path.name}",
                    image_path=str(image_path),
                )
                if album.id in self._albums:
                    raise CatalogError(f"重复专辑 ID：{album.id}")
                if release.id in self._releases:
                    raise CatalogError(f"重复发行版本 ID：{release.id}")
                if any(existing.id == cover.id for existing in self._covers):
                    raise CatalogError(f"重复封面 ID：{cover.id}")
                self._albums[album.id] = album
                self._releases[release.id] = release
                self._covers.append(cover)
        except (KeyError, TypeError) as error:
            raise CatalogError(f"测试数据清单字段无效：{error}") from error

    def list_covers(self) -> tuple[Cover, ...]:
        return tuple(self._covers)

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

    def list_tracks(self, release_id: str) -> tuple[Track, ...]:
        return ()
