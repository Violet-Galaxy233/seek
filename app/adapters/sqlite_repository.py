from __future__ import annotations

import sqlite3
from collections.abc import Sequence
from pathlib import Path

from app.domain import Album, Cover, Release, Track
from app.errors import CatalogError

SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS albums (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    artist TEXT NOT NULL,
    language TEXT NOT NULL DEFAULT 'und',
    first_release_year INTEGER,
    primary_type TEXT NOT NULL DEFAULT 'Album',
    source_uri TEXT,
    genre_score REAL NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS releases (
    id TEXT PRIMARY KEY,
    album_id TEXT NOT NULL REFERENCES albums(id) ON DELETE CASCADE,
    release_date TEXT,
    country TEXT,
    status TEXT,
    barcode TEXT
);

CREATE TABLE IF NOT EXISTS covers (
    id TEXT PRIMARY KEY,
    release_id TEXT NOT NULL REFERENCES releases(id) ON DELETE CASCADE,
    image_uri TEXT NOT NULL,
    image_path TEXT NOT NULL,
    is_front INTEGER NOT NULL DEFAULT 1,
    source_uri TEXT,
    sha256 TEXT NOT NULL,
    perceptual_hash TEXT,
    width INTEGER,
    height INTEGER,
    duplicate_of_cover_id TEXT REFERENCES covers(id)
);

CREATE INDEX IF NOT EXISTS covers_sha256_idx ON covers(sha256);
CREATE INDEX IF NOT EXISTS covers_release_idx ON covers(release_id);

CREATE TABLE IF NOT EXISTS tracks (
    id TEXT PRIMARY KEY,
    release_id TEXT NOT NULL REFERENCES releases(id) ON DELETE CASCADE,
    position INTEGER NOT NULL,
    title TEXT NOT NULL,
    artist TEXT,
    length_ms INTEGER,
    recording_id TEXT
);

CREATE INDEX IF NOT EXISTS tracks_release_idx ON tracks(release_id, position);
"""


class SQLiteAlbumRepository:
    """SQLite 元数据仓库；封面文件本身仍保存在项目目录中。"""

    def __init__(self, database_path: Path, *, require_data: bool = True) -> None:
        self.database_path = database_path.resolve()
        if require_data and not self.database_path.is_file():
            raise CatalogError(
                f"真实专辑数据库不存在：{self.database_path}。"
                "请先运行 `uv run python -m app.import_music --limit 100`。"
            )
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_schema()
        if require_data and not self.list_covers():
            raise CatalogError(
                f"真实专辑数据库中没有可检索封面：{self.database_path}。"
                "请先运行 `uv run python -m app.import_music --limit 100`。"
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _initialize_schema(self) -> None:
        with self._connect() as connection:
            connection.executescript(SCHEMA)

    def album_has_cover(self, album_id: str) -> bool:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT 1
                FROM albums a
                JOIN releases r ON r.album_id = a.id
                JOIN covers c ON c.release_id = r.id
                WHERE a.id = ? AND c.duplicate_of_cover_id IS NULL
                LIMIT 1
                """,
                (album_id,),
            ).fetchone()
        return row is not None

    def find_original_cover_id(self, sha256: str) -> str | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT id FROM covers
                WHERE sha256 = ? AND duplicate_of_cover_id IS NULL
                ORDER BY id LIMIT 1
                """,
                (sha256,),
            ).fetchone()
        return str(row["id"]) if row else None

    def save_bundle(
        self,
        album: Album,
        release: Release,
        cover: Cover,
        tracks: Sequence[Track],
    ) -> None:
        if release.album_id != album.id or cover.release_id != release.id:
            raise ValueError("专辑、发行版本与封面的引用关系不一致")
        if not cover.image_path or not cover.sha256:
            raise ValueError("真实封面必须包含本地路径和 SHA-256")
        duplicate_of = self.find_original_cover_id(cover.sha256)
        if duplicate_of == cover.id:
            duplicate_of = None

        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO albums (
                    id, title, artist, language, first_release_year,
                    primary_type, source_uri, genre_score
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    title=excluded.title,
                    artist=excluded.artist,
                    language=excluded.language,
                    first_release_year=excluded.first_release_year,
                    primary_type=excluded.primary_type,
                    source_uri=excluded.source_uri,
                    genre_score=excluded.genre_score
                """,
                (
                    album.id,
                    album.title,
                    album.artist,
                    album.language,
                    album.first_release_year,
                    album.primary_type,
                    album.source_uri,
                    album.genre_score,
                ),
            )
            connection.execute(
                """
                INSERT INTO releases (
                    id, album_id, release_date, country, status, barcode
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    album_id=excluded.album_id,
                    release_date=excluded.release_date,
                    country=excluded.country,
                    status=excluded.status,
                    barcode=excluded.barcode
                """,
                (
                    release.id,
                    release.album_id,
                    release.release_date,
                    release.country,
                    release.status,
                    release.barcode,
                ),
            )
            connection.execute(
                """
                INSERT INTO covers (
                    id, release_id, image_uri, image_path, is_front, source_uri,
                    sha256, perceptual_hash, width, height, duplicate_of_cover_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    release_id=excluded.release_id,
                    image_uri=excluded.image_uri,
                    image_path=excluded.image_path,
                    is_front=excluded.is_front,
                    source_uri=excluded.source_uri,
                    sha256=excluded.sha256,
                    perceptual_hash=excluded.perceptual_hash,
                    width=excluded.width,
                    height=excluded.height,
                    duplicate_of_cover_id=excluded.duplicate_of_cover_id
                """,
                (
                    cover.id,
                    cover.release_id,
                    cover.image_uri,
                    cover.image_path,
                    int(cover.is_front),
                    cover.source_uri,
                    cover.sha256,
                    cover.perceptual_hash,
                    cover.width,
                    cover.height,
                    duplicate_of,
                ),
            )
            connection.execute("DELETE FROM tracks WHERE release_id = ?", (release.id,))
            connection.executemany(
                """
                INSERT INTO tracks (
                    id, release_id, position, title, artist, length_ms, recording_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        track.id,
                        track.release_id,
                        track.position,
                        track.title,
                        track.artist,
                        track.length_ms,
                        track.recording_id,
                    )
                    for track in tracks
                ],
            )

    def list_covers(self) -> tuple[Cover, ...]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id, release_id, image_uri, image_path, is_front, source_uri,
                       sha256, perceptual_hash, width, height
                FROM covers
                WHERE duplicate_of_cover_id IS NULL
                ORDER BY id
                """
            ).fetchall()
        covers = []
        for row in rows:
            path = Path(row["image_path"])
            if path.is_file():
                covers.append(
                    Cover(
                        id=row["id"],
                        release_id=row["release_id"],
                        image_uri=row["image_uri"],
                        image_path=str(path),
                        is_front=bool(row["is_front"]),
                        source_uri=row["source_uri"],
                        sha256=row["sha256"],
                        perceptual_hash=row["perceptual_hash"],
                        width=row["width"],
                        height=row["height"],
                    )
                )
        return tuple(covers)

    def get_release(self, release_id: str) -> Release:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM releases WHERE id = ?", (release_id,)
            ).fetchone()
        if row is None:
            raise LookupError(f"未找到发行版本：{release_id}")
        return Release(
            id=row["id"],
            album_id=row["album_id"],
            release_date=row["release_date"],
            country=row["country"],
            status=row["status"],
            barcode=row["barcode"],
        )

    def get_album(self, album_id: str) -> Album:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM albums WHERE id = ?", (album_id,)
            ).fetchone()
        if row is None:
            raise LookupError(f"未找到专辑：{album_id}")
        return Album(
            id=row["id"],
            title=row["title"],
            artist=row["artist"],
            language=row["language"],
            first_release_year=row["first_release_year"],
            primary_type=row["primary_type"],
            source_uri=row["source_uri"],
            genre_score=row["genre_score"],
        )

    def list_tracks(self, release_id: str) -> tuple[Track, ...]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM tracks WHERE release_id = ? ORDER BY position, id",
                (release_id,),
            ).fetchall()
        return tuple(
            Track(
                id=row["id"],
                release_id=row["release_id"],
                position=row["position"],
                title=row["title"],
                artist=row["artist"],
                length_ms=row["length_ms"],
                recording_id=row["recording_id"],
            )
            for row in rows
        )

    def counts(self) -> dict[str, int]:
        with self._connect() as connection:
            return {
                table: int(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
                for table in ("albums", "releases", "covers", "tracks")
            }
