from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, ClassVar, Protocol
from urllib.parse import urlencode, urljoin

import numpy as np
import requests
from PIL import Image, UnidentifiedImageError

from app.adapters.sqlite_repository import SQLiteAlbumRepository
from app.domain import Album, Cover, Release, Track
from app.errors import ImportDataError, RemoteDataError

MUSICBRAINZ_ROOT = "https://musicbrainz.org/ws/2"
COVER_ART_ROOT = "https://coverartarchive.org"
DEFAULT_QUERY = (
    '(tag:"j-rock" OR tag:"japanese rock" OR tag:"visual kei" '
    'OR tag:"japanese alternative rock" OR tag:"japanese indie rock" '
    'OR tag:"japanese punk rock") '
    "AND (primarytype:album OR primarytype:ep) "
    "AND firstreleasedate:[1980-01-01 TO 2026-12-31]"
)
DEFAULT_FORCE_TITLES = ("健全な社会", "Underlight & Aftertime")


class MusicDataClient(Protocol):
    def search_release_groups(self, query: str, limit: int, offset: int) -> dict[str, Any]: ...

    def get_release_group(self, release_group_id: str) -> dict[str, Any]: ...

    def get_release(self, release_id: str) -> dict[str, Any]: ...

    def download_front_cover(self, release_id: str) -> bytes | None: ...


class RateLimitedHttpClient:
    """遵循 MusicBrainz 公共服务约束的串行 HTTP 客户端。"""

    RETRYABLE: ClassVar[set[int]] = {429, 500, 502, 503, 504}

    def __init__(
        self,
        user_agent: str,
        *,
        minimum_interval: float = 1.05,
        retries: int = 4,
        timeout: float = 30.0,
    ) -> None:
        if not user_agent.strip():
            raise ValueError("MusicBrainz User-Agent 不能为空")
        self.user_agent = user_agent
        self.minimum_interval = minimum_interval
        self.retries = retries
        self.timeout = timeout
        self._last_request_at = 0.0
        self._session = requests.Session()
        self._session.headers.update(
            {
                "User-Agent": self.user_agent,
                "Accept": "application/json,image/*;q=0.9,*/*;q=0.1",
            }
        )

    def get_json(self, url: str) -> dict[str, Any]:
        payload = self._get(url, allow_not_found=False)
        try:
            result = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise RemoteDataError(f"远端返回了无效 JSON：{url}: {error}") from error
        if not isinstance(result, dict):
            raise RemoteDataError(f"远端 JSON 不是对象：{url}")
        return result

    def get_bytes_or_none(self, url: str) -> bytes | None:
        if url.startswith(f"{COVER_ART_ROOT}/"):
            response = self._get_response(
                url, allow_not_found=True, allow_redirects=False
            )
            if response is None:
                return None
            if response.is_redirect or response.is_permanent_redirect:
                redirect_url = urljoin(url, response.headers["Location"])
                # archive.org 的入口在部分 macOS 网络环境中 TLS 协商失败；其
                # HTTP 入口会立即跳转到带 TLS 的具体镜像节点。只对 CAA 返回的
                # 精确主机应用这一兼容路径，避免影响其他下载。
                if redirect_url.startswith("https://archive.org/"):
                    redirect_url = "http://archive.org/" + redirect_url.removeprefix(
                        "https://archive.org/"
                    )
                return self._get(redirect_url, allow_not_found=True)
            return response.content
        return self._get(url, allow_not_found=True)

    def _get(self, url: str, *, allow_not_found: bool) -> bytes | None:
        response = self._get_response(
            url, allow_not_found=allow_not_found, allow_redirects=True
        )
        return None if response is None else response.content

    def _get_response(
        self, url: str, *, allow_not_found: bool, allow_redirects: bool
    ) -> requests.Response | None:
        for attempt in range(self.retries + 1):
            elapsed = time.monotonic() - self._last_request_at
            if elapsed < self.minimum_interval:
                time.sleep(self.minimum_interval - elapsed)
            try:
                self._last_request_at = time.monotonic()
                response = self._session.get(
                    url, timeout=self.timeout, allow_redirects=allow_redirects
                )
                if allow_not_found and response.status_code == 404:
                    return None
                if response.status_code in self.RETRYABLE:
                    if attempt >= self.retries:
                        raise RemoteDataError(
                            f"远端请求失败 HTTP {response.status_code}：{url}"
                        )
                else:
                    response.raise_for_status()
                    return response
            except requests.RequestException as error:
                if attempt >= self.retries:
                    raise RemoteDataError(f"无法连接远端服务：{url}: {error}") from error
            time.sleep(min(2**attempt, 8))
        raise RemoteDataError(f"远端请求重试耗尽：{url}")


class MusicBrainzClient:
    def __init__(self, http: RateLimitedHttpClient) -> None:
        self.http = http

    def search_release_groups(self, query: str, limit: int, offset: int) -> dict[str, Any]:
        params = urlencode({"query": query, "limit": limit, "offset": offset, "fmt": "json"})
        return self.http.get_json(f"{MUSICBRAINZ_ROOT}/release-group/?{params}")

    def get_release_group(self, release_group_id: str) -> dict[str, Any]:
        params = urlencode({"inc": "releases+artist-credits+tags", "fmt": "json"})
        return self.http.get_json(
            f"{MUSICBRAINZ_ROOT}/release-group/{release_group_id}?{params}"
        )

    def get_release(self, release_id: str) -> dict[str, Any]:
        params = urlencode(
            {"inc": "recordings+artist-credits+release-groups+media", "fmt": "json"}
        )
        return self.http.get_json(f"{MUSICBRAINZ_ROOT}/release/{release_id}?{params}")

    def download_front_cover(self, release_id: str) -> bytes | None:
        return self.http.get_bytes_or_none(
            f"{COVER_ART_ROOT}/release/{release_id}/front-500"
        )


@dataclass(frozen=True, slots=True)
class ImportSummary:
    imported: int
    already_present: int
    without_cover: int
    out_of_scope: int
    failed: int
    examined: int


class MusicBrainzImporter:
    def __init__(
        self,
        client: MusicDataClient,
        repository: SQLiteAlbumRepository,
        cover_dir: Path,
    ) -> None:
        self.client = client
        self.repository = repository
        self.cover_dir = cover_dir.resolve()
        self.cover_dir.mkdir(parents=True, exist_ok=True)

    def import_collection(
        self,
        *,
        limit: int,
        query: str = DEFAULT_QUERY,
        force_titles: tuple[str, ...] = DEFAULT_FORCE_TITLES,
        progress: Any | None = None,
    ) -> ImportSummary:
        if limit < 1:
            raise ValueError("limit 必须大于 0")
        imported = already_present = without_cover = out_of_scope = failed = examined = 0
        needed = max(limit - len(self.repository.list_covers()), 0)
        if needed == 0:
            return ImportSummary(0, 0, 0, 0, 0, 0)
        seen: set[str] = set()

        candidates = self._candidate_stream(query, force_titles)
        while imported < needed:
            try:
                candidate = next(candidates)
            except StopIteration:
                break
            release_group_id = str(candidate.get("id", ""))
            if not release_group_id or release_group_id in seen:
                continue
            seen.add(release_group_id)
            examined += 1
            if not self._is_in_scope(candidate):
                out_of_scope += 1
                outcome = "超出范围"
            elif self.repository.album_has_cover(release_group_id):
                already_present += 1
                outcome = "已存在"
            else:
                try:
                    saved = self._import_release_group(candidate)
                except (ImportDataError, RemoteDataError, OSError) as error:
                    failed += 1
                    outcome = f"失败：{error}"
                else:
                    if saved:
                        imported += 1
                        outcome = "已导入"
                    else:
                        without_cover += 1
                        outcome = "无可用封面"
            if progress:
                progress(
                    examined,
                    imported,
                    already_present,
                    without_cover,
                    out_of_scope,
                    failed,
                    candidate,
                    outcome,
                )
            if examined >= max(needed * 12, 200):
                break

        return ImportSummary(
            imported, already_present, without_cover, out_of_scope, failed, examined
        )

    @staticmethod
    def _is_in_scope(candidate: dict[str, Any]) -> bool:
        primary_type = str(candidate.get("primary-type") or "").casefold()
        if primary_type not in {"album", "ep"}:
            return False
        excluded_secondary_types = {
            "compilation",
            "live",
            "remix",
            "soundtrack",
            "dj-mix",
            "mixtape/street",
            "demo",
            "interview",
            "audiobook",
            "audio drama",
            "spokenword",
        }
        secondary_types = {
            str(value).casefold() for value in candidate.get("secondary-types") or []
        }
        return not secondary_types.intersection(excluded_secondary_types)

    def _candidate_stream(self, query: str, force_titles: tuple[str, ...]):
        for title in force_titles:
            exact_query = f'releasegroup:"{title.replace(chr(34), "")}"'
            payload = self.client.search_release_groups(exact_query, 25, 0)
            matches = [
                item
                for item in payload.get("release-groups", [])
                if str(item.get("title", "")).casefold() == title.casefold()
            ]
            yield from sorted(matches, key=lambda value: -int(value.get("score", 0)))

        offset = 0
        while True:
            payload = self.client.search_release_groups(query, 100, offset)
            groups = payload.get("release-groups", [])
            if not isinstance(groups, list) or not groups:
                return
            yield from groups
            offset += len(groups)
            if offset >= int(payload.get("release-group-count", offset)):
                return

    def _import_release_group(self, summary: dict[str, Any]) -> bool:
        release_group_id = self._required_text(summary, "id")
        details = self.client.get_release_group(release_group_id)
        releases = details.get("releases", [])
        if not isinstance(releases, list):
            raise ImportDataError(f"release group {release_group_id} 的 releases 无效")

        for release_data in self._ordered_releases(releases):
            release_id = self._required_text(release_data, "id")
            image_bytes = self.client.download_front_cover(release_id)
            if not image_bytes:
                continue
            image_info = self._inspect_image(image_bytes, release_group_id)
            release_details = self.client.get_release(release_id)
            image_path = self._save_image(release_group_id, image_bytes, image_info[0])
            album, release, cover, tracks = self._build_bundle(
                summary,
                details,
                release_details,
                image_path,
                image_bytes,
                image_info,
            )
            self.repository.save_bundle(album, release, cover, tracks)
            return True
        return False

    @staticmethod
    def _ordered_releases(releases: list[dict[str, Any]]) -> list[dict[str, Any]]:
        official = [item for item in releases if item.get("status") == "Official"]
        usable = official or releases
        has_cover_metadata = any("cover-art-archive" in item for item in usable)
        if has_cover_metadata:
            usable = [
                item
                for item in usable
                if (item.get("cover-art-archive") or {}).get("front")
            ]

        def rank(item: dict[str, Any]) -> tuple[int, int, str, str]:
            cover_info = item.get("cover-art-archive") or {}
            return (
                0 if item.get("country") == "JP" else 1,
                0 if cover_info.get("front") else 1,
                str(item.get("date") or "9999"),
                str(item.get("id") or ""),
            )

        # MusicBrainz 已明确标记封面时只尝试 Front 版本；旧记录缺少该字段时
        # 最多探测前三个代表版本，避免一个无封面作品触发数十次网络请求。
        maximum = len(usable) if has_cover_metadata else 3
        return sorted(usable, key=rank)[:maximum]

    def _build_bundle(
        self,
        summary: dict[str, Any],
        group: dict[str, Any],
        release_data: dict[str, Any],
        image_path: Path,
        image_bytes: bytes,
        image_info: tuple[str, int, int, str],
    ) -> tuple[Album, Release, Cover, tuple[Track, ...]]:
        album_id = self._required_text(summary, "id")
        release_id = self._required_text(release_data, "id")
        first_date = str(summary.get("first-release-date") or "")
        year = int(first_date[:4]) if len(first_date) >= 4 and first_date[:4].isdigit() else None
        tags = group.get("tags") or summary.get("tags") or []
        genre_score = self._genre_score(tags)
        text_representation = release_data.get("text-representation") or {}
        album = Album(
            id=album_id,
            title=str(summary.get("title") or group.get("title") or "未知专辑"),
            artist=self._artist_name(group.get("artist-credit") or summary.get("artist-credit")),
            language=str(text_representation.get("language") or "und"),
            first_release_year=year,
            primary_type=str(summary.get("primary-type") or group.get("primary-type") or "Album"),
            source_uri=f"https://musicbrainz.org/release-group/{album_id}",
            genre_score=genre_score,
        )
        release = Release(
            id=release_id,
            album_id=album_id,
            release_date=release_data.get("date"),
            country=release_data.get("country"),
            status=release_data.get("status"),
            barcode=release_data.get("barcode"),
        )
        extension, width, height, perceptual_hash = image_info
        del extension
        cover = Cover(
            id=f"{release_id}:front",
            release_id=release_id,
            image_uri=f"/covers/{image_path.name}",
            image_path=str(image_path),
            is_front=True,
            source_uri=f"{COVER_ART_ROOT}/release/{release_id}/front-500",
            sha256=hashlib.sha256(image_bytes).hexdigest(),
            perceptual_hash=perceptual_hash,
            width=width,
            height=height,
        )
        tracks = self._tracks(release_data, release_id, album.artist)
        return album, release, cover, tracks

    @staticmethod
    def _tracks(
        release_data: dict[str, Any], release_id: str, default_artist: str
    ) -> tuple[Track, ...]:
        result = []
        sequence = 0
        for medium in release_data.get("media") or []:
            for item in medium.get("tracks") or []:
                sequence += 1
                recording = item.get("recording") or {}
                result.append(
                    Track(
                        id=str(item.get("id") or f"{release_id}:{sequence}"),
                        release_id=release_id,
                        position=sequence,
                        title=str(item.get("title") or recording.get("title") or "未知曲目"),
                        artist=MusicBrainzImporter._artist_name(
                            item.get("artist-credit") or recording.get("artist-credit")
                        )
                        or default_artist,
                        length_ms=item.get("length") or recording.get("length"),
                        recording_id=recording.get("id"),
                    )
                )
        return tuple(result)

    @staticmethod
    def _artist_name(artist_credit: Any) -> str:
        if not isinstance(artist_credit, list):
            return "未知艺人"
        pieces = []
        for credit in artist_credit:
            if isinstance(credit, str):
                pieces.append(credit)
                continue
            if not isinstance(credit, dict):
                continue
            artist = credit.get("artist") or {}
            pieces.append(str(credit.get("name") or artist.get("name") or ""))
            pieces.append(str(credit.get("joinphrase") or ""))
        return "".join(pieces).strip() or "未知艺人"

    @staticmethod
    def _genre_score(tags: Any) -> float:
        accepted = {
            "j-rock",
            "japanese rock",
            "rock",
            "alternative rock",
            "indie rock",
            "visual kei",
        }
        if not isinstance(tags, list):
            return 0.0
        votes = sum(
            max(int(tag.get("count", 0)), 1)
            for tag in tags
            if str(tag.get("name", "")).casefold() in accepted
        )
        return min(votes / 10.0, 1.0)

    @staticmethod
    def _inspect_image(payload: bytes, album_id: str) -> tuple[str, int, int, str]:
        try:
            with Image.open(BytesIO(payload)) as image:
                image.verify()
            with Image.open(BytesIO(payload)) as image:
                rgb = image.convert("RGB")
                width, height = rgb.size
                if width < 128 or height < 128:
                    raise ImportDataError(f"专辑 {album_id} 的封面尺寸过小：{width}x{height}")
                extension = {
                    "JPEG": ".jpg",
                    "PNG": ".png",
                    "WEBP": ".webp",
                }.get(image.format or "", ".img")
                gray = rgb.resize((9, 8)).convert("L")
                values = np.asarray(gray, dtype=np.int16)
                differences = values[:, 1:] > values[:, :-1]
                bits = "".join("1" if value else "0" for value in differences.reshape(-1))
                perceptual_hash = f"{int(bits, 2):016x}"
        except (UnidentifiedImageError, OSError) as error:
            raise ImportDataError(f"专辑 {album_id} 的封面文件无效：{error}") from error
        return extension, width, height, perceptual_hash

    def _save_image(
        self, album_id: str, payload: bytes, extension: str
    ) -> Path:
        path = self.cover_dir / f"{album_id}{extension}"
        temporary = path.with_suffix(f"{extension}.part")
        temporary.write_bytes(payload)
        temporary.replace(path)
        return path

    @staticmethod
    def _required_text(payload: dict[str, Any], key: str) -> str:
        value = payload.get(key)
        if not isinstance(value, str) or not value:
            raise ImportDataError(f"MusicBrainz 记录缺少字段：{key}")
        return value
