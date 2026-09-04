from __future__ import annotations

import argparse
import os
from pathlib import Path

from app.adapters.musicbrainz import (
    DEFAULT_FORCE_TITLES,
    DEFAULT_QUERY,
    MusicBrainzClient,
    MusicBrainzImporter,
    RateLimitedHttpClient,
)
from app.adapters.sqlite_repository import SQLiteAlbumRepository
from app.settings import Settings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="导入真实的日本摇滚专辑元数据和封面")
    parser.add_argument("--limit", type=int, default=100, help="数据库中希望达到的专辑数量")
    parser.add_argument("--query", default=DEFAULT_QUERY, help="MusicBrainz release-group 查询")
    parser.add_argument(
        "--force-title",
        action="append",
        dest="force_titles",
        help="优先导入的精确专辑标题；可重复提供",
    )
    parser.add_argument(
        "--user-agent",
        default=os.environ.get(
            "SEEK_MUSICBRAINZ_USER_AGENT",
            "Seek/0.1 (local-first album-cover retrieval prototype)",
        ),
        help="符合 MusicBrainz 要求的应用 User-Agent",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = Settings.from_environment()
    cover_dir = Path(os.environ.get("SEEK_COVER_DIR", settings.project_root / "data/covers"))
    repository = SQLiteAlbumRepository(settings.database_path, require_data=False)
    client = MusicBrainzClient(RateLimitedHttpClient(args.user_agent))
    importer = MusicBrainzImporter(client, repository, cover_dir)
    force_titles = tuple(args.force_titles) if args.force_titles else DEFAULT_FORCE_TITLES

    def progress(
        examined, imported, present, no_cover, out_of_scope, failed, candidate, outcome
    ):
        title = candidate.get("title") or candidate.get("id") or "未知专辑"
        print(
            f"[{examined}] {title}: {outcome} "
            f"(新增 {imported} / 已有 {present} / 无封面 {no_cover} / "
            f"超范围 {out_of_scope} / 失败 {failed})",
            flush=True,
        )

    summary = importer.import_collection(
        limit=args.limit,
        query=args.query,
        force_titles=force_titles,
        progress=progress,
    )
    print(
        "导入结束："
        f"新增 {summary.imported}，已有 {summary.already_present}，"
        f"无封面 {summary.without_cover}，超范围 {summary.out_of_scope}，"
        f"失败 {summary.failed}，"
        f"共检查 {summary.examined}。"
    )
    print(f"数据库：{settings.database_path}")
    print(f"封面目录：{cover_dir.resolve()}")


if __name__ == "__main__":
    main()
