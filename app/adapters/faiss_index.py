from __future__ import annotations

import hashlib
import importlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from app.domain import Cover
from app.errors import IndexError
from app.ports import AlbumRepository, CoverEmbeddingProvider, Embedding


@dataclass(frozen=True, slots=True)
class IndexStatus:
    count: int
    reused: bool
    fingerprint: str


class FaissCoverIndex:
    VERSION = 2

    def __init__(self, index_dir: Path) -> None:
        self.index_dir = index_dir.resolve()
        self.index_path = self.index_dir / "covers.faiss"
        self.manifest_path = self.index_dir / "covers.json"
        self.vector_cache_path = self.index_dir / "covers-vectors.npz"
        self._index: Any | None = None
        self._cover_ids: tuple[str, ...] = ()
        self.status: IndexStatus | None = None

    def prepare(
        self, repository: AlbumRepository, provider: CoverEmbeddingProvider
    ) -> IndexStatus:
        covers = tuple(repository.list_covers())
        if not covers:
            raise IndexError("数据仓库中没有可建立索引的封面")
        records = self._cover_records(covers)
        fingerprint = self._fingerprint(records, provider)
        if self._load_if_current(fingerprint, provider.dimension):
            self.status = IndexStatus(len(self._cover_ids), True, fingerprint)
            return self.status

        self._build(covers, records, provider, fingerprint)
        self.status = IndexStatus(len(self._cover_ids), False, fingerprint)
        return self.status

    def search(self, query: Embedding, limit: int) -> tuple[tuple[str, float], ...]:
        if self._index is None:
            raise IndexError("FAISS 索引尚未准备")
        faiss = self._faiss()
        vector = np.asarray(query, dtype=np.float32).reshape(1, -1)
        if vector.shape[1] != self._index.d:
            raise IndexError(f"查询向量维数 {vector.shape[1]} 与索引维数 {self._index.d} 不一致")
        faiss.normalize_L2(vector)
        size = min(max(limit, 0), self._index.ntotal)
        if size == 0:
            return ()
        scores, positions = self._index.search(vector, size)
        hits = [
            (self._cover_ids[position], float(score))
            for score, position in zip(scores[0], positions[0], strict=True)
            if position >= 0
        ]
        hits.sort(key=lambda item: (-item[1], item[0]))
        return tuple(hits)

    def _cover_records(self, covers: tuple[Cover, ...]) -> tuple[dict[str, str], ...]:
        records = []
        for cover in sorted(covers, key=lambda item: item.id):
            if not cover.image_path:
                raise IndexError(f"封面 {cover.id} 没有本地文件路径")
            path = Path(cover.image_path)
            try:
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
            except OSError as error:
                raise IndexError(f"无法读取封面文件：{path}: {error}") from error
            records.append({"cover_id": cover.id, "path": str(path), "sha256": digest})
        return tuple(records)

    def _fingerprint(
        self,
        records: tuple[dict[str, str], ...],
        provider: CoverEmbeddingProvider,
    ) -> str:
        payload = json.dumps(
            {
                "version": self.VERSION,
                "model_id": provider.model_id,
                "dimension": provider.dimension,
                "covers": records,
            },
            ensure_ascii=False,
            sort_keys=True,
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def _load_if_current(self, fingerprint: str, dimension: int) -> bool:
        if not self.index_path.is_file() or not self.manifest_path.is_file():
            return False
        try:
            faiss = self._faiss()
            manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            if manifest.get("fingerprint") != fingerprint:
                return False
            cover_ids = tuple(manifest["cover_ids"])
            index = faiss.read_index(str(self.index_path))
            if index.d != dimension or index.ntotal != len(cover_ids):
                return False
        except (OSError, ValueError, KeyError, json.JSONDecodeError, RuntimeError):
            return False
        self._index = index
        self._cover_ids = cover_ids
        return True

    def _build(
        self,
        covers: tuple[Cover, ...],
        records: tuple[dict[str, str], ...],
        provider: CoverEmbeddingProvider,
        fingerprint: str,
    ) -> None:
        ordered = tuple(sorted(covers, key=lambda item: item.id))
        try:
            faiss = self._faiss()
            record_by_id = {record["cover_id"]: record for record in records}
            cached = self._load_vector_cache(provider)
            vectors: dict[str, np.ndarray] = {}
            pending: list[Cover] = []
            for cover in ordered:
                record = record_by_id[cover.id]
                cached_item = cached.get(cover.id)
                if cached_item and cached_item[0] == record["sha256"]:
                    vectors[cover.id] = cached_item[1]
                else:
                    pending.append(cover)
            if pending:
                generated = provider.embed_covers(pending)
                if len(generated) != len(pending):
                    raise IndexError(
                        f"模型返回 {len(generated)} 个封面向量，期望 {len(pending)} 个"
                    )
                for cover, vector in zip(pending, generated, strict=True):
                    vectors[cover.id] = np.asarray(vector, dtype=np.float32)
            matrix = np.asarray([vectors[cover.id] for cover in ordered], dtype=np.float32)
            if matrix.shape != (len(ordered), provider.dimension):
                raise IndexError(
                    f"封面向量矩阵形状无效：{matrix.shape}，"
                    f"期望 ({len(ordered)}, {provider.dimension})"
                )
            faiss.normalize_L2(matrix)
            index = faiss.IndexFlatIP(provider.dimension)
            index.add(matrix)
            self.index_dir.mkdir(parents=True, exist_ok=True)
            temporary_index = self.index_path.with_suffix(".faiss.tmp")
            temporary_manifest = self.manifest_path.with_suffix(".json.tmp")
            temporary_vectors = self.vector_cache_path.with_suffix(".npz.tmp")
            faiss.write_index(index, str(temporary_index))
            temporary_manifest.write_text(
                json.dumps(
                    {
                        "version": self.VERSION,
                        "fingerprint": fingerprint,
                        "model_id": provider.model_id,
                        "dimension": provider.dimension,
                        "cover_ids": [cover.id for cover in ordered],
                        "covers": [record_by_id[cover.id] for cover in ordered],
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            with temporary_vectors.open("wb") as stream:
                np.savez_compressed(
                    stream,
                    model_id=np.asarray(provider.model_id),
                    dimension=np.asarray(provider.dimension, dtype=np.int64),
                    cover_ids=np.asarray([cover.id for cover in ordered]),
                    content_hashes=np.asarray(
                        [record_by_id[cover.id]["sha256"] for cover in ordered]
                    ),
                    vectors=matrix,
                )
            temporary_index.replace(self.index_path)
            temporary_manifest.replace(self.manifest_path)
            temporary_vectors.replace(self.vector_cache_path)
        except IndexError:
            raise
        except Exception as error:
            raise IndexError(f"无法建立 FAISS 封面索引：{error}") from error
        self._index = index
        self._cover_ids = tuple(cover.id for cover in ordered)

    def _load_vector_cache(
        self, provider: CoverEmbeddingProvider
    ) -> dict[str, tuple[str, np.ndarray]]:
        if not self.vector_cache_path.is_file():
            return {}
        try:
            with np.load(self.vector_cache_path, allow_pickle=False) as payload:
                if str(payload["model_id"].item()) != provider.model_id:
                    return {}
                if int(payload["dimension"].item()) != provider.dimension:
                    return {}
                cover_ids = payload["cover_ids"]
                hashes = payload["content_hashes"]
                vectors = payload["vectors"]
                if vectors.shape != (len(cover_ids), provider.dimension):
                    return {}
                return {
                    str(cover_id): (str(content_hash), np.asarray(vector, dtype=np.float32))
                    for cover_id, content_hash, vector in zip(
                        cover_ids, hashes, vectors, strict=True
                    )
                }
        except (OSError, ValueError, KeyError):
            return {}

    @staticmethod
    def _faiss() -> Any:
        # macOS arm64 上先加载 FAISS、后加载 PyTorch 可能触发底层运行库冲突。
        # 生产运行时先构造 CLIP provider，再在这里延迟导入 FAISS。
        return importlib.import_module("faiss")
