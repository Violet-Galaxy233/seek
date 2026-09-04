from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from typing import Any

from app.adapters.faiss_index import FaissCoverIndex
from app.adapters.file_catalog import FileAlbumRepository
from app.adapters.local_clip import LocalClipEmbeddingProvider
from app.adapters.qwen_vl import Qwen3VLEmbeddingProvider
from app.adapters.sqlite_repository import SQLiteAlbumRepository
from app.errors import SeekError
from app.indexed_search import IndexedSearchService
from app.settings import Settings


@dataclass(frozen=True, slots=True)
class RuntimeStatus:
    state: str
    message: str
    index_count: int = 0
    index_reused: bool | None = None


class LocalRuntime:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._service: IndexedSearchService | None = None
        self._status = RuntimeStatus("starting", "正在加载本地模型并准备索引")
        self._lock = Lock()

    @property
    def status(self) -> RuntimeStatus:
        return self._status

    def initialize(self) -> RuntimeStatus:
        with self._lock:
            if self._service is not None:
                return self._status
            try:
                if self.settings.data_source == "sqlite":
                    repository = SQLiteAlbumRepository(self.settings.database_path)
                elif self.settings.data_source == "testdata":
                    repository = FileAlbumRepository(
                        self.settings.catalog_path, self.settings.cover_dir
                    )
                else:
                    raise SeekError(
                        f"未知 SEEK_DATA_SOURCE：{self.settings.data_source}，"
                        "可选值为 sqlite 或 testdata"
                    )

                if self.settings.embedding_model == "qwen":
                    embeddings = Qwen3VLEmbeddingProvider(
                        self.settings.qwen_model_path,
                        dimension=self.settings.embedding_dimension,
                        max_pixels=self.settings.qwen_max_pixels,
                        device=self.settings.device,
                    )
                elif self.settings.embedding_model == "clip":
                    embeddings = LocalClipEmbeddingProvider(
                        self.settings.image_model_path,
                        self.settings.text_model_path,
                        self.settings.device,
                    )
                else:
                    raise SeekError(
                        f"未知 SEEK_EMBEDDING_MODEL：{self.settings.embedding_model}，"
                        "可选值为 qwen 或 clip"
                    )
                service = IndexedSearchService(
                    repository, embeddings, FaissCoverIndex(self.settings.index_dir)
                )
                index_status = service.prepare()
                self._service = service
                action = "复用" if index_status.reused else "建立"
                device_message = getattr(embeddings, "device", self.settings.device)
                self._status = RuntimeStatus(
                    "ready",
                    f"已{action}本地 FAISS 索引，共 {index_status.count} 张封面；"
                    f"模型运行设备：{device_message}",
                    index_status.count,
                    index_status.reused,
                )
            except SeekError as error:
                self._status = RuntimeStatus("error", str(error))
            return self._status

    def get_service(self) -> IndexedSearchService:
        if self._service is None:
            status = self.initialize()
            if self._service is None:
                raise SeekError(status.message)
        return self._service


class StaticRuntime:
    """测试使用：将已有搜索服务包装成运行时。"""

    def __init__(self, service: Any) -> None:
        self._service = service
        self._status = RuntimeStatus("ready", "测试搜索服务已就绪")

    @property
    def status(self) -> RuntimeStatus:
        return self._status

    def initialize(self) -> RuntimeStatus:
        return self._status

    def get_service(self) -> Any:
        return self._service
