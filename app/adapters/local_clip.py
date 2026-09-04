from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from app.domain import Cover
from app.errors import ModelUnavailableError
from app.ports import Embedding


class LocalClipEmbeddingProvider:
    """完全离线的 CLIP 图像编码器和多语言 CLIP 文本编码器。"""

    def __init__(self, image_model_path: Path, text_model_path: Path, device: str = "auto") -> None:
        self.image_model_path = image_model_path.resolve()
        self.text_model_path = text_model_path.resolve()
        missing = [
            str(path)
            for path in (self.image_model_path, self.text_model_path)
            if not path.is_dir()
        ]
        if missing:
            joined = "、".join(missing)
            raise ModelUnavailableError(
                f"本地 CLIP 模型不存在：{joined}。"
                "请先运行 `uv run python -m app.download_models`；"
                "Seek 不会自动切换到外部模型 API。"
            )

        try:
            from sentence_transformers import SentenceTransformer

            selected_device = self._select_device(device)
            self._image_model: Any = SentenceTransformer(
                str(self.image_model_path), device=selected_device, local_files_only=True
            )
            self._text_model: Any = SentenceTransformer(
                str(self.text_model_path), device=selected_device, local_files_only=True
            )
            image_dimension = self._image_model.get_embedding_dimension()
            text_dimension = self._text_model.get_embedding_dimension()
        except Exception as error:
            raise ModelUnavailableError(
                "本地 CLIP 模型无法加载。请确认模型文件完整且与当前环境兼容："
                f"{error}"
            ) from error

        if image_dimension is None or text_dimension is None or image_dimension != text_dimension:
            raise ModelUnavailableError(
                f"图像模型与文本模型向量维数不一致：{image_dimension} != {text_dimension}"
            )
        self._dimension = image_dimension

    @staticmethod
    def _select_device(requested: str) -> str:
        if requested != "auto":
            return requested
        try:
            import torch

            return "mps" if torch.backends.mps.is_available() else "cpu"
        except ImportError:
            return "cpu"

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def model_id(self) -> str:
        return (
            "sentence-transformers/clip-ViT-B-32"
            "+sentence-transformers/clip-ViT-B-32-multilingual-v1"
        )

    def embed_text(self, text: str) -> Embedding:
        try:
            vector = self._text_model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
        except Exception as error:
            raise ModelUnavailableError(f"本地 CLIP 无法编码查询：{error}") from error
        return self._as_embedding(vector)

    def embed_cover(self, cover: Cover) -> Embedding:
        if not cover.image_path:
            raise ModelUnavailableError(f"封面 {cover.id} 没有本地文件路径")
        path = Path(cover.image_path)
        if not path.is_file():
            raise ModelUnavailableError(f"封面文件不存在：{path}")
        try:
            with Image.open(path) as image:
                vector = self._image_model.encode(
                    image.convert("RGB"), convert_to_numpy=True, normalize_embeddings=True
                )
        except Exception as error:
            raise ModelUnavailableError(f"本地 CLIP 无法编码封面 {path}：{error}") from error
        return self._as_embedding(vector)

    def embed_covers(self, covers: Sequence[Cover]) -> Sequence[Embedding]:
        images: list[Image.Image] = []
        try:
            for cover in covers:
                if not cover.image_path:
                    raise ModelUnavailableError(f"封面 {cover.id} 没有本地文件路径")
                path = Path(cover.image_path)
                if not path.is_file():
                    raise ModelUnavailableError(f"封面文件不存在：{path}")
                with Image.open(path) as image:
                    images.append(image.convert("RGB"))
            vectors = self._image_model.encode(
                images, convert_to_numpy=True, normalize_embeddings=True
            )
            return tuple(self._as_embedding(vector) for vector in vectors)
        except ModelUnavailableError:
            raise
        except Exception as error:
            raise ModelUnavailableError(f"本地 CLIP 无法批量编码封面：{error}") from error

    def _as_embedding(self, vector: Any) -> Embedding:
        array = np.asarray(vector, dtype=np.float32).reshape(-1)
        if array.shape[0] != self.dimension:
            raise ModelUnavailableError(
                f"模型返回了错误的向量维数：{array.shape[0]}，期望 {self.dimension}"
            )
        return tuple(float(value) for value in array)
