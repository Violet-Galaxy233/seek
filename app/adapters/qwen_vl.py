from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

from app.domain import Cover
from app.errors import ModelUnavailableError
from app.ports import Embedding

MODEL_ID = "Qwen/Qwen3-VL-Embedding-2B"
QUERY_INSTRUCTION = (
    "Retrieve album cover images matching the user's uncertain visual memory."
)


class Qwen3VLEmbeddingProvider:
    """本地 Qwen3-VL-Embedding 双塔适配器，不包含任何在线推理回退。"""

    def __init__(
        self,
        model_path: Path,
        *,
        dimension: int = 2048,
        max_pixels: int = 512 * 512,
        batch_size: int = 2,
        device: str = "auto",
    ) -> None:
        self.model_path = model_path.resolve()
        if not self.model_path.is_dir():
            raise ModelUnavailableError(
                f"本地 Qwen 模型不存在：{self.model_path}。"
                "请先运行 `uv run python -m app.download_models --model qwen`；"
                "Seek 不会自动切换到外部模型 API。"
            )
        if not 64 <= dimension <= 2048:
            raise ValueError("Qwen3-VL-Embedding-2B 向量维数必须位于 64 到 2048 之间")
        if batch_size < 1:
            raise ValueError("batch_size 必须大于 0")

        self._dimension = dimension
        self.max_pixels = max_pixels
        self.min_pixels = 64 * 64
        self.batch_size = batch_size

        try:
            import torch
            from qwen_vl_utils import process_vision_info
            from torch.nn import functional
            from transformers.models.qwen3_vl.modeling_qwen3_vl import (
                Qwen3VLModel,
                Qwen3VLPreTrainedModel,
            )
            from transformers.models.qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor

            selected_device = self._select_device(torch, device)
            selected_dtype = self._select_dtype(torch, selected_device)

            class Qwen3VLForEmbedding(Qwen3VLPreTrainedModel):
                _checkpoint_conversion_mapping: ClassVar[dict[str, str]] = {}

                def __init__(self, config):
                    super().__init__(config)
                    self.model = Qwen3VLModel(config)
                    self.post_init()

                def forward(self, **inputs):
                    return self.model(**inputs)

            self._torch = torch
            self._functional = functional
            self._process_vision_info = process_vision_info
            self._device = selected_device
            self._model: Any = Qwen3VLForEmbedding.from_pretrained(
                str(self.model_path),
                local_files_only=True,
                trust_remote_code=True,
                dtype=selected_dtype,
            ).to(selected_device)
            self._processor: Any = Qwen3VLProcessor.from_pretrained(
                str(self.model_path), local_files_only=True, padding_side="right"
            )
            self._model.eval()
            native_dimension = int(self._model.config.text_config.hidden_size)
        except Exception as error:
            raise ModelUnavailableError(
                "本地 Qwen3-VL-Embedding 模型无法加载。"
                f"设备请求为 {device}，不会切换到在线 API：{error}"
            ) from error

        if dimension > native_dimension:
            raise ModelUnavailableError(
                f"请求向量维数 {dimension} 超过模型原生维数 {native_dimension}"
            )
        self._native_dimension = native_dimension

    @staticmethod
    def _select_device(torch: Any, requested: str) -> str:
        if requested != "auto":
            return requested
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    @staticmethod
    def _select_dtype(torch: Any, device: str) -> Any:
        if device == "mps":
            return torch.float16
        if device == "cuda":
            return torch.bfloat16
        return torch.float32

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def model_id(self) -> str:
        config_path = self.model_path / "config.json"
        config_hash = "missing"
        if config_path.is_file():
            config_hash = hashlib.sha256(config_path.read_bytes()).hexdigest()[:12]
        return (
            f"{MODEL_ID}@{config_hash}:dim={self.dimension}:"
            f"max_pixels={self.max_pixels}"
        )

    @property
    def device(self) -> str:
        return self._device

    def embed_text(self, text: str) -> Embedding:
        normalized = " ".join(text.split())
        if not normalized:
            raise ValueError("待编码文本不能为空")
        return self._encode(({"text": normalized, "instruction": QUERY_INSTRUCTION},))[0]

    def embed_cover(self, cover: Cover) -> Embedding:
        return self.embed_covers((cover,))[0]

    def embed_covers(self, covers: Sequence[Cover]) -> Sequence[Embedding]:
        results: list[Embedding] = []
        for start in range(0, len(covers), self.batch_size):
            batch = covers[start : start + self.batch_size]
            items = []
            for cover in batch:
                if not cover.image_path:
                    raise ModelUnavailableError(f"封面 {cover.id} 没有本地文件路径")
                path = Path(cover.image_path)
                if not path.is_file():
                    raise ModelUnavailableError(f"封面文件不存在：{path}")
                items.append({"image": str(path)})
            results.extend(self._encode(tuple(items)))
        return tuple(results)

    def _encode(self, items: Sequence[dict[str, str]]) -> tuple[Embedding, ...]:
        conversations = [self._conversation(item) for item in items]
        try:
            prompts = self._processor.apply_chat_template(
                conversations,
                add_generation_prompt=True,
                tokenize=False,
            )
            vision_result = self._process_vision_info(
                conversations,
                image_patch_size=16,
                return_video_metadata=True,
                return_video_kwargs=True,
            )
            images, video_inputs, video_kwargs = vision_result
            if video_inputs is not None:
                videos, video_metadata = zip(*video_inputs, strict=True)
                videos = list(videos)
                video_metadata = list(video_metadata)
            else:
                videos = video_metadata = None
            processed = self._processor(
                text=prompts,
                images=images,
                videos=videos,
                video_metadata=video_metadata,
                truncation=True,
                max_length=8192,
                padding=True,
                do_resize=False,
                return_tensors="pt",
                **video_kwargs,
            )
            processed = {key: value.to(self._model.device) for key, value in processed.items()}
            with self._torch.no_grad():
                outputs = self._model(**processed)
                hidden = outputs.last_hidden_state
                mask = processed["attention_mask"]
                positions = mask.shape[1] - mask.flip(dims=[1]).argmax(dim=1) - 1
                rows = self._torch.arange(hidden.shape[0], device=hidden.device)
                embeddings = hidden[rows, positions, : self.dimension]
                embeddings = self._functional.normalize(embeddings, p=2, dim=-1)
            matrix = embeddings.detach().float().cpu().numpy()
        except Exception as error:
            raise ModelUnavailableError(
                f"Qwen3-VL-Embedding 在本地设备 {self.device} 上编码失败：{error}"
            ) from error
        return tuple(
            tuple(float(value) for value in np.asarray(vector, dtype=np.float32))
            for vector in matrix
        )

    def _conversation(self, item: dict[str, str]) -> list[dict[str, Any]]:
        instruction = item.get("instruction", "Represent the user's input.")
        content: list[dict[str, Any]] = []
        if "image" in item:
            content.append(
                {
                    "type": "image",
                    "image": f"file://{Path(item['image']).resolve()}",
                    "min_pixels": self.min_pixels,
                    "max_pixels": self.max_pixels,
                }
            )
        if "text" in item:
            content.append({"type": "text", "text": item["text"]})
        return [
            {"role": "system", "content": [{"type": "text", "text": instruction}]},
            {"role": "user", "content": content},
        ]
