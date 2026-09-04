import pytest

from app.adapters.qwen_vl import Qwen3VLEmbeddingProvider
from app.errors import ModelUnavailableError


def test_missing_qwen_model_is_explicit_and_never_mentions_api_fallback(tmp_path) -> None:
    with pytest.raises(ModelUnavailableError) as caught:
        Qwen3VLEmbeddingProvider(tmp_path / "missing")

    message = str(caught.value)
    assert "本地 Qwen 模型不存在" in message
    assert "download_models --model qwen" in message
    assert "不会自动切换到外部模型 API" in message


def test_qwen_dimension_is_validated_before_model_loading(tmp_path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()

    with pytest.raises(ValueError, match="64 到 2048"):
        Qwen3VLEmbeddingProvider(model_dir, dimension=4096)


def test_auto_device_prefers_mps() -> None:
    class Flag:
        @staticmethod
        def is_available():
            return True

    class Torch:
        class backends:
            mps = Flag()

        cuda = Flag()

    assert Qwen3VLEmbeddingProvider._select_device(Torch, "auto") == "mps"
