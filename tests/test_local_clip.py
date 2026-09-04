import pytest

from app.adapters.local_clip import LocalClipEmbeddingProvider
from app.errors import ModelUnavailableError


def test_missing_local_model_has_clear_error_and_no_api_fallback(tmp_path) -> None:
    with pytest.raises(ModelUnavailableError) as caught:
        LocalClipEmbeddingProvider(tmp_path / "missing-image", tmp_path / "missing-text")

    message = str(caught.value)
    assert "本地 CLIP 模型不存在" in message
    assert "download_models" in message
    assert "不会自动切换到外部模型 API" in message

