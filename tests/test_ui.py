from app.fixtures import build_demo_service
from app.runtime import StaticRuntime
from app.ui import search_gallery


def test_gallery_contains_title_artist_and_similarity() -> None:
    gallery, message = search_gallery(StaticRuntime(build_demo_service()), "红色背景，有鞋")

    assert gallery
    assert "示例艺人" in gallery[0][1]
    assert "相似度" in gallery[0][1]
    assert "图文向量" in message
    assert "不是概率" in message


def test_gallery_rejects_blank_query() -> None:
    gallery, message = search_gallery(StaticRuntime(build_demo_service()), "   ")

    assert gallery == []
    assert message == "请输入封面描述。"
