from fastapi.testclient import TestClient

from app.api import build_test_app, create_app
from app.errors import ModelUnavailableError
from app.fixtures import build_demo_service
from app.runtime import RuntimeStatus

client = TestClient(build_test_app(build_demo_service()))


def test_health() -> None:
    assert client.get("/health").json() == {
        "status": "ready",
        "message": "测试搜索服务已就绪",
        "index_count": 0,
        "index_reused": None,
    }


def test_search_api_returns_candidates_and_question() -> None:
    response = client.post("/search", json={"description": "红色背景，有一只小皮鞋"})

    assert response.status_code == 200
    body = response.json()
    assert len(body["candidates"]) == 20
    assert body["question"]["answers"] == ["yes", "no", "uncertain"]
    assert body["raw_query"] == "红色背景，有一只小皮鞋"


def test_search_api_rejects_blank_description() -> None:
    response = client.post("/search", json={"description": "   "})

    assert response.status_code == 422


def test_search_api_rejects_unknown_question() -> None:
    response = client.post(
        "/search",
        json={
            "description": "红色背景，有鞋",
            "answers": [{"question_id": "missing", "value": "yes"}],
        },
    )

    assert response.status_code == 422
    assert "未知问题" in response.json()["detail"]


def test_search_api_reports_missing_local_model_without_fallback() -> None:
    class MissingModelRuntime:
        status = RuntimeStatus(
            "error",
            "本地 CLIP 模型不存在；Seek 不会自动切换到外部模型 API。",
        )

        def initialize(self):
            return self.status

        def get_service(self):
            raise ModelUnavailableError(self.status.message)

    missing_client = TestClient(create_app(MissingModelRuntime(), mount_ui=False))

    response = missing_client.post("/search", json={"description": "shoes"})

    assert response.status_code == 503
    assert "本地 CLIP 模型不存在" in response.json()["detail"]
    assert "不会自动切换到外部模型 API" in response.json()["detail"]
