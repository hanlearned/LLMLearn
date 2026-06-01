"""
Capstone FastAPI 服务测试：健康检查 + 前端托管 + 异常优雅兜底。

用 `with TestClient(app)` 触发 startup 事件（真正构建多 Agent 图）。
没有真实 API Key，所以 /api/chat 内部的 LLM 调用会失败——但服务**必须**把它兜底成
友好话术并返回 200，而不是 500 崩给前端。这条正是「健壮性」的回归测试。
"""

import sys

from conftest import REPO_ROOT
from fastapi.testclient import TestClient

sys.path.insert(0, str(REPO_ROOT / "capstone" / "backend"))
import api as cap_api  # noqa: E402


def test_health():
    with TestClient(cap_api.app) as c:
        assert c.get("/api/health").json() == {"status": "ok"}


def test_frontend_served():
    with TestClient(cap_api.app) as c:
        r = c.get("/")
        assert r.status_code == 200
        assert "text/html" in r.headers["content-type"]


def test_chat_graceful_fallback_on_llm_error():
    """假 Key 下 LLM 调用必失败 → 应返回 200 + 兜底话术，不得 500。"""
    with TestClient(cap_api.app) as c:
        r = c.post("/api/chat", json={"session_id": "test-x", "message": "退货政策？"})
        assert r.status_code == 200
        body = r.json()
        assert "answer" in body and body["answer"]
        assert body["needs_approval"] is False
