"""Capstone 逻辑测试：业务工具 + 路由 + 退款 HITL 决策 + 图编译。无需 API Key。"""

import sys

from conftest import REPO_ROOT

# 把 capstone/backend 加入路径，便于 import 其 config/kb/tools/graph
sys.path.insert(0, str(REPO_ROOT / "capstone" / "backend"))

import graph as cap_graph  # noqa: E402
import tools as cap_tools  # noqa: E402
from langgraph.checkpoint.memory import MemorySaver  # noqa: E402
from langgraph.graph import END  # noqa: E402


# ---------- 业务工具 ----------
def test_get_order_found_and_missing():
    assert cap_tools.get_order("10001")["item"] == "蓝牙耳机"
    assert cap_tools.get_order("99999") is None


def test_order_status_text():
    assert "蓝牙耳机" in cap_tools.order_status_text("10001")
    assert "未找到" in cap_tools.order_status_text("00000")


# ---------- 路由 ----------
def test_route_from_triage():
    assert cap_graph.route_from_triage({"route": "policy"}) == "policy"
    assert cap_graph.route_from_triage({"route": "refund"}) == "refund"


def test_after_prepare_refund():
    # 没拿到有效订单号 → 直接结束让用户补充
    assert cap_graph.after_prepare_refund({"route": "refund_incomplete"}) == END
    # 正常 → 进入需审批的执行节点
    assert cap_graph.after_prepare_refund({"route": "refund"}) == "execute_refund"


# ---------- 退款 HITL 执行节点（纯逻辑，不调用 LLM）----------
def test_execute_refund_approved():
    out = cap_graph.execute_refund_node({"refund": {"order_id": "10001", "amount": 299}, "approved": True})
    assert "已为订单 10001 退款" in out["answer"]


def test_execute_refund_rejected():
    out = cap_graph.execute_refund_node({"refund": {"order_id": "10001", "amount": 299}, "approved": False})
    assert "未通过" in out["answer"] and "取消" in out["answer"]


# ---------- 图编译 ----------
def test_graph_compiles_with_hitl_node():
    app = cap_graph.build_graph(MemorySaver())
    nodes = set(app.get_graph().nodes.keys())
    # 关键节点齐全，且退款执行节点存在（HITL 会在它之前中断）
    for n in ["triage", "policy", "order", "prepare_refund", "execute_refund", "other"]:
        assert n in nodes
