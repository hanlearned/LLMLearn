"""业务数据与动作：订单查询、退款金额核算。

真实项目里这些会去调订单系统/支付系统的 API；这里用内存数据模拟，
对上层 graph 的接口不变——这正是「把业务能力工具化」的好处：换实现不换接口。
"""

# 模拟订单库
_ORDERS = {
    "10001": {"item": "蓝牙耳机", "amount": 299, "status": "已签收", "logistics": "顺丰 SF123456，已签收"},
    "10002": {"item": "定制马克杯", "amount": 89, "status": "待发货", "logistics": "暂无"},
    "10003": {"item": "机械键盘", "amount": 459, "status": "运输中", "logistics": "中通 ZT987654，预计后天送达"},
}


def get_order(order_id: str) -> dict | None:
    return _ORDERS.get(order_id.strip())


def order_status_text(order_id: str) -> str:
    o = get_order(order_id)
    if not o:
        return f"未找到订单 {order_id}，请核对订单号。"
    return f"订单{order_id}（{o['item']}，¥{o['amount']}）：状态={o['status']}，物流={o['logistics']}"
