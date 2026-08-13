# -*- coding: utf-8 -*-
"""飞书Webhook测试脚本"""
import sys
import os
sys.path.insert(0, ".")
from feishu_notifier import FeishuNotifier

def main():
    webhook = os.environ.get("FEISHU_WEBHOOK", "")
    if not webhook:
        print("请先设置飞书Webhook地址：")
        print("  set FEISHU_WEBHOOK=https://open.feishu.cn/open-apis/bot/v2/hook/你的地址")
        webhook = input("或直接输入Webhook地址: ").strip()
        if not webhook:
            return

    notifier = FeishuNotifier(webhook)

    print("正在发送测试消息...")

    # 测试文本消息
    ok1 = notifier.send_text("✅ 实时监控系统测试 - 文本消息发送成功")
    print(f"文本消息: {'成功' if ok1 else '失败'}")

    # 测试入场信号卡片
    ok2 = notifier.send_entry_signal(
        code="600519", name="贵州茅台", pattern="W底",
        price=1680.50, volume_ratio=1.85, ml_score=78,
        regime="range", neckline=1675.00, quality_score=72,
        signal_time="2026-08-13 10:35:00"
    )
    print(f"入场信号卡片: {'成功' if ok2 else '失败'}")

    # 测试出场信号卡片
    ok3 = notifier.send_exit_signal(
        code="600519", name="贵州茅台", exit_price=1888.30,
        entry_price=1680.50, hold_days=8, pnl_pct=12.35,
        reason="移动止盈(高点回撤5%)", exit_time="2026-08-13 14:20:00"
    )
    print(f"出场信号卡片: {'成功' if ok3 else '失败'}")

    if ok1 and ok2 and ok3:
        print("\n✅ 全部测试通过！飞书推送配置正确。")
    else:
        print("\n❌ 部分测试失败，请检查Webhook地址和网络连接。")

if __name__ == "__main__":
    main()
