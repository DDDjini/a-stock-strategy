# -*- coding: utf-8 -*-
"""飞书Webhook通知模块"""
import requests
import json
import time

class FeishuNotifier:
    def __init__(self, webhook_url):
        self.webhook_url = webhook_url
        self.last_send_time = 0
        self.min_interval = 1  # 最小发送间隔(秒)

    def _send(self, payload):
        """发送消息，带频率控制"""
        now = time.time()
        if now - self.last_send_time < self.min_interval:
            time.sleep(self.min_interval - (now - self.last_send_time))
        try:
            resp = requests.post(
                self.webhook_url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=10
            )
            self.last_send_time = time.time()
            return resp.status_code == 200
        except Exception as e:
            print(f"[飞书推送失败] {e}")
            return False

    def send_entry_signal(self, code, name, pattern, price, volume_ratio,
                          ml_score, regime, neckline, quality_score, signal_time):
        """入场信号推送（交互式卡片）"""
        color_map = {
            "W底": "green", "头肩底": "green", "矩形底": "green",
            "箱体": "blue", "收敛三角": "blue", "菱形": "orange"
        }
        color = color_map.get(pattern, "blue")

        card = {
            "msg_type": "interactive",
            "card": {
                "header": {
                    "title": {
                        "tag": "plain_text",
                        "content": f"🔴 入场信号 | {name}({code})"
                    },
                    "template": color
                },
                "elements": [
                    {
                        "tag": "div",
                        "text": {
                            "tag": "lark_md",
                            "content": (
                                f"**形态**: {pattern}\n"
                                f"**突破价**: {price:.2f}\n"
                                f"**颈线位**: {neckline:.2f}\n"
                                f"**量比**: {volume_ratio:.2f}x\n"
                                f"**ML评分**: {ml_score:.0f}/100\n"
                                f"**形态质量**: {quality_score:.0f}/100\n"
                                f"**市场状态**: {regime}\n"
                                f"**信号时间**: {signal_time}"
                            )
                        }
                    },
                    {
                        "tag": "note",
                        "elements": [
                            {
                                "tag": "plain_text",
                                "content": "策略: K线形态突破+趋势跟踪 | 次日开盘买入"
                            }
                        ]
                    }
                ]
            }
        }
        return self._send(card)

    def send_exit_signal(self, code, name, exit_price, entry_price, hold_days,
                         pnl_pct, reason, exit_time):
        """出场信号推送"""
        is_win = pnl_pct >= 0
        color = "green" if is_win else "red"
        emoji = "🟢" if is_win else "🔴"
        pnl_str = f"+{pnl_pct:.2f}%" if is_win else f"{pnl_pct:.2f}%"

        card = {
            "msg_type": "interactive",
            "card": {
                "header": {
                    "title": {
                        "tag": "plain_text",
                        "content": f"{emoji} 出场信号 | {name}({code}) {pnl_str}"
                    },
                    "template": color
                },
                "elements": [
                    {
                        "tag": "div",
                        "text": {
                            "tag": "lark_md",
                            "content": (
                                f"**出场原因**: {reason}\n"
                                f"**入场价**: {entry_price:.2f}\n"
                                f"**出场价**: {exit_price:.2f}\n"
                                f"**持仓天数**: {hold_days}天\n"
                                f"**收益率**: {pnl_str}\n"
                                f"**出场时间**: {exit_time}"
                            )
                        }
                    }
                ]
            }
        }
        return self._send(card)

    def send_daily_summary(self, date, entry_count, exit_count, win_count,
                           total_pnl, monitored_positions):
        """每日收盘总结"""
        card = {
            "msg_type": "interactive",
            "card": {
                "header": {
                    "title": {
                        "tag": "plain_text",
                        "content": f"📊 每日监控总结 | {date}"
                    },
                    "template": "purple"
                },
                "elements": [
                    {
                        "tag": "div",
                        "text": {
                            "tag": "lark_md",
                            "content": (
                                f"**今日入场信号**: {entry_count}个\n"
                                f"**今日出场信号**: {exit_count}个\n"
                                f"**出场盈利**: {win_count}/{exit_count}\n"
                                f"**今日出场累计收益**: {total_pnl:+.2f}%\n"
                                f"**当前监控持仓**: {monitored_positions}只"
                            )
                        }
                    }
                ]
            }
        }
        return self._send(card)

    def send_text(self, text):
        """简单文本消息"""
        payload = {
            "msg_type": "text",
            "content": {"text": text}
        }
        return self._send(payload)

    def send_test(self):
        """测试连接"""
        return self.send_text("✅ 实时监控系统已启动，飞书推送连接正常")
