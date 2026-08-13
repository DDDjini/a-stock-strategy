@echo off
chcp 65001 >nul
cd /d "D:\a股突破形态做趋势"

REM ========== 配置区域 ==========
REM 替换为你的飞书Webhook地址
set FEISHU_WEBHOOK=https://open.feishu.cn/open-apis/bot/v2/hook/39c3de44-639c-4f8d-8584-53e3304519c2

REM ========== 运行监控 ==========
python realtime_monitor.py --force

REM 暂停查看结果（可选）
REM pause
