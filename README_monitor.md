# A股形态突破策略 - 实时信号监控系统

## 功能概述

实时监控662只股票（沪深300+深证成指+科创50+创业板指），当策略触发入场/出场信号时，通过飞书Webhook推送通知。

**入场信号条件**：
- K线形态识别（W底、头肩底、箱体、收敛三角等6种）
- 收盘价突破颈线位
- 当日成交量 > 前3天平均值
- ML形态评分 ≥ 55分
- 市场状态过滤（熊市收紧参数）
- 假突破过滤

**出场信号条件**：
- 硬止损：收盘价 < 入场价 - 2×ATR
- 移动止盈：盈利>10%后从高点回撤5%
- 亏损止损：跌破入场价5%
- 趋势线跌破

## 快速开始

### 1. 配置飞书Webhook

**方式A：环境变量（推荐）**
```bash
# Windows CMD
set FEISHU_WEBHOOK=https://open.feishu.cn/open-apis/bot/v2/hook/你的hook地址

# Windows PowerShell
$env:FEISHU_WEBHOOK="https://open.feishu.cn/open-apis/bot/v2/hook/你的hook地址"
```

**方式B：修改run_monitor.bat**
编辑 `run_monitor.bat`，将 `set FEISHU_WEBHOOK=` 后面替换为你的地址。

### 2. 测试运行

```bash
# 强制运行一次（非交易时间也执行）
python realtime_monitor.py --force

# 发送飞书测试消息
python realtime_monitor.py --force --test
```

### 3. 查看当前持仓

```bash
python realtime_monitor.py --list-pos
```

### 4. 手动添加持仓

实际买入后，记录持仓以便后续监控出场信号：
```bash
python realtime_monitor.py --add-pos 600000 2026-08-13 10.50
# 参数：股票代码 入场日期 入场价格
```

## 部署方式

### 方式一：Windows任务计划程序（推荐，实时性最好）

1. 按 `Win+R`，输入 `taskschd.msc` 打开任务计划程序
2. 创建基本任务 → 名称：A股策略监控
3. 触发器：每天，重复任务间隔5分钟，持续时间8小时
4. 操作：启动程序 → 选择 `D:\a股突破形态做趋势\run_monitor.bat`
5. 起始于：`D:\a股突破形态做趋势`

**设置建议**：
- 开始时间：09:25（开盘前5分钟）
- 重复间隔：5分钟
- 持续时间：到15:05（收盘后）
- 仅在工作日触发

### 方式二：GitHub Actions（云端运行，无需开机）

1. 将项目推送到GitHub仓库
2. 仓库 Settings → Secrets and variables → Actions
3. 新建 Secret：`FEISHU_WEBHOOK`，值为你的飞书Webhook地址
4. 启用 `.github/workflows/monitor.yml`
5. 自动在交易时间每30分钟运行一次

**注意**：GitHub Actions免费版每月2000分钟，本策略每次运行约2分钟，每月约960分钟，在免费额度内。

### 方式三：Python定时循环

```bash
# 持续运行，每5分钟扫描一次
python -c "
import time, subprocess
while True:
    subprocess.run(['python', 'realtime_monitor.py'])
    time.sleep(300)
"
```

## 飞书消息示例

### 入场信号
```
🔴 入场信号 | 贵州茅台(600519)
形态: W底
突破价: 1680.50
颈线位: 1675.00
量比: 1.85x
ML评分: 78/100
形态质量: 72/100
市场状态: range
信号时间: 2026-08-13 10:35:00
```

### 出场信号
```
🟢 出场信号 | 贵州茅台(600519) +12.35%
出场原因: 移动止盈(高点回撤5%)
入场价: 1680.50
出场价: 1888.30
持仓天数: 8天
收益率: +12.35%
```

## 文件说明

| 文件 | 说明 |
|------|------|
| `realtime_monitor.py` | 主监控脚本 |
| `feishu_notifier.py` | 飞书推送模块 |
| `run_monitor.bat` | Windows一键运行脚本 |
| `monitor_state.json` | 状态持久化文件（自动生成） |
| `.github/workflows/monitor.yml` | GitHub Actions配置 |
| `requirements.txt` | Python依赖 |

## 状态文件说明

`monitor_state.json` 自动生成，包含：
- `notified_entry`：已通知的入场信号（避免重复推送）
- `notified_exit`：已通知的出场信号
- `positions`：当前持仓列表（入场价、入场日期、形态）
- `daily_stats`：每日统计

## 常见问题

**Q: 为什么扫描后没有信号？**
A: 策略条件较严格（形态+突破+量能+ML+市场状态），不是每天都有信号。历史回测6年498笔，平均每周约1.6笔。

**Q: 可以调整信号灵敏度吗？**
A: 修改 `realtime_monitor.py` 中的参数：
- `ML_THRESHOLD = 55` → 降低到50可增加信号
- `MIN_VOLUME_RATIO = 1.0` → 降低到0.8可放宽量能要求

**Q: 飞书收不到消息？**
A: 检查：①Webhook地址是否正确 ②机器人是否在群内 ③是否开启了IP白名单限制

**Q: 如何只监控特定股票？**
A: 修改 `data_cache/expanded_universe.csv`，只保留你想监控的股票。

## 策略回测参考（6年）

- 胜率：71.69%
- 年化收益：117.22%
- 盈亏比：8.87
- 最大回撤：-11.46%
- 夏普比率：3.79
