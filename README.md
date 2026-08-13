# A股K线形态突破 + 趋势跟踪策略系统

基于K线形态识别（W底/头肩底/箱体/收敛三角等）+ 量能确认 + ML评分 + 市场状态过滤的量化交易策略系统。

## 核心特性

- **6种K线形态识别**：W底、头肩底、矩形底、箱体、收敛三角、菱形
- **多因子入场过滤**：形态质量分 + 量能突破 + 假突破过滤 + 市场状态 + XGBoost ML评分
- **动态趋势跟踪出场**：移动止盈 + 硬止损(2ATR) + 趋势线跌破 + 亏损止损
- **实时监控系统**：662只股票全市场扫描，飞书Webhook信号推送，支持GitHub Actions云端运行
- **交互式可视化**：回测结果网站，K线买卖点标注，多维度统计

## 回测结果（6年，2020.08 - 2026.08，662只股票）

| 指标 | 数值 |
|------|------|
| 交易笔数 | 498笔 |
| **胜率** | **71.69%** |
| 平均盈利 | +17.77% |
| 平均亏损 | -4.67% |
| **盈亏比** | **8.87** |
| 总收益率 | 8877.74%（约88倍） |
| 年化收益率 | 117.22% |
| 夏普比率 | 3.79 |
| 最大回撤 | -11.46% |
| 平均持仓 | 15.0天 |

### 各形态表现

| 形态 | 笔数 | 胜率 | 平均收益 |
|------|------|------|----------|
| 头肩底 | 84 | 73.8% | +8.54% |
| W底 | 258 | 72.5% | +12.11% |
| 箱体 | 123 | 69.1% | +12.05% |
| 收敛三角 | 32 | 68.8% | +11.11% |

## 项目结构

```
├── config.py                  # 全局配置参数
├── data_fetcher.py            # 多源容错数据获取（东方财富+新浪）
├── pattern_recognizer.py      # K线形态识别引擎（6种形态）
├── strategy.py                # 策略核心逻辑（阻力位+突破+过滤）
├── market_regime.py           # 市场状态识别（GMM三状态：牛/熊/震荡）
├── ml_pattern_scorer.py       # ML形态评分器
├── train_ml_from_trades.py    # 从交易记录训练XGBoost模型
├── minute_confirmer.py        # 60分钟级别二次确认
├── enhanced_strategy.py       # 高胜率增强模块（分批止盈）
├── optimized_strategy.py      # 整合优化版回测引擎
├── backtest_engine.py         # 基础回测引擎
├── robustness_test.py         # 鲁棒性测试
├── visualization.py           # 可视化模块（6张图）
├── report_generator.py        # 策略报告生成器
│
├── realtime_monitor.py        # 实时监控主脚本（662只扫描）
├── feishu_notifier.py         # 飞书Webhook通知模块
├── run_monitor.bat            # Windows一键运行
├── test_feishu.py             # 飞书推送测试
│
├── run_backtest.py            # 基础版回测
├── run_comparison.py          # 基础vs优化对比回测
├── run_expanded_backtest.py   # 扩展股票池3年回测
├── run_6year_backtest.py      # 6年全市场回测
├── save_optimized_results.py  # 保存优化版结果
│
├── fetch_expanded_universe.py # 获取扩展股票池（663只）
├── download_expanded_data.py  # 下载3年数据
├── download_6year_data.py     # 下载6年数据
│
├── export_web_data.py         # 导出网站数据
├── export_expanded_web_data.py
├── export_6year_web_data.py
├── export_stock_kline.py      # 导出个股K线数据
│
├── deploy_to_github.py        # GitHub一键部署脚本
├── requirements.txt           # Python依赖
└── web/                       # 交互式回测网站
    ├── index.html
    └── *.js                   # 回测数据
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 运行回测

```bash
# 基础版回测
python run_backtest.py

# 6年全市场优化版回测（约40-50分钟）
python run_6year_backtest.py
```

### 3. 实时监控

```bash
# 立即扫描一次
python realtime_monitor.py --force

# 查看当前持仓
python realtime_monitor.py --list-pos

# 移除持仓（未实际买入时）
python realtime_monitor.py --remove-pos 股票代码
```

配置飞书Webhook：
```bash
# Windows
set FEISHU_WEBHOOK=https://open.feishu.cn/open-apis/bot/v2/hook/你的hook

# Linux/Mac
export FEISHU_WEBHOOK=https://open.feishu.cn/open-apis/bot/v2/hook/你的hook
```

### 4. 云端部署（GitHub Actions）

1. Fork本仓库
2. Settings → Secrets → Actions → 添加 `FEISHU_WEBHOOK`
3. Actions页面启用workflow，交易时间自动运行

## 策略逻辑

### 入场条件（全部满足）

1. **形态识别**：detect_all_patterns识别6种形态，取质量分最高的
2. **质量分阈值**：形态质量分 ≥ 50
3. **价格突破**：收盘价 > 颈线位
4. **量能确认**：当日成交量 > 前3天平均值 × 1.0
5. **假突破过滤**：check_false_breakout验证
6. **市场状态过滤**：check_market_filter（GMM三状态自适应参数）
7. **ML评分**：XGBoost评分 ≥ 55

### 出场条件（任一触发）

1. **硬止损**：收盘价 < 入场价 - 2×ATR14
2. **移动止盈**：盈利>10%后，从最高点回撤>5%
3. **亏损止损**：收盘价 < 入场价 × 0.95
4. **趋势线跌破**：收盘价 < 动态趋势线 × 0.99

## 数据说明

- 数据源：东方财富（akshare）+ 新浪财经（实时行情）
- 股票池：663只（沪深300+中证500+部分活跃股）
- 历史数据：parquet缓存，6年约几百MB
- 实时行情：新浪API批量获取，一次请求700只

## 注意事项

- 本策略仅供学习研究，不构成投资建议
- 历史回测收益不代表未来表现
- 实盘交易需考虑滑点、手续费、流动性等因素
- 策略条件严格，信号频率较低（平均每周1-2笔）

## 技术栈

- Python 3.10+
- pandas / numpy / scipy（数据处理）
- akshare（行情数据）
- xgboost / lightgbm / scikit-learn（ML模型）
- matplotlib（可视化）
- Chart.js（前端K线图）
