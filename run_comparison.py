# -*- coding: utf-8 -*-
"""同50只股票的基础版vs优化版对比回测"""
import warnings
warnings.filterwarnings("ignore")
import sys
sys.path.insert(0, ".")

import config as C
from run_backtest import get_hs300_constituents, fetch_universe_data
from data_fetcher import get_index_daily
from backtest_engine import BacktestEngine, print_metrics
from optimized_strategy import run_optimized_backtest

print("=" * 70)
print("  基础版 vs 整合优化版 对比回测（同50只股票）")
print("=" * 70)

# 获取数据
print("\n获取数据...")
stocks = get_hs300_constituents()
stock_codes = stocks["code"].tolist()[:50]
index_df = get_index_daily(C.INDEX_CODE, C.DATA_START_DATE, C.DATA_END_DATE)
stock_data = fetch_universe_data(stock_codes, C.DATA_START_DATE, C.DATA_END_DATE)
print(f"有效股票: {len(stock_data)} 只")

# === 基础版 ===
print("\n" + "=" * 70)
print("  【基础版】回测中...")
print("=" * 70)
base_engine = BacktestEngine(initial_capital=C.INITIAL_CAPITAL)
base_metrics = base_engine.run(stock_data, index_df)
print_metrics(base_metrics)

# 保存基础版交易
import pandas as pd
base_trades = pd.DataFrame(base_engine.trades)
base_trades.to_csv("results/trades_baseline_50.csv", index=False)
base_equity = pd.DataFrame(base_engine.equity_curve)
base_equity.to_csv("results/equity_baseline_50.csv", index=False)

# === 优化版 ===
print("\n" + "=" * 70)
print("  【整合优化版】回测中（市场状态+ML评分，ML阈值55）...")
print("=" * 70)
opt_metrics = run_optimized_backtest(
    stock_data, index_df,
    use_regime=True, use_ml=True, use_minute=False,
    ml_threshold=55
)
if opt_metrics:
    print_metrics(opt_metrics)

# === 对比 ===
print("\n" + "=" * 70)
print("  对比汇总")
print("=" * 70)
print(f"{'指标':<15} {'基础版':>12} {'优化版':>12} {'变化':>12}")
print("-" * 55)
comparisons = [
    ("总交易数", "total_trades", ""),
    ("胜率(%)", "win_rate", "%"),
    ("平均盈利(%)", "avg_win_pct", "%"),
    ("平均亏损(%)", "avg_loss_pct", "%"),
    ("盈亏比", "profit_factor", ""),
    ("总收益率(%)", "total_return", "%"),
    ("年化收益率(%)", "annual_return", "%"),
    ("夏普比率", "sharpe", ""),
    ("最大回撤(%)", "max_drawdown", "%"),
    ("平均持仓(天)", "avg_hold_days", ""),
]
for label, key, unit in comparisons:
    base_val = base_metrics.get(key, 0)
    opt_val = opt_metrics.get(key, 0) if opt_metrics else 0
    if isinstance(base_val, float):
        base_str = f"{base_val:.2f}{unit}"
        opt_str = f"{opt_val:.2f}{unit}"
        diff = opt_val - base_val
        diff_str = f"{diff:+.2f}{unit}"
    else:
        base_str = str(base_val)
        opt_str = str(opt_val)
        diff_str = f"{opt_val - base_val:+d}"
    print(f"{label:<15} {base_str:>12} {opt_str:>12} {diff_str:>12}")

print("\n对比完成！")
