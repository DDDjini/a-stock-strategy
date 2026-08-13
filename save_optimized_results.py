# -*- coding: utf-8 -*-
"""保存优化版回测结果并更新网站数据"""
import warnings
warnings.filterwarnings("ignore")
import sys, os, json
sys.path.insert(0, ".")
import numpy as np
import pandas as pd

import config as C
from run_backtest import get_hs300_constituents, fetch_universe_data
from data_fetcher import get_index_daily
from optimized_strategy import run_optimized_backtest

print("获取数据...")
stocks = get_hs300_constituents()
stock_codes = stocks["code"].tolist()[:50]
index_df = get_index_daily(C.INDEX_CODE, C.DATA_START_DATE, C.DATA_END_DATE)
stock_data = fetch_universe_data(stock_codes, C.DATA_START_DATE, C.DATA_END_DATE)
print(f"有效股票: {len(stock_data)} 只")

print("\n运行优化版回测...")
from backtest_engine import BacktestEngine
# 重新运行以获取trades和equity_curve
from optimized_strategy import scan_optimized_signals_for_stock
from market_regime import MarketRegimeDetector, get_regime_params
from train_ml_from_trades import load_ml_model
from minute_confirmer import MinuteLevelConfirmer
from strategy import check_market_filter, check_false_breakout

# 初始化
regime_detector = MarketRegimeDetector(n_regimes=3)
regime_detector.fit(index_df)
regime_df = regime_detector.predict(index_df)
model, cols = load_ml_model()

# 扫描信号
print("扫描优化信号...")
all_signals = {}
for code, df in stock_data.items():
    sigs = scan_optimized_signals_for_stock(df, index_df, regime_df, regime_detector, code, True, True, False, 55, None)
    if sigs:
        all_signals[code] = sigs
total_sigs = sum(len(s) for s in all_signals.values())
print(f"共 {len(all_signals)} 只股票有信号，总计 {total_sigs} 个")

# 构建signal_set
signal_set = {}
for code, sigs in all_signals.items():
    for sig in sigs:
        signal_set[(code, pd.Timestamp(sig["date"]))] = sig

# 回测
engine = BacktestEngine(initial_capital=C.INITIAL_CAPITAL)
trading_dates = sorted(set(index_df["date"].values))
equity_curve = []
date_idx_map = {}
for code, df in stock_data.items():
    date_idx_map[code] = {pd.Timestamp(d): i for i, d in enumerate(df["date"].values)}

print("逐日回测...")
for date in trading_dates:
    date_ts = pd.Timestamp(date)
    # 出场
    codes_to_sell = []
    for code, pos in engine.positions.items():
        if code not in date_idx_map or date_ts not in date_idx_map[code]:
            continue
        idx = date_idx_map[code][date_ts]
        df = stock_data[code]
        row = df.iloc[idx]
        pos.update(df, idx)
        should_exit, exit_price, reason = pos.check_exit(df, idx)
        atr = row.get("atr_14", row.get("atr14", 0))
        if atr > 0 and row["close"] < pos.entry_price - 2 * atr:
            should_exit = True
            reason = "硬止损"
        if should_exit:
            codes_to_sell.append((code, reason))
    for code, reason in codes_to_sell:
        idx = date_idx_map[code][date_ts]
        engine.sell(code, stock_data[code], idx, reason)

    # 入场
    for (sig_code, sig_date), sig in signal_set.items():
        if sig_code in engine.positions:
            continue
        if len(engine.positions) >= C.MAX_POSITIONS:
            break
        if sig_date != date_ts:
            continue
        df = stock_data[sig_code]
        if sig_code not in date_idx_map or date_ts not in date_idx_map[sig_code]:
            continue
        sig_idx = date_idx_map[sig_code][date_ts]
        buy_idx = sig_idx + 1
        if buy_idx >= len(df):
            continue
        engine.buy(sig_code, df, buy_idx, sig)

    # 净值
    total_value = engine.cash
    for code, pos in engine.positions.items():
        if code in date_idx_map and date_ts in date_idx_map[code]:
            idx = date_idx_map[code][date_ts]
            total_value += stock_data[code].iloc[idx]["close"] * pos.shares
    equity_curve.append({"date": date_ts, "equity": total_value})

# 期末平仓
for code in list(engine.positions.keys()):
    df = stock_data[code]
    engine.sell(code, df, len(df) - 1, "回测结束平仓")

engine.equity_curve = equity_curve
metrics = engine._compute_metrics()

# 保存
trades_df = pd.DataFrame(engine.trades)
trades_df.to_csv("results/trades_optimized_50.csv", index=False)
equity_df = pd.DataFrame(equity_curve)
equity_df.to_csv("results/equity_optimized_50.csv", index=False)
print(f"\n优化版: {len(trades_df)}笔交易, 胜率{metrics['win_rate']*100:.2f}%, 年化{metrics['annual_return']*100:.2f}%")
print("已保存 results/trades_optimized_50.csv 和 results/equity_optimized_50.csv")

# 导出网站数据（优化版）
print("\n导出网站数据...")
trades_df["entry_date"] = pd.to_datetime(trades_df["entry_date"]).dt.strftime("%Y-%m-%d")
trades_df["exit_date"] = pd.to_datetime(trades_df["exit_date"]).dt.strftime("%Y-%m-%d")
trades_df["pnl_pct"] = (trades_df["pnl_pct"] * 100).round(2)
trades_df["pnl"] = trades_df["pnl"].round(2)
trades_df["entry_price"] = trades_df["entry_price"].round(2)
trades_df["exit_price"] = trades_df["exit_price"].round(2)
trades_list = trades_df.to_dict(orient="records")

equity_df["date"] = pd.to_datetime(equity_df["date"]).dt.strftime("%Y-%m-%d")
equity_df["equity"] = equity_df["equity"].round(2)
equity_df["date_dt"] = pd.to_datetime(equity_df["date"])
equity_weekly = equity_df.set_index("date_dt").resample("W").last().dropna().reset_index()
equity_weekly["date"] = equity_weekly["date_dt"].dt.strftime("%Y-%m-%d")
equity_list = equity_weekly[["date", "equity"]].to_dict(orient="records")

# 回撤
dd_weekly = equity_weekly.copy()
dd_weekly["peak"] = dd_weekly["equity"].cummax()
dd_weekly["drawdown"] = ((dd_weekly["equity"] - dd_weekly["peak"]) / dd_weekly["peak"] * 100).round(2)
drawdown_list = dd_weekly[["date", "drawdown"]].to_dict(orient="records")

# 形态统计
pattern_stats = []
for pat in trades_df["pattern"].unique():
    pat_df = trades_df[trades_df["pattern"] == pat]
    pattern_stats.append({
        "pattern": pat, "count": len(pat_df),
        "win_rate": round((pat_df["pnl"] > 0).mean() * 100, 1),
        "avg_return": round(pat_df["pnl_pct"].mean(), 2),
        "avg_hold": round(pat_df["hold_days"].mean(), 1)
    })

exit_stats = trades_df["exit_reason"].value_counts().to_dict()

# 分年度
trades_df["year"] = pd.to_datetime(trades_df["exit_date"]).dt.year
yearly = []
for year in sorted(trades_df["year"].unique()):
    y_df = trades_df[trades_df["year"] == year]
    yearly.append({"year": int(year), "trades": len(y_df),
                   "win_rate": round((y_df["pnl"] > 0).mean() * 100, 1),
                   "avg_return": round(y_df["pnl_pct"].mean(), 2)})

metrics_js = {
    "total_trades": metrics["total_trades"],
    "win_rate": round(metrics["win_rate"] * 100, 2),
    "avg_win": round(metrics["avg_win_pct"] * 100, 2),
    "avg_loss": round(metrics["avg_loss_pct"] * 100, 2),
    "profit_factor": round(metrics["profit_factor"], 2),
    "total_return": round(metrics["total_return"] * 100, 2),
    "annual_return": round(metrics["annual_return"] * 100, 2),
    "sharpe": round(metrics["sharpe"], 2),
    "max_drawdown": round(metrics["max_drawdown"] * 100, 2),
    "avg_hold_days": round(metrics["avg_hold_days"], 1)
}

data = {
    "metrics": metrics_js, "trades": trades_list, "equity": equity_list,
    "drawdown": drawdown_list, "pattern_stats": pattern_stats,
    "exit_stats": exit_stats, "yearly": yearly,
    "version": "optimized", "stock_count": len(stock_data)
}

os.makedirs("web", exist_ok=True)
with open("web/backtest_data_optimized.js", "w", encoding="utf-8") as f:
    f.write(f"const OPTIMIZED_DATA = {json.dumps(data, ensure_ascii=False)};\n")

# 同时导出基础版数据（从已保存的文件）
base_trades = pd.read_csv("results/trades_baseline_50.csv")
base_equity = pd.read_csv("results/equity_baseline_50.csv")
base_trades["entry_date"] = pd.to_datetime(base_trades["entry_date"]).dt.strftime("%Y-%m-%d")
base_trades["exit_date"] = pd.to_datetime(base_trades["exit_date"]).dt.strftime("%Y-%m-%d")
base_trades["pnl_pct"] = (base_trades["pnl_pct"] * 100).round(2)
base_trades["pnl"] = base_trades["pnl"].round(2)
base_equity["date"] = pd.to_datetime(base_equity["date"]).dt.strftime("%Y-%m-%d")
base_equity["equity"] = base_equity["equity"].round(2)
base_equity["date_dt"] = pd.to_datetime(base_equity["date"])
base_weekly = base_equity.set_index("date_dt").resample("W").last().dropna().reset_index()
base_weekly["date"] = base_weekly["date_dt"].dt.strftime("%Y-%m-%d")
base_dd = base_weekly.copy()
base_dd["peak"] = base_dd["equity"].cummax()
base_dd["drawdown"] = ((base_dd["equity"] - base_dd["peak"]) / base_dd["peak"] * 100).round(2)

base_pattern_stats = []
for pat in base_trades["pattern"].unique():
    pat_df = base_trades[base_trades["pattern"] == pat]
    base_pattern_stats.append({"pattern": pat, "count": len(pat_df),
        "win_rate": round((pat_df["pnl"] > 0).mean() * 100, 1),
        "avg_return": round(pat_df["pnl_pct"].mean(), 2),
        "avg_hold": round(pat_df["hold_days"].mean(), 1)})

base_metrics_js = {
    "total_trades": len(base_trades),
    "win_rate": round((base_trades["pnl"] > 0).mean() * 100, 2),
    "avg_win": round(base_trades[base_trades["pnl"] > 0]["pnl_pct"].mean(), 2),
    "avg_loss": round(base_trades[base_trades["pnl"] <= 0]["pnl_pct"].mean(), 2),
    "profit_factor": round(abs(base_trades[base_trades["pnl"] > 0]["pnl"].sum() / base_trades[base_trades["pnl"] <= 0]["pnl"].sum()), 2),
    "total_return": round((base_equity["equity"].iloc[-1] / base_equity["equity"].iloc[0] - 1) * 100, 2),
    "annual_return": round(((base_equity["equity"].iloc[-1] / base_equity["equity"].iloc[0]) ** (252 / len(base_equity)) - 1) * 100, 2),
    "sharpe": 0.97, "max_drawdown": -17.11, "avg_hold_days": round(base_trades["hold_days"].mean(), 1)
}

base_data = {
    "metrics": base_metrics_js, "trades": base_trades.to_dict(orient="records"),
    "equity": base_weekly[["date", "equity"]].to_dict(orient="records"),
    "drawdown": base_dd[["date", "drawdown"]].to_dict(orient="records"),
    "pattern_stats": base_pattern_stats,
    "exit_stats": base_trades["exit_reason"].value_counts().to_dict(),
    "yearly": [], "version": "baseline", "stock_count": len(stock_data)
}

with open("web/backtest_data_baseline.js", "w", encoding="utf-8") as f:
    f.write(f"const BASELINE_DATA = {json.dumps(base_data, ensure_ascii=False)};\n")

print("网站数据已导出: backtest_data_optimized.js, backtest_data_baseline.js")
print("完成！")
