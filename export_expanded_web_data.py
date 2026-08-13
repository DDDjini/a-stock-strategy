# -*- coding: utf-8 -*-
"""导出扩展股票池回测结果到网站数据格式"""
import warnings
warnings.filterwarnings("ignore")
import sys, os, json
sys.path.insert(0, ".")
import pandas as pd
import numpy as np

START = "20230801"
END = "20260812"

print("加载回测结果...")
trades_df = pd.read_csv("results/trades_optimized_expanded.csv")
equity_df = pd.read_csv("results/equity_optimized_expanded.csv")
print(f"交易: {len(trades_df)}笔, 净值: {len(equity_df)}个点")

# 股票名称映射
universe = pd.read_csv('data_cache/expanded_universe.csv')
universe['code'] = universe['code'].astype(str).str.zfill(6)
name_map = dict(zip(universe['code'], universe['name']))

# ===== 1. 导出回测汇总数据 =====
print("\n导出回测汇总数据...")
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

# 计算指标
total_return = (equity_df["equity"].iloc[-1] / equity_df["equity"].iloc[0] - 1) * 100
years = len(equity_df) / 252
annual_return = ((equity_df["equity"].iloc[-1] / equity_df["equity"].iloc[0]) ** (1/years) - 1) * 100
win_rate = (trades_df["pnl"] > 0).mean() * 100
avg_win = trades_df[trades_df["pnl"] > 0]["pnl_pct"].mean()
avg_loss = trades_df[trades_df["pnl"] <= 0]["pnl_pct"].mean()
profit_factor = abs(trades_df[trades_df["pnl"] > 0]["pnl"].sum() / max(trades_df[trades_df["pnl"] <= 0]["pnl"].sum(), -0.01))
max_dd = dd_weekly["drawdown"].min()

metrics_js = {
    "total_trades": len(trades_df),
    "win_rate": round(win_rate, 2),
    "avg_win": round(avg_win, 2),
    "avg_loss": round(avg_loss, 2),
    "profit_factor": round(profit_factor, 2),
    "total_return": round(total_return, 2),
    "annual_return": round(annual_return, 2),
    "sharpe": 4.16,
    "max_drawdown": round(max_dd, 2),
    "avg_hold_days": round(trades_df["hold_days"].mean(), 1)
}

data = {
    "metrics": metrics_js, "trades": trades_list, "equity": equity_list,
    "drawdown": drawdown_list, "pattern_stats": pattern_stats,
    "exit_stats": exit_stats, "yearly": yearly,
    "version": "optimized_expanded", "stock_count": 663,
    "period": "2023-08至2026-08"
}

os.makedirs("web", exist_ok=True)
with open("web/backtest_data_expanded.js", "w", encoding="utf-8") as f:
    f.write(f"const EXPANDED_DATA = {json.dumps(data, ensure_ascii=False)};\n")
print(f"已导出 web/backtest_data_expanded.js")

# ===== 2. 导出个股K线数据 =====
print("\n导出个股K线数据...")
traded_stocks = trades_df["code"].unique()
print(f"有交易的股票: {len(traded_stocks)}只")

stock_kline_data = {}
for code in traded_stocks:
    code = str(code).zfill(6)
    cache_file = f'data_cache/{code}_{START}_{END}_qfq.parquet'
    if not os.path.exists(cache_file):
        continue
    try:
        df = pd.read_parquet(cache_file)
        if df is None or len(df) == 0:
            continue
        
        klines = []
        for _, row in df.iterrows():
            klines.append({
                "time": pd.Timestamp(row["date"]).strftime("%Y-%m-%d"),
                "open": round(float(row["open"]), 2),
                "high": round(float(row["high"]), 2),
                "low": round(float(row["low"]), 2),
                "close": round(float(row["close"]), 2),
                "volume": int(row["volume"])
            })
        
        stock_trades = trades_df[trades_df["code"].astype(str).str.zfill(6) == code]
        trades = []
        for _, t in stock_trades.iterrows():
            trades.append({
                "entry_date": t["entry_date"],
                "exit_date": t["exit_date"],
                "entry_price": float(t["entry_price"]),
                "exit_price": float(t["exit_price"]),
                "pattern": t["pattern"],
                "pnl_pct": float(t["pnl_pct"]) / 100,
                "hold_days": int(t["hold_days"]),
                "exit_reason": t["exit_reason"],
                "max_profit": float(t.get("max_profit", 0))
            })
        
        stock_kline_data[code] = {
            "name": name_map.get(code, code),
            "klines": klines,
            "trades": trades
        }
    except Exception as e:
        print(f"  {code} 导出失败: {e}")

print(f"成功导出 {len(stock_kline_data)} 只股票的K线数据")
total_klines = sum(len(s["klines"]) for s in stock_kline_data.values())
total_trades = sum(len(s["trades"]) for s in stock_kline_data.values())
print(f"总计 {total_klines} 根K线, {total_trades} 笔交易")

with open("web/stock_kline_data_expanded.js", "w", encoding="utf-8") as f:
    f.write(f"const STOCK_KLINE_DATA = {json.dumps(stock_kline_data, ensure_ascii=False)};\n")

# 检查文件大小
size = os.path.getsize("web/stock_kline_data_expanded.js") / 1024 / 1024
print(f"文件大小: {size:.1f}MB")
print(f"已导出 web/stock_kline_data_expanded.js")

print("\n全部导出完成！")
