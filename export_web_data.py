# -*- coding: utf-8 -*-
"""导出回测数据为JSON，供交互式网站使用"""
import json
import pandas as pd
import numpy as np

def export_backtest_data(trades_csv="results/trades_hs300.csv",
                         equity_csv="results/equity_hs300.csv",
                         output_path="web/backtest_data.js"):
    """导出回测数据为JS变量"""
    import os
    os.makedirs("web", exist_ok=True)

    # 交易数据
    trades = pd.read_csv(trades_csv)
    trades["entry_date"] = pd.to_datetime(trades["entry_date"]).dt.strftime("%Y-%m-%d")
    trades["exit_date"] = pd.to_datetime(trades["exit_date"]).dt.strftime("%Y-%m-%d")
    trades["pnl_pct"] = (trades["pnl_pct"] * 100).round(2)
    trades["pnl"] = trades["pnl"].round(2)
    trades["entry_price"] = trades["entry_price"].round(2)
    trades["exit_price"] = trades["exit_price"].round(2)

    trades_list = trades.to_dict(orient="records")

    # 净值曲线（降采样到每周，减少数据量）
    equity = pd.read_csv(equity_csv)
    equity["date"] = pd.to_datetime(equity["date"]).dt.strftime("%Y-%m-%d")
    equity["equity"] = equity["equity"].round(2)
    # 每周取一个点
    equity["date_dt"] = pd.to_datetime(equity["date"])
    equity_weekly = equity.set_index("date_dt").resample("W").last().dropna().reset_index()
    equity_weekly["date"] = equity_weekly["date_dt"].dt.strftime("%Y-%m-%d")
    equity_list = equity_weekly[["date", "equity"]].to_dict(orient="records")

    # 计算指标
    total_trades = len(trades)
    wins = trades[trades["pnl"] > 0]
    losses = trades[trades["pnl"] <= 0]
    win_rate = len(wins) / total_trades if total_trades > 0 else 0
    avg_win = wins["pnl_pct"].mean() if len(wins) > 0 else 0
    avg_loss = losses["pnl_pct"].mean() if len(losses) > 0 else 0
    profit_factor = abs(wins["pnl"].sum() / losses["pnl"].sum()) if losses["pnl"].sum() != 0 else 0
    total_return = (equity["equity"].iloc[-1] / equity["equity"].iloc[0] - 1) * 100
    days = (pd.to_datetime(equity["date"]).iloc[-1] - pd.to_datetime(equity["date"]).iloc[0]).days
    annual_return = ((1 + total_return / 100) ** (365 / days) - 1) * 100
    daily_ret = equity["equity"].pct_change().dropna()
    sharpe = np.sqrt(252) * daily_ret.mean() / daily_ret.std() if daily_ret.std() > 0 else 0
    equity["peak"] = equity["equity"].cummax()
    equity["drawdown"] = ((equity["equity"] - equity["peak"]) / equity["peak"] * 100).round(2)
    max_drawdown = equity["drawdown"].min()
    avg_hold = trades["hold_days"].mean()

    # 回撤曲线（每周）
    dd_weekly = equity_weekly.copy()
    dd_weekly["peak"] = dd_weekly["equity"].cummax()
    dd_weekly["drawdown"] = ((dd_weekly["equity"] - dd_weekly["peak"]) / dd_weekly["peak"] * 100).round(2)
    drawdown_list = dd_weekly[["date", "drawdown"]].to_dict(orient="records")

    # 形态统计
    pattern_stats = []
    for pat in trades["pattern"].unique():
        pat_df = trades[trades["pattern"] == pat]
        pattern_stats.append({
            "pattern": pat,
            "count": len(pat_df),
            "win_rate": round((pat_df["pnl"] > 0).mean() * 100, 1),
            "avg_return": round(pat_df["pnl_pct"].mean(), 2),
            "avg_hold": round(pat_df["hold_days"].mean(), 1)
        })

    # 出场原因
    exit_stats = trades["exit_reason"].value_counts().to_dict()

    # 分年度
    trades["year"] = pd.to_datetime(trades["exit_date"]).dt.year
    yearly = []
    for year in sorted(trades["year"].unique()):
        y_df = trades[trades["year"] == year]
        yearly.append({
            "year": int(year),
            "trades": len(y_df),
            "win_rate": round((y_df["pnl"] > 0).mean() * 100, 1),
            "avg_return": round(y_df["pnl_pct"].mean(), 2)
        })

    metrics = {
        "total_trades": total_trades,
        "win_rate": round(win_rate * 100, 2),
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
        "profit_factor": round(profit_factor, 2),
        "total_return": round(total_return, 2),
        "annual_return": round(annual_return, 2),
        "sharpe": round(sharpe, 2),
        "max_drawdown": round(max_drawdown, 2),
        "avg_hold_days": round(avg_hold, 1)
    }

    # 写入JS文件
    data = {
        "metrics": metrics,
        "trades": trades_list,
        "equity": equity_list,
        "drawdown": drawdown_list,
        "pattern_stats": pattern_stats,
        "exit_stats": exit_stats,
        "yearly": yearly
    }

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"const BACKTEST_DATA = {json.dumps(data, ensure_ascii=False)};\n")

    print(f"数据已导出: {output_path}")
    print(f"  交易: {len(trades_list)} 笔")
    print(f"  净值点: {len(equity_list)} 个")
    print(f"  指标: {metrics}")


if __name__ == "__main__":
    export_backtest_data()
