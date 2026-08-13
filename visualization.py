# -*- coding: utf-8 -*-
"""
可视化模块
生成净值曲线、回撤曲线、交易分布、月度收益热力图等
"""
import warnings
warnings.filterwarnings("ignore")
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import rcParams

# 中文字体设置
rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
rcParams["axes.unicode_minus"] = False

OUTPUT_DIR = "results/charts"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def plot_equity_curve(equity_df, benchmark_df=None, title="策略净值曲线"):
    """绘制净值曲线"""
    if equity_df is None or equity_df.empty:
        return
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(equity_df["date"], equity_df["equity"] / 1000000,
            label="策略净值", color="#1f77b4", linewidth=1.5)

    if benchmark_df is not None and not benchmark_df.empty:
        benchmark_df = benchmark_df.copy()
        benchmark_df = benchmark_df[benchmark_df["date"] >= equity_df["date"].min()]
        benchmark_df = benchmark_df[benchmark_df["date"] <= equity_df["date"].max()]
        if not benchmark_df.empty:
            base = benchmark_df["close"].iloc[0]
            ax.plot(benchmark_df["date"], benchmark_df["close"] / base,
                    label="上证指数", color="#ff7f0e", linewidth=1, alpha=0.7)

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("日期")
    ax.set_ylabel("净值（初始=1）")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.xticks(rotation=45)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "equity_curve.png")
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def plot_drawdown(equity_df, title="策略回撤曲线"):
    """绘制回撤曲线"""
    if equity_df is None or equity_df.empty:
        return
    df = equity_df.copy()
    df["peak"] = df["equity"].cummax()
    df["drawdown"] = (df["equity"] - df["peak"]) / df["peak"]

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.fill_between(df["date"], df["drawdown"] * 100, 0,
                    color="#d62728", alpha=0.4)
    ax.plot(df["date"], df["drawdown"] * 100, color="#d62728", linewidth=1)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("日期")
    ax.set_ylabel("回撤 (%)")
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.xticks(rotation=45)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "drawdown.png")
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def plot_trade_distribution(trades_df, title="交易盈亏分布"):
    """绘制交易盈亏分布直方图"""
    if trades_df is None or trades_df.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 盈亏分布
    ax1 = axes[0]
    returns = trades_df["pnl_pct"] * 100
    ax1.hist(returns, bins=30, color="#1f77b4", alpha=0.7, edgecolor="white")
    ax1.axvline(0, color="red", linestyle="--", linewidth=1.5, label="盈亏平衡")
    ax1.axvline(returns.mean(), color="green", linestyle="--", linewidth=1.5,
                label=f"均值={returns.mean():.2f}%")
    ax1.set_title("单笔交易收益率分布", fontsize=12, fontweight="bold")
    ax1.set_xlabel("收益率 (%)")
    ax1.set_ylabel("交易次数")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 持仓天数分布
    ax2 = axes[1]
    ax2.hist(trades_df["hold_days"], bins=20, color="#2ca02c", alpha=0.7, edgecolor="white")
    ax2.axvline(trades_df["hold_days"].mean(), color="red", linestyle="--", linewidth=1.5,
                label=f"均值={trades_df['hold_days'].mean():.1f}天")
    ax2.set_title("持仓天数分布", fontsize=12, fontweight="bold")
    ax2.set_xlabel("持仓天数")
    ax2.set_ylabel("交易次数")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "trade_distribution.png")
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def plot_monthly_returns(trades_df, title="月度收益热力图"):
    """绘制月度收益热力图"""
    if trades_df is None or trades_df.empty:
        return
    trades_df = trades_df.copy()
    trades_df["exit_date"] = pd.to_datetime(trades_df["exit_date"])
    trades_df["year"] = trades_df["exit_date"].dt.year
    trades_df["month"] = trades_df["exit_date"].dt.month

    # 按月汇总收益
    monthly = trades_df.groupby(["year", "month"])["pnl_pct"].sum().unstack()
    monthly = monthly.reindex(columns=range(1, 13))

    fig, ax = plt.subplots(figsize=(12, max(4, len(monthly) * 0.6)))
    im = ax.imshow(monthly.values * 100, cmap="RdYlGn", aspect="auto",
                   vmin=-10, vmax=10)
    ax.set_xticks(range(12))
    ax.set_xticklabels([f"{m}月" for m in range(1, 13)])
    ax.set_yticks(range(len(monthly)))
    ax.set_yticklabels(monthly.index)
    ax.set_title(title, fontsize=14, fontweight="bold")

    # 标注数值
    for i in range(len(monthly)):
        for j in range(12):
            val = monthly.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val*100:.1f}%", ha="center", va="center",
                        fontsize=8, color="black" if abs(val) < 0.05 else "white")

    plt.colorbar(im, ax=ax, label="收益率 (%)")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "monthly_heatmap.png")
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def plot_pattern_performance(trades_df, title="各形态表现对比"):
    """各形态表现对比柱状图"""
    if trades_df is None or trades_df.empty:
        return
    pattern_stats = trades_df.groupby("pattern").agg(
        count=("pnl", "count"),
        win_rate=("pnl", lambda x: (x > 0).mean()),
        avg_return=("pnl_pct", "mean"),
        total_pnl=("pnl", "sum")
    ).reset_index()

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    x = range(len(pattern_stats))
    labels = pattern_stats["pattern"]

    axes[0].bar(x, pattern_stats["count"], color="#1f77b4", alpha=0.7)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=45)
    axes[0].set_title("交易次数")
    axes[0].grid(True, alpha=0.3, axis="y")

    axes[1].bar(x, pattern_stats["win_rate"] * 100, color="#2ca02c", alpha=0.7)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=45)
    axes[1].set_title("胜率 (%)")
    axes[1].axhline(50, color="red", linestyle="--", alpha=0.5)
    axes[1].grid(True, alpha=0.3, axis="y")

    axes[2].bar(x, pattern_stats["avg_return"] * 100, color="#ff7f0e", alpha=0.7)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels, rotation=45)
    axes[2].set_title("平均收益率 (%)")
    axes[2].axhline(0, color="black", linewidth=0.5)
    axes[2].grid(True, alpha=0.3, axis="y")

    plt.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "pattern_performance.png")
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def plot_exit_reasons(trades_df, title="出场原因分布"):
    """出场原因饼图"""
    if trades_df is None or trades_df.empty:
        return
    exit_counts = trades_df["exit_reason"].value_counts()

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    wedges, texts, autotexts = ax.pie(
        exit_counts.values, labels=exit_counts.index,
        autopct="%1.1f%%", colors=colors[:len(exit_counts)],
        startangle=90
    )
    ax.set_title(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "exit_reasons.png")
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def generate_all_charts(metrics, index_df=None):
    """生成所有图表"""
    charts = {}
    equity_df = metrics.get("equity_df")
    trades_df = metrics.get("trades_df")

    if equity_df is not None and not equity_df.empty:
        charts["equity"] = plot_equity_curve(equity_df, index_df)
        charts["drawdown"] = plot_drawdown(equity_df)

    if trades_df is not None and not trades_df.empty:
        charts["distribution"] = plot_trade_distribution(trades_df)
        charts["monthly"] = plot_monthly_returns(trades_df)
        charts["pattern"] = plot_pattern_performance(trades_df)
        charts["exit"] = plot_exit_reasons(trades_df)

    print(f"  生成图表 {len(charts)} 张，保存到 {OUTPUT_DIR}/")
    return charts
