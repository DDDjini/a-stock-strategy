# -*- coding: utf-8 -*-
"""
鲁棒性测试模块
1. 参数敏感性分析（量比阈值、质量阈值、突破缓冲）
2. 不同市场周期表现（牛市/熊市/震荡市）
3. 滚动窗口回测
4. 分年度表现
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import config as C
from data_fetcher import get_index_daily
from backtest_engine import BacktestEngine


def run_with_config(stock_data, index_df, config_overrides=None):
    """用指定参数覆盖运行回测"""
    if config_overrides:
        for key, value in config_overrides.items():
            setattr(C, key, value)

    engine = BacktestEngine(initial_capital=C.INITIAL_CAPITAL)
    metrics = engine.run(stock_data, index_df)
    return metrics


def parameter_sensitivity(stock_data, index_df):
    """
    参数敏感性分析（精简版：3x3网格）
    """
    print("\n" + "=" * 60)
    print("  参数敏感性分析")
    print("=" * 60)

    results = []
    vol_ratios = [0.8, 1.0, 1.5]
    quality_thresholds = [40, 55, 70]

    orig_vol = C.BREAKOUT_VOL_RATIO
    orig_qual = C.PATTERN_QUALITY_THRESHOLD

    for vol_r in vol_ratios:
        for qual in quality_thresholds:
            C.BREAKOUT_VOL_RATIO = vol_r
            C.PATTERN_QUALITY_THRESHOLD = qual
            metrics = run_with_config(stock_data, index_df)
            results.append({
                "vol_ratio": vol_r,
                "quality_thresh": qual,
                "trades": metrics["total_trades"],
                "win_rate": metrics["win_rate"],
                "total_return": metrics["total_return"],
                "sharpe": metrics["sharpe"],
                "max_drawdown": metrics["max_drawdown"]
            })
            print(f"  量比>={vol_r}, 质量>={qual}: "
                  f"{metrics['total_trades']}笔, 胜率{metrics['win_rate']:.1%}, "
                  f"收益{metrics['total_return']:.1%}, 夏普{metrics['sharpe']:.2f}")

    C.BREAKOUT_VOL_RATIO = orig_vol
    C.PATTERN_QUALITY_THRESHOLD = orig_qual

    df = pd.DataFrame(results)
    df.to_csv("results/sensitivity_analysis.csv", index=False, encoding="utf-8-sig")
    return df


def period_analysis(stock_data, index_df):
    """
    分市场周期分析
    2019-2021: 结构性牛市
    2022: 熊市
    2023-2024: 震荡市
    """
    print("\n" + "=" * 60)
    print("  分市场周期表现")
    print("=" * 60)

    periods = [
        ("2019-2021 牛市", "2019-01-01", "2021-12-31"),
        ("2022 熊市", "2022-01-01", "2022-12-31"),
        ("2023-2024 震荡", "2023-01-01", "2024-12-31"),
        ("2025-2026", "2025-01-01", "2026-08-01"),
    ]

    results = []
    for name, start, end in periods:
        # 过滤数据到该周期
        filtered_data = {}
        for code, df in stock_data.items():
            mask = (df["date"] >= start) & (df["date"] <= end)
            df_filtered = df[mask].reset_index(drop=True)
            if len(df_filtered) >= 60:
                filtered_data[code] = df_filtered

        if not filtered_data:
            continue

        engine = BacktestEngine(initial_capital=C.INITIAL_CAPITAL)
        metrics = engine.run(filtered_data, index_df)

        results.append({
            "period": name,
            "trades": metrics["total_trades"],
            "win_rate": metrics["win_rate"],
            "total_return": metrics["total_return"],
            "max_drawdown": metrics["max_drawdown"],
            "sharpe": metrics["sharpe"]
        })
        print(f"  {name}: {metrics['total_trades']}笔, "
              f"胜率{metrics['win_rate']:.1%}, 收益{metrics['total_return']:.1%}, "
              f"回撤{metrics['max_drawdown']:.1%}")

    df = pd.DataFrame(results)
    df.to_csv("results/period_analysis.csv", index=False, encoding="utf-8-sig")
    return df


def yearly_breakdown(trades_df):
    """分年度交易统计"""
    print("\n" + "=" * 60)
    print("  分年度交易统计")
    print("=" * 60)

    if trades_df.empty:
        return pd.DataFrame()

    trades_df = trades_df.copy()
    trades_df["year"] = pd.to_datetime(trades_df["entry_date"]).dt.year
    yearly = trades_df.groupby("year").agg(
        trades=("pnl", "count"),
        wins=("pnl", lambda x: (x > 0).sum()),
        avg_return=("pnl_pct", "mean"),
        total_pnl=("pnl", "sum")
    ).reset_index()
    yearly["win_rate"] = yearly["wins"] / yearly["trades"]

    for _, row in yearly.iterrows():
        print(f"  {int(row['year'])}: {int(row['trades'])}笔, "
              f"胜率{row['win_rate']:.1%}, 均收益{row['avg_return']:.2%}")

    yearly.to_csv("results/yearly_breakdown.csv", index=False, encoding="utf-8-sig")
    return yearly


def rolling_window_test(stock_data, index_df, window_years=2, step_years=1):
    """
    滚动窗口回测
    每2年一个窗口，步长1年，检验策略稳定性
    """
    print("\n" + "=" * 60)
    print("  滚动窗口回测")
    print("=" * 60)

    all_dates = []
    for df in stock_data.values():
        if not df.empty:
            all_dates.extend(df["date"].tolist())
    if not all_dates:
        return pd.DataFrame()

    min_date = min(all_dates)
    max_date = max(all_dates)

    results = []
    start = min_date
    while start + pd.Timedelta(days=window_years * 365) <= max_date:
        end = start + pd.Timedelta(days=window_years * 365)
        window_name = f"{start.strftime('%Y')}-{end.strftime('%Y')}"

        filtered_data = {}
        for code, df in stock_data.items():
            mask = (df["date"] >= start) & (df["date"] <= end)
            df_filtered = df[mask].reset_index(drop=True)
            if len(df_filtered) >= 60:
                filtered_data[code] = df_filtered

        if filtered_data:
            engine = BacktestEngine(initial_capital=C.INITIAL_CAPITAL)
            metrics = engine.run(filtered_data, index_df)
            results.append({
                "window": window_name,
                "trades": metrics["total_trades"],
                "win_rate": metrics["win_rate"],
                "total_return": metrics["total_return"],
                "sharpe": metrics["sharpe"]
            })
            print(f"  {window_name}: {metrics['total_trades']}笔, "
                  f"胜率{metrics['win_rate']:.1%}, 收益{metrics['total_return']:.1%}")

        start += pd.Timedelta(days=step_years * 365)

    df = pd.DataFrame(results)
    if not df.empty:
        df.to_csv("results/rolling_window.csv", index=False, encoding="utf-8-sig")
    return df


def run_all_robustness_tests(stock_data, index_df, metrics):
    """运行所有鲁棒性测试"""
    print("\n" + "#" * 70)
    print("#  鲁棒性测试")
    print("#" * 70)

    # 1. 参数敏感性（用前30只股票子集加速）
    subset = dict(list(stock_data.items())[:30])
    sensitivity_df = parameter_sensitivity(subset, index_df)

    # 2. 分周期
    period_df = period_analysis(stock_data, index_df)

    # 3. 分年度
    yearly_df = yearly_breakdown(metrics.get("trades_df", pd.DataFrame()))

    # 4. 滚动窗口（用子集30只）
    rolling_df = rolling_window_test(subset, index_df)

    return {
        "sensitivity": sensitivity_df,
        "period": period_df,
        "yearly": yearly_df,
        "rolling": rolling_df
    }
