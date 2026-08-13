# -*- coding: utf-8 -*-
"""
主运行脚本 - A股K线形态突破+趋势跟踪策略
1. 获取沪深300成分股数据
2. 运行回测
3. 输出结果
"""
import os
import sys
import time
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import akshare as ak

import config as C
from data_fetcher import (get_stock_list, get_stock_daily, get_index_daily,
                          compute_technical_indicators, get_weekly_data)
from pattern_recognizer import detect_all_patterns
from strategy import scan_breakout_signals
from backtest_engine import BacktestEngine, print_metrics


def get_hs300_constituents():
    """获取沪深300成分股列表"""
    cache_file = os.path.join("data_cache", "hs300_constituents.csv")
    if os.path.exists(cache_file):
        mtime = os.path.getmtime(cache_file)
        if time.time() - mtime < 86400 * 7:
            return pd.read_csv(cache_file, dtype={"code": str})

    try:
        df = ak.index_stock_cons_csindex(symbol="000300")
        df.columns = [c.lower() for c in df.columns]
        # 标准化列名
        if "成分券代码" in df.columns:
            df = df.rename(columns={"成分券代码": "code", "成分券名称": "name"})
        elif "code" not in df.columns:
            # 尝试其他列名
            for col in df.columns:
                if "代码" in col or "code" in col.lower():
                    df = df.rename(columns={col: "code"})
                if "名称" in col or "name" in col.lower():
                    df = df.rename(columns={col: "name"})
        df["code"] = df["code"].astype(str).str.zfill(6)
        df = df[["code", "name"]].drop_duplicates().reset_index(drop=True)
        df.to_csv(cache_file, index=False)
        return df
    except Exception as e:
        print(f"获取沪深300成分失败: {e}，使用全部A股前300只")
        stocks = get_stock_list()
        return stocks.head(300)


def get_csi500_constituents():
    """获取中证500成分股"""
    cache_file = os.path.join("data_cache", "csi500_constituents.csv")
    if os.path.exists(cache_file):
        mtime = os.path.getmtime(cache_file)
        if time.time() - mtime < 86400 * 7:
            return pd.read_csv(cache_file, dtype={"code": str})

    try:
        df = ak.index_stock_cons_csindex(symbol="000905")
        df.columns = [c.lower() for c in df.columns]
        for col in df.columns:
            if "代码" in col or "code" in col.lower():
                df = df.rename(columns={col: "code"})
            if "名称" in col or "name" in col.lower():
                df = df.rename(columns={col: "name"})
        df["code"] = df["code"].astype(str).str.zfill(6)
        df = df[["code", "name"]].drop_duplicates().reset_index(drop=True)
        df.to_csv(cache_file, index=False)
        return df
    except Exception as e:
        print(f"获取中证500成分失败: {e}")
        return pd.DataFrame()


def fetch_universe_data(stock_codes, start_date, end_date, max_stocks=None):
    """批量获取股票数据"""
    stock_data = {}
    total = len(stock_codes) if max_stocks is None else min(max_stocks, len(stock_codes))
    success = 0

    for i, code in enumerate(stock_codes[:max_stocks]):
        if (i + 1) % 20 == 0:
            print(f"  数据获取进度: {i + 1}/{total} (成功{success})")

        df = get_stock_daily(code, start_date, end_date, adjust=C.ADJUST_TYPE)
        if df is None or df.empty:
            continue
        if len(df) < C.MIN_LIST_DAYS:
            continue

        # 价格和流动性过滤
        avg_price = df["close"].mean()
        avg_vol = df["volume"].mean()
        if avg_price < C.MIN_PRICE or avg_price > C.MAX_PRICE:
            continue
        if avg_vol < C.MIN_AVG_VOLUME:
            continue

        df = compute_technical_indicators(df)
        stock_data[code] = df
        success += 1

    print(f"  数据获取完成: 成功{success}/{total}")
    return stock_data


def run_backtest(universe="hs300", max_stocks=None):
    """运行完整回测"""
    print("=" * 70)
    print("  A股K线形态突破+趋势跟踪策略 - 回测")
    print("=" * 70)

    # 1. 获取股票池
    print(f"\n[1/5] 获取{universe}成分股...")
    if universe == "hs300":
        stocks = get_hs300_constituents()
    elif universe == "csi500":
        stocks = get_csi500_constituents()
    else:
        stocks = get_stock_list()

    if stocks.empty:
        print("股票池为空，退出")
        return None

    stock_codes = stocks["code"].tolist()
    print(f"  股票池: {len(stock_codes)} 只")

    # 2. 获取指数数据（大盘过滤）
    print("\n[2/5] 获取大盘指数数据...")
    index_df = get_index_daily(C.INDEX_CODE, C.DATA_START_DATE, C.DATA_END_DATE)
    if not index_df.empty:
        print(f"  上证指数: {len(index_df)} 条记录")
    else:
        print("  [WARN] 指数数据获取失败，将禁用大盘过滤")
        C.INDEX_FILTER = False

    # 3. 获取股票数据
    print(f"\n[3/5] 获取股票日线数据 ({C.DATA_START_DATE} ~ {C.DATA_END_DATE})...")
    stock_data = fetch_universe_data(stock_codes, C.DATA_START_DATE, C.DATA_END_DATE, max_stocks)

    if not stock_data:
        print("无有效股票数据，退出")
        return None

    # 4. 运行回测
    print("\n[4/5] 运行回测引擎...")
    engine = BacktestEngine(initial_capital=C.INITIAL_CAPITAL)
    metrics = engine.run(stock_data, index_df)

    # 5. 输出结果
    print("\n[5/5] 回测结果:")
    print_metrics(metrics)

    # 保存结果
    os.makedirs("results", exist_ok=True)
    if not metrics["trades_df"].empty:
        metrics["trades_df"].to_csv("results/trades.csv", index=False, encoding="utf-8-sig")
    if not metrics["equity_df"].empty:
        metrics["equity_df"].to_csv("results/equity_curve.csv", index=False, encoding="utf-8-sig")

    # 保存绩效摘要
    summary = {k: v for k, v in metrics.items() if k not in ["trades_df", "equity_df", "pattern_stats", "exit_stats"]}
    summary["pattern_stats"] = str(metrics["pattern_stats"])
    summary["exit_stats"] = str(metrics["exit_stats"])
    pd.Series(summary).to_csv("results/summary.csv", encoding="utf-8-sig")

    print("\n结果已保存到 results/ 目录")
    return metrics


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--universe", default="hs300", choices=["hs300", "csi500", "all"])
    parser.add_argument("--max-stocks", type=int, default=None)
    args = parser.parse_args()

    metrics = run_backtest(universe=args.universe, max_stocks=args.max_stocks)
