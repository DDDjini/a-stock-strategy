# -*- coding: utf-8 -*-
"""
整合优化版策略 - 简化版
在信号扫描层应用三大优化，复用基础回测引擎
"""
import warnings
warnings.filterwarnings("ignore")
import os
import numpy as np
import pandas as pd

import config as C
from data_fetcher import compute_technical_indicators, get_index_daily
from pattern_recognizer import detect_all_patterns, find_resistance_levels
from strategy import check_market_filter, check_false_breakout
from backtest_engine import BacktestEngine, Position, print_metrics
from market_regime import MarketRegimeDetector, get_regime_params
from train_ml_from_trades import score_signal, load_ml_model
from minute_confirmer import MinuteLevelConfirmer


def scan_optimized_signals_for_stock(df, index_df, regime_df, regime_detector,
                                     stock_code="", use_regime=True, use_ml=True,
                                     use_minute=True, ml_threshold=55,
                                     minute_confirmer=None):
    """对单只股票扫描优化后的信号"""
    signals = []
    df = compute_technical_indicators(df.copy())
    if df is None or len(df) < 150:
        return signals

    resistance_levels = find_resistance_levels(df)
    if not resistance_levels:
        return signals

    for res_price, first_idx, last_idx, touch_count in resistance_levels:
        if last_idx < C.PATTERN_LOOKBACK:
            continue

        lookback_start = max(0, last_idx - C.PATTERN_LOOKBACK)
        window_df = df.iloc[lookback_start:last_idx + 1].reset_index(drop=True)
        patterns = detect_all_patterns(window_df)
        if not patterns:
            continue

        search_end = min(len(df), last_idx + 120)
        for pattern in patterns:
            for i in range(last_idx + 1, search_end):
                if i >= len(df) - 5:
                    break
                row_i = df.iloc[i]
                prev_close = df.iloc[i - 1]["close"]
                vol_ma3 = df.iloc[max(0, i - 3):i]["volume"].mean()

                # 基础突破
                if not (row_i["close"] > res_price * (1 + C.BREAKOUT_CLOSE_BUFFER) and
                        prev_close <= res_price and
                        row_i["volume"] > vol_ma3 * C.BREAKOUT_VOL_RATIO):
                    continue

                # === 优化1: 市场状态 ===
                if use_regime and regime_df is not None:
                    regime = regime_detector.get_regime_on_date(regime_df, row_i["date"])
                    rparams = get_regime_params(regime)
                    if pattern.get("quality", 50) < rparams.get("quality_threshold", 50):
                        continue
                    if row_i["volume"] < vol_ma3 * rparams.get("vol_ratio", 1.0):
                        continue
                    if rparams.get("use_market_filter", True) and not check_market_filter(index_df, row_i["date"]):
                        continue
                else:
                    regime = "default"
                    if pattern.get("quality", 50) < C.PATTERN_QUALITY_THRESHOLD:
                        continue
                    if C.INDEX_FILTER and not check_market_filter(index_df, row_i["date"]):
                        continue

                # === 优化2: ML评分 ===
                if use_ml:
                    ml_score = score_signal(df, i, res_price, pattern.get("pattern", "矩形底"), index_df)
                    if ml_score < ml_threshold:
                        continue
                else:
                    ml_score = pattern.get("quality", 50)

                # === 优化3: 60分钟确认 ===
                minute_ok = True
                minute_score = 0
                if use_minute and minute_confirmer is not None:
                    confirm = minute_confirmer.confirm_breakout(stock_code, row_i["date"], res_price, df)
                    minute_ok = confirm["confirmed"]
                    minute_score = confirm["details"].get("confirmation_score", 0)
                    if not minute_ok:
                        continue

                # 假突破过滤
                if C.USE_FALSE_BREAKOUT_FILTER and not check_false_breakout(df, i, res_price):
                    continue

                signals.append({
                    "idx": i,
                    "date": row_i["date"],
                    "price": row_i["close"],
                    "neckline": res_price,
                    "pattern": pattern.get("pattern", "unknown"),
                    "pattern_type": pattern.get("pattern", "unknown"),
                    "quality_score": pattern.get("quality", 50),
                    "ml_score": ml_score,
                    "regime": regime,
                    "minute_confirmed": minute_ok,
                    "minute_score": minute_score,
                })
                break

    return signals


def run_optimized_backtest(stock_data, index_df, use_regime=True, use_ml=True,
                           use_minute=True, ml_threshold=55, initial_capital=1000000):
    """运行整合优化版回测"""
    print("\n" + "=" * 60)
    print("  整合优化版回测")
    print(f"  市场状态:{'开' if use_regime else '关'} ML:{'开' if use_ml else '关'} 60min:{'开' if use_minute else '关'}")
    print(f"  ML阈值: {ml_threshold}")
    print("=" * 60)

    # 初始化市场状态
    regime_detector = None
    regime_df = None
    if use_regime:
        regime_detector = MarketRegimeDetector(n_regimes=3)
        regime_detector.fit(index_df)
        regime_df = regime_detector.predict(index_df)
        print(f"  市场状态: {regime_df['regime'].value_counts().to_dict()}")

    # 初始化ML
    if use_ml:
        model, cols = load_ml_model()
        if model is None:
            print("  警告：ML模型未找到，关闭ML")
            use_ml = False
        else:
            print(f"  ML模型已加载（{len(cols)}特征）")

    # 初始化60分钟
    minute_confirmer = MinuteLevelConfirmer() if use_minute else None

    # 扫描信号
    print("\n  扫描优化信号...")
    all_signals = {}
    total_candidates = 0
    for code, df in stock_data.items():
        sigs = scan_optimized_signals_for_stock(
            df, index_df, regime_df, regime_detector,
            code, use_regime, use_ml, use_minute, ml_threshold, minute_confirmer
        )
        if sigs:
            all_signals[code] = sigs

    total_sigs = sum(len(s) for s in all_signals.values())
    print(f"  共 {len(all_signals)} 只股票有信号，总计 {total_sigs} 个信号")

    if total_sigs == 0:
        print("  无信号！")
        return None

    # 构建信号集合 (code, date) -> signal
    signal_set = {}
    for code, sigs in all_signals.items():
        for sig in sigs:
            signal_set[(code, pd.Timestamp(sig["date"]))] = sig

    # 用基础回测引擎，但修改入场逻辑只接受信号集合中的信号
    engine = BacktestEngine(initial_capital=initial_capital)

    # 预扫描（基础引擎会自己扫描，我们需要覆盖它的信号）
    # 简单方法：直接运行逐日回测，用signal_set控制入场
    print("  开始逐日回测...")
    trading_dates = sorted(set(index_df["date"].values))
    equity_curve = []

    # 建立code->df的日期索引映射
    date_idx_map = {}
    for code, df in stock_data.items():
        date_idx_map[code] = {pd.Timestamp(d): i for i, d in enumerate(df["date"].values)}

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

        # 入场（只接受signal_set中的信号）
        for (sig_code, sig_date), sig in signal_set.items():
            if sig_code in engine.positions:
                continue
            if len(engine.positions) >= C.MAX_POSITIONS:
                break
            if sig_date != date_ts:
                continue
            # 次日开盘买入（找到下一个交易日）
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

    # 计算指标
    engine.equity_curve = equity_curve
    metrics = engine._compute_metrics()
    return metrics


if __name__ == "__main__":
    from run_backtest import get_hs300_constituents, fetch_universe_data

    print("获取数据...")
    stocks = get_hs300_constituents()
    stock_codes = stocks["code"].tolist()[:50]
    index_df = get_index_daily(C.INDEX_CODE, C.DATA_START_DATE, C.DATA_END_DATE)
    stock_data = fetch_universe_data(stock_codes, C.DATA_START_DATE, C.DATA_END_DATE)
    print(f"有效股票: {len(stock_data)} 只")

    metrics = run_optimized_backtest(
        stock_data, index_df,
        use_regime=True, use_ml=True, use_minute=False,  # 先关60min（数据获取慢）
        ml_threshold=55
    )
    if metrics:
        print_metrics(metrics)
