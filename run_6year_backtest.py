# -*- coding: utf-8 -*-
"""扩展股票池(663只)6年优化版回测"""
import warnings
warnings.filterwarnings("ignore")
import sys, os, json
sys.path.insert(0, ".")
import numpy as np
import pandas as pd

import config as C
from data_fetcher import get_stock_daily, get_index_daily, compute_technical_indicators
from optimized_strategy import scan_optimized_signals_for_stock
from market_regime import MarketRegimeDetector, get_regime_params
from train_ml_from_trades import load_ml_model
from strategy import check_market_filter, check_false_breakout
from backtest_engine import BacktestEngine

START = "20200801"
END = "20260812"
ML_THRESHOLD = 55

print("=" * 60)
print("扩展股票池优化版回测（6年）")
print("=" * 60)

# 1. 读取股票池
universe = pd.read_csv('data_cache/expanded_universe.csv')
universe['code'] = universe['code'].astype(str).str.zfill(6)
stock_codes = universe['code'].tolist()
print(f"股票池: {len(stock_codes)}只")

# 2. 加载指数数据
print("\n加载指数数据...")
index_df = get_index_daily(C.INDEX_CODE, START, END)
index_df = compute_technical_indicators(index_df)
print(f"指数数据: {len(index_df)}个交易日")

# 3. 加载股票数据
print("\n加载股票数据...")
stock_data = {}
for code in stock_codes:
    cache_file = f'data_cache/{code}_{START}_{END}_qfq.parquet'
    if not os.path.exists(cache_file):
        continue
    try:
        df = pd.read_parquet(cache_file)
        if df is not None and len(df) >= 60:
            df = compute_technical_indicators(df)
            stock_data[code] = df
    except:
        pass
    if len(stock_data) % 100 == 0 and len(stock_data) > 0:
        print(f"  已加载 {len(stock_data)} 只")

print(f"有效股票: {len(stock_data)} 只")

# 4. 初始化优化模块
print("\n初始化市场状态识别和ML模型...")
regime_detector = MarketRegimeDetector(n_regimes=3)
regime_detector.fit(index_df)
regime_df = regime_detector.predict(index_df)
model, cols = load_ml_model()
print(f"ML模型已加载 (特征数: {len(cols)})")

# 5. 扫描信号
print("\n扫描优化信号...")
all_signals = {}
for i, (code, df) in enumerate(stock_data.items()):
    sigs = scan_optimized_signals_for_stock(
        df, index_df, regime_df, regime_detector, code,
        use_regime=True, use_ml=True, use_minute=False,
        ml_threshold=ML_THRESHOLD, minute_confirmer=None
    )
    if sigs:
        all_signals[code] = sigs
    if (i + 1) % 100 == 0:
        print(f"  已扫描 {i+1}/{len(stock_data)}, 有信号股票 {len(all_signals)}")

total_sigs = sum(len(s) for s in all_signals.values())
print(f"共 {len(all_signals)} 只股票有信号，总计 {total_sigs} 个信号")

# 6. 构建signal_set
signal_set = {}
for code, sigs in all_signals.items():
    for sig in sigs:
        signal_set[(code, pd.Timestamp(sig["date"]))] = sig

# 7. 回测
print("\n逐日回测...")
engine = BacktestEngine(initial_capital=C.INITIAL_CAPITAL)
trading_dates = sorted(set(index_df["date"].values))
equity_curve = []
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

# 8. 保存结果
trades_df = pd.DataFrame(engine.trades)
trades_df.to_csv("results/trades_optimized_6year.csv", index=False)
equity_df = pd.DataFrame(equity_curve)
equity_df.to_csv("results/equity_optimized_6year.csv", index=False)

print(f"\n{'='*60}")
print(f"回测结果（{len(stock_data)}只股票，6年）")
print(f"{'='*60}")
print(f"总交易数: {metrics['total_trades']}")
print(f"胜率: {metrics['win_rate']*100:.2f}%")
print(f"平均盈利: {metrics['avg_win_pct']*100:.2f}%")
print(f"平均亏损: {metrics['avg_loss_pct']*100:.2f}%")
print(f"盈亏比: {metrics['profit_factor']:.2f}")
print(f"总收益率: {metrics['total_return']*100:.2f}%")
print(f"年化收益率: {metrics['annual_return']*100:.2f}%")
print(f"夏普比率: {metrics['sharpe']:.2f}")
print(f"最大回撤: {metrics['max_drawdown']*100:.2f}%")
print(f"平均持仓: {metrics['avg_hold_days']:.1f}天")

# 形态统计
print(f"\n各形态表现:")
for pat in trades_df["pattern"].unique():
    pat_df = trades_df[trades_df["pattern"] == pat]
    wr = (pat_df["pnl"] > 0).mean() * 100
    avg = pat_df["pnl_pct"].mean() * 100
    print(f"  {pat}: {len(pat_df)}笔, 胜率{wr:.1f}%, 均收益{avg:.2f}%")

print(f"\n已保存 results/trades_optimized_6year.csv 和 results/equity_optimized_6year.csv")
