# -*- coding: utf-8 -*-
"""
策略逻辑模块（优化版v2）
架构：阻力位识别 → 形态回溯 → 突破确认 → 趋势线持仓
阻力位 = 多次触碰的价格水平（更符合实际颈线定义）
"""
import numpy as np
import pandas as pd
from scipy.stats import linregress
import config as C
from pattern_recognizer import (
    detect_all_patterns, find_swing_points_combined, find_resistance_levels
)


def check_breakout_at_idx(df, idx, neckline):
    """检查第idx日是否有效突破"""
    if idx < 3 or idx >= len(df):
        return False, 0, 0
    row = df.iloc[idx]
    if row["close"] <= neckline * (1 + C.BREAKOUT_CLOSE_BUFFER):
        return False, 0, 0
    if df.iloc[idx - 1]["close"] > neckline:
        return False, 0, 0
    vol_ma3 = df.iloc[idx - 3:idx]["volume"].mean()
    if vol_ma3 <= 0:
        return False, 0, 0
    vol_ratio = row["volume"] / vol_ma3
    if vol_ratio < C.BREAKOUT_VOL_RATIO:
        return False, 0, vol_ratio
    return True, row["close"], vol_ratio


def check_market_filter(index_df, date):
    if not C.INDEX_FILTER or index_df is None or index_df.empty:
        return True
    idx_data = index_df[index_df["date"] <= date]
    if len(idx_data) < C.INDEX_MA_PERIOD:
        return True
    ma60 = idx_data["close"].iloc[-C.INDEX_MA_PERIOD:].mean()
    return idx_data["close"].iloc[-1] > ma60


def check_false_breakout(df, idx, neckline, days=3):
    """假突破过滤：突破后N日收盘价不低于颈线的98%（允许正常回踩）"""
    if not C.USE_FALSE_BREAKOUT_FILTER:
        return True
    end_idx = min(idx + days, len(df) - 1)
    for i in range(idx + 1, end_idx + 1):
        if df.iloc[i]["close"] < neckline * 0.98:
            return False
    return True


def check_multi_timeframe(df_daily, date):
    if not C.USE_MULTI_TIMEFRAME:
        return True
    from data_fetcher import get_weekly_data
    df_w = get_weekly_data(df_daily[df_daily["date"] <= date])
    if len(df_w) < 20:
        return True
    df_w["ma20"] = df_w["close"].rolling(20).mean()
    return df_w["close"].iloc[-1] > df_w["ma20"].iloc[-1]


class TrendLineTracker:
    def __init__(self, entry_idx, entry_price, atr):
        self.entry_idx = entry_idx
        self.entry_price = entry_price
        self.atr = atr
        self.low_points = []
        self.trendline_slope = 0
        self.trendline_intercept = 0
        self.highest_close = entry_price
        self.trailing_stop = entry_price - C.INITIAL_STOP_ATR * atr

    def update(self, df, current_idx):
        if current_idx <= self.entry_idx:
            return
        row = df.iloc[current_idx]
        self.highest_close = max(self.highest_close, row["close"])

        if (current_idx - self.entry_idx) % 5 == 0 or len(self.low_points) < 2:
            segment = df.iloc[self.entry_idx:current_idx + 1]
            swings = find_swing_points_combined(segment, order=3)
            new_lows = [(s["idx"] + self.entry_idx, s["price"])
                        for s in swings if s["type"] == "low"
                        and s["price"] >= self.entry_price * 0.97]
            if len(new_lows) >= 2:
                self.low_points = new_lows[-3:]

        if len(self.low_points) >= 2:
            x = np.array([p[0] for p in self.low_points])
            y = np.array([p[1] for p in self.low_points])
            reg = linregress(x, y)
            self.trendline_slope = reg.slope
            self.trendline_intercept = reg.intercept
            if self.trendline_slope > 0:
                tl_val = self.trendline_intercept + self.trendline_slope * current_idx
                self.trailing_stop = max(self.trailing_stop,
                                         tl_val * (1 - C.TRENDLINE_BREAK_BUFFER))
        else:
            drawdown_stop = self.highest_close * (1 - 0.08)
            self.trailing_stop = max(self.trailing_stop, drawdown_stop)

    def get_trendline_value(self, idx):
        if len(self.low_points) >= 2 and self.trendline_slope != 0:
            return self.trendline_intercept + self.trendline_slope * idx
        return self.trailing_stop

    def check_exit(self, df, current_idx):
        if current_idx <= self.entry_idx:
            return False, 0, ""
        row = df.iloc[current_idx]
        trendline_val = self.get_trendline_value(current_idx)
        if row["close"] < trendline_val and self.trendline_slope > 0:
            return True, row["close"], "跌破趋势线"
        if row["close"] < self.trailing_stop:
            return True, row["close"], "移动止盈"
        hard_stop = self.entry_price - C.INITIAL_STOP_ATR * self.atr
        if row["low"] < hard_stop:
            exit_price = min(row["open"], hard_stop) if row["open"] < hard_stop else hard_stop
            return True, exit_price, "硬止损"
        return False, 0, ""


def scan_breakout_signals(df, index_df=None, start_idx=120):
    """
    高效信号扫描（阻力位驱动）
    1. 识别所有阻力位（多次触碰的价格水平）
    2. 对每个阻力位，回溯检测形态
    3. 找突破日并确认
    """
    if df is None or df.empty or len(df) < start_idx + 20:
        return []

    signals = []
    n = len(df)

    # 1. 识别阻力位（用全部数据，但只考虑start_idx之后的）
    resistance_levels = find_resistance_levels(df, tol=0.03, min_touches=2)

    for rl_price, rl_first, rl_last, rl_touches in resistance_levels:
        # 阻力位形成后才可能突破
        breakout_start = rl_last + 1
        if breakout_start < start_idx:
            breakout_start = start_idx
        if breakout_start >= n - 5:
            continue

        # 2. 回溯检测形态（用阻力位形成之前的数据）
        lookback_end = rl_last + 1
        lookback_start = max(0, lookback_end - C.PATTERN_LOOKBACK)
        df_slice = df.iloc[lookback_start:lookback_end].reset_index(drop=True)

        if len(df_slice) < C.MIN_PATTERN_DAYS:
            continue

        best_pattern = None
        try:
            patterns = detect_all_patterns(df_slice, len(df_slice))
            if patterns:
                best_pattern = patterns[0]
        except Exception:
            pass

        # 即使没有明确形态，只要有清晰阻力位也可作为箱体突破
        if best_pattern is None:
            # 检查是否是简单的箱体整理
            seg = df_slice
            resistance = seg["high"].max()
            support = seg["low"].min()
            if support > 0:
                amplitude = (resistance - support) / support
                res_touches = (seg["high"] >= resistance * 0.97).sum()
                sup_touches = (seg["low"] <= support * 1.03).sum()
                if 0.05 < amplitude < 0.35 and res_touches >= 2 and sup_touches >= 2:
                    from pattern_recognizer import compute_quality_score
                    metrics = {"touches": min(res_touches, sup_touches), "flatness": 0.5,
                               "amplitude": amplitude, "duration": len(seg)}
                    quality = compute_quality_score("箱体", metrics)
                    best_pattern = {
                        "pattern": "箱体", "neckline": resistance,
                        "quality": quality, "metrics": metrics
                    }

        if best_pattern is None:
            continue
        if C.USE_PATTERN_QUALITY_SCORE and best_pattern["quality"] < C.PATTERN_QUALITY_THRESHOLD:
            continue

        actual_neckline = best_pattern["neckline"]

        # 3. 找突破日
        for i in range(breakout_start, min(breakout_start + 120, n - 1)):
            is_bo, bo_price, vol_ratio = check_breakout_at_idx(df, i, actual_neckline)
            if not is_bo:
                continue

            if not check_market_filter(index_df, df.iloc[i]["date"]):
                break

            entry_idx = i
            if C.USE_FALSE_BREAKOUT_FILTER:
                if not check_false_breakout(df, i, actual_neckline, days=3):
                    continue
                entry_idx = min(i + 3, n - 1)

            if not check_multi_timeframe(df, df.iloc[i]["date"]):
                continue

            atr_val = df.iloc[i].get("atr14", 0)
            if pd.isna(atr_val):
                atr_val = 0

            signals.append({
                "breakout_idx": i,
                "entry_idx": entry_idx,
                "date": df.iloc[entry_idx]["date"],
                "pattern": best_pattern["pattern"],
                "neckline": actual_neckline,
                "entry_price": df.iloc[entry_idx]["open"],
                "vol_ratio": vol_ratio,
                "quality": best_pattern["quality"],
                "atr": atr_val
            })
            break  # 每个阻力位只取第一个突破

    # 去重
    if signals:
        sig_df = pd.DataFrame(signals)
        sig_df = sig_df.sort_values("quality", ascending=False).drop_duplicates("entry_idx")
        sig_df = sig_df.sort_values("entry_idx")
        signals = sig_df.to_dict("records")

    return signals


if __name__ == "__main__":
    from data_fetcher import get_stock_daily, compute_technical_indicators, get_index_daily
    import time
    df = get_stock_daily("000001", "20200101", "20240601")
    if not df.empty:
        df = compute_technical_indicators(df)
        index_df = get_index_daily()
        t0 = time.time()
        signals = scan_breakout_signals(df, index_df)
        print(f"扫描耗时: {time.time()-t0:.2f}秒")
        print(f"信号数: {len(signals)}")
        for s in signals:
            print(f"  {s['date'].strftime('%Y-%m-%d')} {s['pattern']} "
                  f"入场={s['entry_price']:.2f} 量比={s['vol_ratio']:.2f} 质量={s['quality']:.1f}")
