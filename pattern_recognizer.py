# -*- coding: utf-8 -*-
"""
K线形态识别引擎（优化版）
核心改进：
1. 阻力位驱动的颈线识别（多次触碰的价格水平）
2. 快速矩形检测（限制窗口步长）
3. 更有区分度的质量评分
"""
import numpy as np
import pandas as pd
from scipy.signal import argrelextrema
from scipy.stats import linregress
import config as C


def find_swing_points(df, order=5):
    highs = df["high"].values
    lows = df["low"].values
    max_idx = argrelextrema(highs, np.greater_equal, order=order)[0]
    min_idx = argrelextrema(lows, np.less_equal, order=order)[0]
    return min_idx, max_idx


def find_swing_points_combined(df, order=5):
    min_idx, max_idx = find_swing_points(df, order)
    points = []
    for i in min_idx:
        points.append({"idx": int(i), "type": "low", "price": float(df["low"].iloc[i]),
                       "date": df["date"].iloc[i]})
    for i in max_idx:
        points.append({"idx": int(i), "type": "high", "price": float(df["high"].iloc[i]),
                       "date": df["date"].iloc[i]})
    points.sort(key=lambda x: x["idx"])
    filtered = []
    for p in points:
        if filtered and filtered[-1]["type"] == p["type"]:
            if p["type"] == "low" and p["price"] < filtered[-1]["price"]:
                filtered[-1] = p
            elif p["type"] == "high" and p["price"] > filtered[-1]["price"]:
                filtered[-1] = p
        else:
            filtered.append(p)
    return filtered


def find_resistance_levels(df, tol=0.04, min_touches=2):
    """
    识别阻力位：被多次触碰的价格水平
    改进：用order=3获取更多摆动点，更宽松的聚类
    返回: [(price, first_idx, last_idx, touch_count), ...]
    """
    highs = df["high"].values
    n = len(df)
    if n < 20:
        return []

    levels = []
    # 用摆动高点聚类（order=3获取更多点）
    _, max_idx = find_swing_points(df, order=3)
    if len(max_idx) < min_touches:
        return levels

    high_prices = [(int(i), float(highs[i])) for i in max_idx]
    high_prices.sort(key=lambda x: x[1])

    # 聚类相近价格（更宽松的tol）
    clusters = []
    current_cluster = [high_prices[0]]
    for i in range(1, len(high_prices)):
        if high_prices[i][1] <= current_cluster[-1][1] * (1 + tol):
            current_cluster.append(high_prices[i])
        else:
            if len(current_cluster) >= min_touches:
                clusters.append(current_cluster)
            current_cluster = [high_prices[i]]
    if len(current_cluster) >= min_touches:
        clusters.append(current_cluster)

    for cluster in clusters:
        price = float(np.mean([p[1] for p in cluster]))
        indices = [p[0] for p in cluster]
        levels.append((price, min(indices), max(indices), len(cluster)))

    # 补充：用固定价格区间统计触碰次数（捕捉非摆动高点的阻力位）
    # 每2%一个价格桶，统计每个桶内high的出现次数
    price_min = df["low"].min()
    price_max = df["high"].max()
    if price_min > 0 and price_max > price_min:
        n_bins = int((price_max - price_min) / (price_min * 0.02)) + 1
        bin_edges = np.linspace(price_min, price_max, n_bins + 1)
        for b in range(n_bins):
            mask = (highs >= bin_edges[b]) & (highs < bin_edges[b + 1])
            touch_indices = np.where(mask)[0]
            if len(touch_indices) >= min_touches + 1:  # 更严格
                bin_price = (bin_edges[b] + bin_edges[b + 1]) / 2
                # 检查是否与已有levels重复
                is_dup = any(abs(bin_price - lp[0]) / lp[0] < 0.03 for lp in levels)
                if not is_dup:
                    levels.append((float(bin_price), int(touch_indices[0]),
                                   int(touch_indices[-1]), int(len(touch_indices))))

    return levels


def compute_quality_score(pattern_type, metrics):
    """质量评分（更有区分度，0-100）"""
    score = 40.0  # 基础分降低

    if pattern_type in ["W底", "头肩底"]:
        sym = metrics.get("symmetry", 1.0)
        score += max(0, (1 - sym) * 20)  # 对称性加分降低
        depth = metrics.get("depth", 0)
        score += min(15, depth * 60)
        vol_shrink = metrics.get("vol_shrink", 1.0)
        if vol_shrink < 1:
            score += (1 - vol_shrink) * 10
        # 触碰次数
        touches = metrics.get("neckline_touches", 1)
        score += min(15, (touches - 1) * 5)

    elif pattern_type in ["矩形底", "箱体"]:
        touches = metrics.get("touches", 2)
        score += min(15, (touches - 2) * 4)
        flatness = metrics.get("flatness", 1.0)
        score += max(0, (1 - flatness) * 15)
        amplitude = metrics.get("amplitude", 0)
        if 0.05 < amplitude < 0.25:
            score += 10
        # 形态持续时间
        duration = metrics.get("duration", 20)
        if 20 <= duration <= 60:
            score += 10

    elif pattern_type == "收敛三角":
        touches = metrics.get("touches", 4)
        score += min(12, (touches - 4) * 3)
        converge = metrics.get("convergence", 1.0)
        score += max(0, (1 - converge) * 20)
        width_end = metrics.get("width_end", 0.1)
        if width_end < 0.06:
            score += 8

    elif pattern_type == "菱形":
        score += metrics.get("expand_clarity", 0.5) * 15
        score += metrics.get("contract_clarity", 0.5) * 15

    return min(100, max(0, score))


def detect_double_bottom(df, lookback=C.PATTERN_LOOKBACK):
    if len(df) < C.MIN_PATTERN_DAYS:
        return None
    recent = df.iloc[-lookback:].reset_index(drop=True)
    swings = find_swing_points_combined(recent, order=5)
    lows = [s for s in swings if s["type"] == "low"]
    if len(lows) < 2:
        return None

    results = []
    for i in range(len(lows) - 1):
        low1 = lows[i]
        for j in range(i + 1, min(i + 5, len(lows))):
            low2 = lows[j]
            gap = low2["idx"] - low1["idx"]
            if gap < C.W_BOTTOM_GAP_MIN or gap > C.MAX_PATTERN_DAYS:
                continue
            price_diff = abs(low1["price"] - low2["price"]) / low1["price"]
            if price_diff > C.W_BOTTOM_SYMMETRY:
                continue
            between = recent.iloc[low1["idx"]:low2["idx"] + 1]
            neckline_price = between["high"].max()
            avg_low = (low1["price"] + low2["price"]) / 2
            depth = (neckline_price - avg_low) / neckline_price
            if depth < C.W_BOTTOM_DEPTH_MIN:
                continue

            vol_left = recent.iloc[max(0, low1["idx"] - 3):low1["idx"] + 3]["volume"].mean()
            vol_right = recent.iloc[max(0, low2["idx"] - 3):low2["idx"] + 3]["volume"].mean()
            vol_shrink = vol_right / vol_left if vol_left > 0 else 1.0

            # 颈线触碰次数
            neckline_touches = (between["high"] >= neckline_price * 0.97).sum()

            metrics = {"symmetry": price_diff, "depth": depth, "vol_shrink": vol_shrink,
                       "neckline_touches": neckline_touches}
            quality = compute_quality_score("W底", metrics)
            results.append({
                "pattern": "W底", "neckline": neckline_price,
                "neckline_idx": int(low2["idx"]), "start_idx": int(low1["idx"]),
                "end_idx": int(low2["idx"]), "quality": quality, "metrics": metrics
            })

    return max(results, key=lambda x: x["quality"]) if results else None


def detect_head_shoulders_bottom(df, lookback=C.PATTERN_LOOKBACK):
    if len(df) < C.MIN_PATTERN_DAYS:
        return None
    recent = df.iloc[-lookback:].reset_index(drop=True)
    swings = find_swing_points_combined(recent, order=5)
    lows = [s for s in swings if s["type"] == "low"]
    if len(lows) < 3:
        return None

    results = []
    for i in range(len(lows) - 2):
        left, head, right = lows[i], lows[i + 1], lows[i + 2]
        if head["price"] >= left["price"] or head["price"] >= right["price"]:
            continue
        shoulder_diff = abs(left["price"] - right["price"]) / left["price"]
        if shoulder_diff > C.HSH_SHOULDER_SYMM:
            continue
        avg_shoulder = (left["price"] + right["price"]) / 2
        head_depth = (avg_shoulder - head["price"]) / avg_shoulder
        if head_depth < C.HSH_HEAD_DEPTH_MIN:
            continue

        seg1 = recent.iloc[left["idx"]:head["idx"] + 1]
        seg2 = recent.iloc[head["idx"]:right["idx"] + 1]
        neckline_price = max(seg1["high"].max(), seg2["high"].max())

        vol_left = recent.iloc[max(0, left["idx"] - 3):left["idx"] + 3]["volume"].mean()
        vol_right = recent.iloc[max(0, right["idx"] - 3):right["idx"] + 3]["volume"].mean()
        vol_shrink = vol_right / vol_left if vol_left > 0 else 1.0

        metrics = {"symmetry": shoulder_diff, "depth": head_depth, "vol_shrink": vol_shrink,
                   "neckline_touches": 2}
        quality = compute_quality_score("头肩底", metrics)
        results.append({
            "pattern": "头肩底", "neckline": neckline_price,
            "neckline_idx": int(right["idx"]), "start_idx": int(left["idx"]),
            "end_idx": int(right["idx"]), "quality": quality, "metrics": metrics
        })

    return max(results, key=lambda x: x["quality"]) if results else None


def detect_rectangle(df, lookback=C.PATTERN_LOOKBACK, is_bottom=True):
    """快速矩形/箱体检测"""
    if len(df) < C.MIN_PATTERN_DAYS:
        return None
    recent = df.iloc[-lookback:].reset_index(drop=True)
    n = len(recent)

    best = None
    # 限制窗口数量：步长10天
    for wlen in range(C.MIN_PATTERN_DAYS, min(C.MAX_PATTERN_DAYS, n), 10):
        for start in range(0, n - wlen, 10):
            segment = recent.iloc[start:start + wlen]
            resistance = segment["high"].max()
            support = segment["low"].min()
            if support <= 0:
                continue
            amplitude = (resistance - support) / support
            if amplitude < 0.05 or amplitude > 0.35:
                continue

            tol = C.TOUCH_TOLERANCE
            res_touches = (segment["high"] >= resistance * (1 - tol)).sum()
            sup_touches = (segment["low"] <= support * (1 + tol)).sum()
            if res_touches < 2 or sup_touches < 2:
                continue

            x = np.arange(len(segment))
            slope_high = linregress(x, segment["high"].values).slope / resistance if resistance > 0 else 0
            slope_low = linregress(x, segment["low"].values).slope / support if support > 0 else 0
            flatness = (abs(slope_high) + abs(slope_low)) / 2

            pattern_name = "矩形底" if is_bottom else "箱体"
            metrics = {"touches": min(res_touches, sup_touches), "flatness": flatness,
                       "amplitude": amplitude, "duration": wlen}
            quality = compute_quality_score(pattern_name, metrics)

            result = {"pattern": pattern_name, "neckline": resistance, "support": support,
                      "neckline_idx": int(start + wlen - 1), "start_idx": int(start),
                      "end_idx": int(start + wlen - 1), "quality": quality, "metrics": metrics}
            if best is None or quality > best["quality"]:
                best = result

    return best


def detect_converging_triangle(df, lookback=C.PATTERN_LOOKBACK):
    if len(df) < C.MIN_PATTERN_DAYS:
        return None
    recent = df.iloc[-lookback:].reset_index(drop=True)
    swings = find_swing_points_combined(recent, order=4)
    highs = [s for s in swings if s["type"] == "high"]
    lows = [s for s in swings if s["type"] == "low"]
    if len(highs) < 2 or len(lows) < 2:
        return None

    high_x = np.array([h["idx"] for h in highs])
    high_y = np.array([h["price"] for h in highs])
    low_x = np.array([l["idx"] for l in lows])
    low_y = np.array([l["price"] for l in lows])

    reg_high = linregress(high_x, high_y)
    reg_low = linregress(low_x, low_y)
    if reg_high.slope >= 0 or reg_low.slope <= 0:
        return None

    start_idx = min(high_x[0], low_x[0])
    end_idx = max(high_x[-1], low_x[-1])
    width_start = (reg_high.intercept + reg_high.slope * start_idx) - \
                  (reg_low.intercept + reg_low.slope * start_idx)
    width_end = (reg_high.intercept + reg_high.slope * end_idx) - \
                (reg_low.intercept + reg_low.slope * end_idx)
    if width_start <= 0 or width_end <= 0:
        return None
    convergence = width_end / width_start
    if convergence > C.TRIANGLE_CONVERGE:
        return None

    neckline_price = reg_high.intercept + reg_high.slope * end_idx
    metrics = {"touches": len(highs) + len(lows), "convergence": convergence,
               "width_end": width_end / neckline_price if neckline_price > 0 else 0.1}
    quality = compute_quality_score("收敛三角", metrics)
    return {"pattern": "收敛三角", "neckline": neckline_price,
            "upper_slope": reg_high.slope, "lower_slope": reg_low.slope,
            "neckline_idx": int(end_idx), "start_idx": int(start_idx),
            "end_idx": int(end_idx), "quality": quality, "metrics": metrics}


def detect_diamond(df, lookback=C.PATTERN_LOOKBACK):
    if len(df) < C.MIN_PATTERN_DAYS + 10:
        return None
    recent = df.iloc[-lookback:].reset_index(drop=True)
    n = len(recent)
    best = None
    for split in range(n // 3, 2 * n // 3, 10):
        expand_seg = recent.iloc[:split]
        contract_seg = recent.iloc[split:]
        if len(expand_seg) < 10 or len(contract_seg) < 10:
            continue
        x_exp = np.arange(len(expand_seg))
        slope_eh = linregress(x_exp, expand_seg["high"].values).slope
        slope_el = linregress(x_exp, expand_seg["low"].values).slope
        if slope_eh <= 0 or slope_el >= 0:
            continue
        x_con = np.arange(len(contract_seg))
        slope_ch = linregress(x_con, contract_seg["high"].values).slope
        slope_cl = linregress(x_con, contract_seg["low"].values).slope
        if slope_ch >= 0 or slope_cl <= 0:
            continue
        neckline_price = contract_seg["high"].iloc[-1]
        expand_clarity = min(1.0, abs(slope_eh) / (expand_seg["high"].mean() / 100))
        contract_clarity = min(1.0, abs(slope_ch) / (contract_seg["high"].mean() / 100))
        metrics = {"expand_clarity": expand_clarity, "contract_clarity": contract_clarity}
        quality = compute_quality_score("菱形", metrics)
        result = {"pattern": "菱形", "neckline": neckline_price,
                  "neckline_idx": int(n - 1), "start_idx": 0, "end_idx": int(n - 1),
                  "split_idx": int(split), "quality": quality, "metrics": metrics}
        if best is None or quality > best["quality"]:
            best = result
    return best


def detect_all_patterns(df, lookback=C.PATTERN_LOOKBACK):
    patterns = []
    detectors = [
        lambda d: detect_double_bottom(d, lookback),
        lambda d: detect_head_shoulders_bottom(d, lookback),
        lambda d: detect_rectangle(d, lookback, is_bottom=True),
        lambda d: detect_rectangle(d, lookback, is_bottom=False),
        lambda d: detect_converging_triangle(d, lookback),
        lambda d: detect_diamond(d, lookback),
    ]
    for detector in detectors:
        try:
            result = detector(df)
            if result and result["quality"] >= 20:
                patterns.append(result)
        except Exception:
            continue
    patterns.sort(key=lambda x: x["quality"], reverse=True)
    return patterns


def get_best_pattern(df, lookback=C.PATTERN_LOOKBACK):
    patterns = detect_all_patterns(df, lookback)
    return patterns[0] if patterns else None
