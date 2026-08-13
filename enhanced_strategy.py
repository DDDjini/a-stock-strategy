# -*- coding: utf-8 -*-
"""
高胜率增强模块
探索将胜率从~53%提升到70%+的方法：
1. 分批止盈（+5%卖半仓，剩余趋势跟踪）
2. 多因子入场过滤（RSI区间、MACD金叉、量能结构）
3. 市场状态自适应（牛市放宽/熊市收紧）
4. 形态质量动态阈值
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import config as C
from strategy import TrendLineTracker, check_market_filter, check_false_breakout
from pattern_recognizer import find_resistance_levels, detect_all_patterns, compute_quality_score
from backtest_engine import BacktestEngine


class EnhancedPosition:
    """增强持仓：支持分批止盈"""
    def __init__(self, code, entry_date, entry_price, total_shares, atr, pattern):
        self.code = code
        self.entry_date = entry_date
        self.entry_price = entry_price
        self.total_shares = total_shares
        self.remaining_shares = total_shares
        self.atr = atr
        self.pattern = pattern
        self.tracker = None
        self.entry_idx = None
        self.highest_price = entry_price
        self.first_target_hit = False  # +5%止盈是否触发
        self.realized_pnl = 0  # 已实现盈亏

    def update(self, df, current_idx):
        if self.tracker is None:
            self.tracker = TrendLineTracker(self.entry_idx, self.entry_price, self.atr)
        self.tracker.update(df, current_idx)
        self.highest_price = max(self.highest_price, df.iloc[current_idx]["high"])

    def check_partial_take_profit(self, df, current_idx):
        """检查是否触发第一目标位（+5%），返回卖出信号（不修改持仓状态）"""
        if self.first_target_hit:
            return None
        row = df.iloc[current_idx]
        if row["high"] >= self.entry_price * 1.05:
            self.first_target_hit = True
            sell_shares = self.total_shares // 2
            sell_price = self.entry_price * 1.05
            # 只返回信号，实际扣减由sell()处理，避免重复计算
            return {"shares": sell_shares, "price": sell_price, "reason": "第一目标+5%"}
        return None

    def check_exit(self, df, current_idx):
        return self.tracker.check_exit(df, current_idx)


def scan_enhanced_signals(df, index_df=None, start_idx=120):
    """
    增强版信号扫描：加入RSI、MACD、量能结构过滤
    """
    if df is None or df.empty or len(df) < start_idx + 20:
        return []

    signals = []
    n = len(df)
    resistance_levels = find_resistance_levels(df, tol=0.04, min_touches=2)

    for rl_price, rl_first, rl_last, rl_touches in resistance_levels:
        breakout_start = max(rl_last + 1, start_idx)
        if breakout_start >= n - 5:
            continue

        lookback_end = rl_last + 1
        lookback_start = max(0, lookback_end - C.PATTERN_LOOKBACK)
        df_slice = df.iloc[lookback_start:lookback_end].reset_index(drop=True)
        if len(df_slice) < C.MIN_PATTERN_DAYS:
            continue

        patterns = detect_all_patterns(df_slice, len(df_slice))
        best_pattern = patterns[0] if patterns else None

        if best_pattern is None:
            seg = df_slice
            resistance = seg["high"].max()
            support = seg["low"].min()
            if support > 0:
                amplitude = (resistance - support) / support
                res_t = (seg["high"] >= resistance * 0.97).sum()
                sup_t = (seg["low"] <= support * 1.03).sum()
                if 0.05 < amplitude < 0.35 and res_t >= 2 and sup_t >= 2:
                    metrics = {"touches": min(res_t, sup_t), "flatness": 0.5,
                               "amplitude": amplitude, "duration": len(seg)}
                    quality = compute_quality_score("箱体", metrics)
                    best_pattern = {"pattern": "箱体", "neckline": resistance, "quality": quality}

        if best_pattern is None or best_pattern["quality"] < C.PATTERN_QUALITY_THRESHOLD:
            continue

        neckline = best_pattern["neckline"]

        for i in range(breakout_start, min(breakout_start + 120, n - 1)):
            row = df.iloc[i]
            # 突破
            if row["close"] <= neckline * (1 + C.BREAKOUT_CLOSE_BUFFER):
                continue
            if df.iloc[i - 1]["close"] > neckline:
                continue
            # 量能
            vol_ma3 = df.iloc[i - 3:i]["volume"].mean()
            if vol_ma3 <= 0:
                continue
            vol_ratio = row["volume"] / vol_ma3
            if vol_ratio < C.BREAKOUT_VOL_RATIO:
                continue

            # 增强过滤1: RSI在50-75之间（有动量但未超买）
            rsi = row.get("rsi14", 50)
            if pd.isna(rsi) or rsi < 45 or rsi > 80:
                continue

            # 增强过滤2: MACD柱状图为正（多头动能）
            macd_hist = row.get("macd_hist", 0)
            if pd.isna(macd_hist) or macd_hist <= 0:
                continue

            # 增强过滤3: 收盘价在20日均线之上
            ma20 = row.get("ma20", 0)
            if pd.isna(ma20) or row["close"] <= ma20:
                continue

            # 大盘过滤
            if not check_market_filter(index_df, df.iloc[i]["date"]):
                break

            # 假突破
            entry_idx = i
            if not check_false_breakout(df, i, neckline, days=3):
                continue
            entry_idx = min(i + 3, n - 1)

            atr_val = row.get("atr14", 0)
            if pd.isna(atr_val):
                atr_val = 0

            signals.append({
                "breakout_idx": i, "entry_idx": entry_idx,
                "date": df.iloc[entry_idx]["date"],
                "pattern": best_pattern["pattern"],
                "neckline": neckline,
                "entry_price": df.iloc[entry_idx]["open"],
                "vol_ratio": vol_ratio,
                "quality": best_pattern["quality"],
                "atr": atr_val,
                "rsi": rsi, "macd_hist": macd_hist
            })
            break

    if signals:
        sig_df = pd.DataFrame(signals)
        sig_df = sig_df.sort_values("quality", ascending=False).drop_duplicates("entry_idx")
        sig_df = sig_df.sort_values("entry_idx")
        signals = sig_df.to_dict("records")

    return signals


class EnhancedBacktestEngine(BacktestEngine):
    """增强回测引擎：支持分批止盈"""

    def __init__(self, initial_capital=C.INITIAL_CAPITAL, use_partial_tp=True):
        super().__init__(initial_capital)
        self.use_partial_tp = use_partial_tp
        self.positions = {}  # 覆盖为EnhancedPosition

    def buy(self, code, df, idx, signal):
        if code in self.positions or len(self.positions) >= C.MAX_POSITIONS:
            return False
        if self._is_limit_up(df, idx):
            return False
        price = df.iloc[idx]["open"]
        if price <= 0:
            return False
        atr = signal.get("atr", df.iloc[idx].get("atr14", 0))
        if C.USE_ATR_SIZING and atr > 0:
            risk_amount = self.cash * 0.02
            stop_dist = 2 * atr
            shares = int(risk_amount / stop_dist / 100) * 100
        else:
            shares = int(self.cash * C.SINGLE_POSITION / price / 100) * 100
        shares = max(100, shares)
        cost = self._calc_buy_cost(price, shares)
        if cost > self.cash:
            shares = int(self.cash * 0.95 / price / 100) * 100
            if shares < 100:
                return False
            cost = self._calc_buy_cost(price, shares)
        self.cash -= cost
        pos = EnhancedPosition(code, df.iloc[idx]["date"], price, shares, atr, signal["pattern"])
        pos.entry_idx = idx
        self.positions[code] = pos
        return True

    def sell(self, code, df, idx, reason="", shares=None):
        if code not in self.positions:
            return False
        if self._is_limit_down(df, idx):
            return False
        pos = self.positions[code]
        sell_shares = shares if shares else pos.remaining_shares
        if sell_shares <= 0:
            return False
        price = df.iloc[idx]["open"]
        if price <= 0:
            price = df.iloc[idx]["close"]
        proceeds = self._calc_sell_proceeds(price, sell_shares)
        self.cash += proceeds

        # 计算这部分的盈亏
        cost_basis = self._calc_buy_cost(pos.entry_price, sell_shares)
        pnl = proceeds - cost_basis
        pnl_pct = pnl / cost_basis if cost_basis > 0 else 0

        pos.remaining_shares -= sell_shares
        pos.realized_pnl += pnl

        # 只有清仓时才记录完整交易
        if pos.remaining_shares <= 0:
            total_pnl = pos.realized_pnl
            total_cost = self._calc_buy_cost(pos.entry_price, pos.total_shares)
            total_pnl_pct = total_pnl / total_cost if total_cost > 0 else 0
            self.trades.append({
                "code": code, "pattern": pos.pattern,
                "entry_date": pos.entry_date, "exit_date": df.iloc[idx]["date"],
                "entry_price": pos.entry_price, "exit_price": price,
                "shares": pos.total_shares, "pnl": total_pnl,
                "pnl_pct": total_pnl_pct,
                "hold_days": idx - pos.entry_idx,
                "exit_reason": reason,
                "max_profit": (pos.highest_price / pos.entry_price - 1)
            })
            del self.positions[code]
        return True

    def run(self, stock_data_dict, index_df=None, use_enhanced_signals=True):
        print("  预扫描增强信号...")
        all_signals = {}
        for code, df in stock_data_dict.items():
            if df is None or df.empty or len(df) < 150:
                continue
            if use_enhanced_signals:
                signals = scan_enhanced_signals(df, index_df, start_idx=120)
            else:
                from strategy import scan_breakout_signals
                signals = scan_breakout_signals(df, index_df, start_idx=120)
            if signals:
                all_signals[code] = signals

        print(f"  共 {len(all_signals)} 只股票有信号, "
              f"总计 {sum(len(s) for s in all_signals.values())} 个信号")

        all_dates = set()
        for df in stock_data_dict.values():
            if df is not None and not df.empty:
                all_dates.update(df["date"].tolist())
        all_dates = sorted(all_dates)

        date_to_idx = {}
        for code, df in stock_data_dict.items():
            if df is not None and not df.empty:
                date_to_idx[code] = {d: i for i, d in enumerate(df["date"].tolist())}

        print("  开始逐日回测...")
        signal_index = {code: 0 for code in all_signals}

        for date in all_dates:
            # 卖出检查
            codes_to_sell = []
            partial_sells = []
            for code, pos in self.positions.items():
                df = stock_data_dict.get(code)
                if df is None or date not in date_to_idx.get(code, {}):
                    continue
                idx = date_to_idx[code][date]
                if idx <= pos.entry_idx:
                    continue
                pos.update(df, idx)

                # 分批止盈
                if self.use_partial_tp:
                    partial = pos.check_partial_take_profit(df, idx)
                    if partial:
                        partial_sells.append((code, idx, partial))

                should_exit, exit_price, reason = pos.check_exit(df, idx)
                if should_exit and pos.remaining_shares > 0:
                    codes_to_sell.append((code, idx, reason))

            for code, idx, partial in partial_sells:
                self.sell(code, stock_data_dict[code], idx, partial["reason"], partial["shares"])

            for code, idx, reason in codes_to_sell:
                self.sell(code, stock_data_dict[code], idx, reason)

            # 买入
            for code, signals in all_signals.items():
                if code in self.positions:
                    continue
                si = signal_index[code]
                while si < len(signals):
                    sig = signals[si]
                    if sig["date"] == date:
                        df = stock_data_dict[code]
                        idx = date_to_idx[code].get(date)
                        if idx is not None:
                            self.buy(code, df, idx, sig)
                        si += 1
                        break
                    elif sig["date"] > date:
                        break
                    else:
                        si += 1
                signal_index[code] = si

            # 净值
            total_value = self.cash
            for code, pos in self.positions.items():
                df = stock_data_dict.get(code)
                if df is None or date not in date_to_idx.get(code, {}):
                    total_value += pos.entry_price * pos.remaining_shares
                else:
                    idx = date_to_idx[code][date]
                    total_value += df.iloc[idx]["close"] * pos.remaining_shares
            self.equity_curve.append({"date": date, "equity": total_value})

        # 强制平仓
        for code in list(self.positions.keys()):
            df = stock_data_dict[code]
            if df is not None and not df.empty:
                self.sell(code, df, len(df) - 1, "回测结束平仓")

        return self._compute_metrics()


def compare_strategies(stock_data, index_df):
    """对比基础策略 vs 增强策略"""
    print("\n" + "=" * 70)
    print("  策略对比：基础版 vs 高胜率增强版")
    print("=" * 70)

    # 基础版
    print("\n[基础版]")
    base_engine = BacktestEngine()
    base_metrics = base_engine.run(stock_data, index_df)
    from backtest_engine import print_metrics
    print_metrics(base_metrics)

    # 增强版（多因子过滤+分批止盈）
    print("\n[增强版：RSI+MACD+MA20过滤 + 分批止盈]")
    enh_engine = EnhancedBacktestEngine(use_partial_tp=True)
    enh_metrics = enh_engine.run(stock_data, index_df, use_enhanced_signals=True)
    print_metrics(enh_metrics)

    # 对比表
    print("\n" + "-" * 70)
    print(f"  {'指标':<15} {'基础版':>12} {'增强版':>12} {'变化':>10}")
    print("-" * 70)
    for key, label in [("total_trades", "交易次数"), ("win_rate", "胜率"),
                       ("total_return", "总收益"), ("annual_return", "年化收益"),
                       ("sharpe", "夏普"), ("max_drawdown", "最大回撤"),
                       ("profit_factor", "盈亏比")]:
        base_val = base_metrics[key]
        enh_val = enh_metrics[key]
        if key in ["win_rate", "total_return", "annual_return", "max_drawdown"]:
            base_str = f"{base_val:.2%}"
            enh_str = f"{enh_val:.2%}"
            diff = f"{(enh_val - base_val):.2%}"
        else:
            base_str = f"{base_val:.2f}"
            enh_str = f"{enh_val:.2f}"
            diff = f"{(enh_val - base_val):.2f}"
        print(f"  {label:<15} {base_str:>12} {enh_str:>12} {diff:>10}")

    return base_metrics, enh_metrics
