# -*- coding: utf-8 -*-
"""
回测引擎
- 多股票组合回测
- 交易成本：佣金+印花税+滑点
- 涨跌停处理：涨停买不进，跌停卖不出
- 仓位管理：最多10只，单只10%
- 绩效统计：胜率、收益率、夏普、最大回撤、盈亏比
"""
import numpy as np
import pandas as pd
import config as C
from strategy import TrendLineTracker, scan_breakout_signals


class Position:
    """持仓对象"""
    def __init__(self, code, entry_date, entry_price, shares, atr, pattern):
        self.code = code
        self.entry_date = entry_date
        self.entry_price = entry_price
        self.shares = shares
        self.atr = atr
        self.pattern = pattern
        self.tracker = None
        self.highest_price = entry_price
        self.entry_idx = None

    def update(self, df, current_idx):
        """更新持仓状态"""
        if self.tracker is None:
            self.tracker = TrendLineTracker(self.entry_idx, self.entry_price, self.atr)
        self.tracker.update(df, current_idx)
        self.highest_price = max(self.highest_price, df.iloc[current_idx]["high"])

    def check_exit(self, df, current_idx):
        return self.tracker.check_exit(df, current_idx)


class BacktestEngine:
    def __init__(self, initial_capital=C.INITIAL_CAPITAL):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions = {}  # code -> Position
        self.trades = []     # 交易记录
        self.equity_curve = []  # 净值曲线
        self.daily_returns = []

    def _calc_buy_cost(self, price, shares):
        """计算买入成本"""
        amount = price * shares
        commission = max(amount * C.COMMISSION_RATE, C.MIN_COMMISSION)
        slippage = price * C.SLIPPAGE_RATE * shares
        return amount + commission + slippage

    def _calc_sell_proceeds(self, price, shares):
        """计算卖出收入"""
        amount = price * shares
        commission = max(amount * C.COMMISSION_RATE, C.MIN_COMMISSION)
        stamp_tax = amount * C.STAMP_TAX_RATE
        slippage = price * C.SLIPPAGE_RATE * shares
        return amount - commission - stamp_tax - slippage

    def _is_limit_up(self, df, idx):
        """是否涨停（无法买入）"""
        if idx < 1:
            return False
        prev_close = df.iloc[idx - 1]["close"]
        limit_price = prev_close * 1.10  # 主板10%
        # 创业板/科创板20%，简化处理
        return df.iloc[idx]["high"] >= limit_price * 0.999

    def _is_limit_down(self, df, idx):
        """是否跌停（无法卖出）"""
        if idx < 1:
            return False
        prev_close = df.iloc[idx - 1]["close"]
        limit_price = prev_close * 0.90
        return df.iloc[idx]["low"] <= limit_price * 1.001

    def buy(self, code, df, idx, signal):
        """执行买入"""
        if code in self.positions:
            return False
        if len(self.positions) >= C.MAX_POSITIONS:
            return False
        if self._is_limit_up(df, idx):
            return False

        price = df.iloc[idx]["open"]
        if price <= 0:
            return False

        # 仓位计算
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
        pos = Position(code, df.iloc[idx]["date"], price, shares, atr, signal["pattern"])
        pos.entry_idx = idx
        self.positions[code] = pos
        return True

    def sell(self, code, df, idx, reason=""):
        """执行卖出"""
        if code not in self.positions:
            return False
        if self._is_limit_down(df, idx):
            return False  # 跌停卖不出，次日再试

        pos = self.positions[code]
        price = df.iloc[idx]["open"]
        # 如果开盘价异常，用收盘价
        if price <= 0:
            price = df.iloc[idx]["close"]

        proceeds = self._calc_sell_proceeds(price, pos.shares)
        self.cash += proceeds

        pnl = proceeds - self._calc_buy_cost(pos.entry_price, pos.shares)
        pnl_pct = pnl / self._calc_buy_cost(pos.entry_price, pos.shares)

        self.trades.append({
            "code": code,
            "pattern": pos.pattern,
            "entry_date": pos.entry_date,
            "exit_date": df.iloc[idx]["date"],
            "entry_price": pos.entry_price,
            "exit_price": price,
            "shares": pos.shares,
            "pnl": pnl,
            "pnl_pct": pnl_pct,
            "hold_days": idx - pos.entry_idx,
            "exit_reason": reason,
            "max_profit": (pos.highest_price / pos.entry_price - 1)
        })

        del self.positions[code]
        return True

    def run(self, stock_data_dict, index_df=None):
        """
        运行回测
        stock_data_dict: {code: df} 每只股票的日线数据（含技术指标）
        """
        print("  预扫描所有股票的突破信号...")
        all_signals = {}  # code -> [signals]
        for code, df in stock_data_dict.items():
            if df is None or df.empty or len(df) < 150:
                continue
            signals = scan_breakout_signals(df, index_df, start_idx=120)
            if signals:
                all_signals[code] = signals

        print(f"  共 {len(all_signals)} 只股票有信号，"
              f"总计 {sum(len(s) for s in all_signals.values())} 个信号")

        # 获取所有交易日期
        all_dates = set()
        for df in stock_data_dict.values():
            if df is not None and not df.empty:
                all_dates.update(df["date"].tolist())
        all_dates = sorted(all_dates)

        # 建立日期->索引映射
        date_to_idx = {}
        for code, df in stock_data_dict.items():
            if df is not None and not df.empty:
                date_to_idx[code] = {d: i for i, d in enumerate(df["date"].tolist())}

        print("  开始逐日回测...")
        signal_index = {code: 0 for code in all_signals}

        for date in all_dates:
            # 1. 先处理卖出（检查持仓是否触发出场）
            codes_to_sell = []
            for code, pos in self.positions.items():
                df = stock_data_dict.get(code)
                if df is None or date not in date_to_idx.get(code, {}):
                    continue
                idx = date_to_idx[code][date]
                if idx <= pos.entry_idx:
                    continue
                pos.update(df, idx)
                should_exit, exit_price, reason = pos.check_exit(df, idx)
                if should_exit:
                    codes_to_sell.append((code, idx, reason))

            for code, idx, reason in codes_to_sell:
                self.sell(code, stock_data_dict[code], idx, reason)

            # 2. 再处理买入（新信号）
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

            # 3. 记录净值
            total_value = self.cash
            for code, pos in self.positions.items():
                df = stock_data_dict.get(code)
                if df is None or date not in date_to_idx.get(code, {}):
                    total_value += pos.entry_price * pos.shares
                else:
                    idx = date_to_idx[code][date]
                    total_value += df.iloc[idx]["close"] * pos.shares

            self.equity_curve.append({"date": date, "equity": total_value})

        # 强制平仓（回测结束）
        for code in list(self.positions.keys()):
            df = stock_data_dict[code]
            if df is not None and not df.empty:
                self.sell(code, df, len(df) - 1, "回测结束平仓")

        return self._compute_metrics()

    def _compute_metrics(self):
        """计算绩效指标"""
        if not self.trades:
            return self._empty_metrics()

        equity_df = pd.DataFrame(self.equity_curve)
        equity_df["date"] = pd.to_datetime(equity_df["date"])
        equity_df = equity_df.sort_values("date").reset_index(drop=True)
        equity_df["returns"] = equity_df["equity"].pct_change().fillna(0)

        trades_df = pd.DataFrame(self.trades)
        wins = trades_df[trades_df["pnl"] > 0]
        losses = trades_df[trades_df["pnl"] <= 0]

        win_rate = len(wins) / len(trades_df) if len(trades_df) > 0 else 0
        avg_win = wins["pnl_pct"].mean() if len(wins) > 0 else 0
        avg_loss = losses["pnl_pct"].mean() if len(losses) > 0 else 0
        profit_factor = abs(wins["pnl"].sum() / losses["pnl"].sum()) if len(losses) > 0 and losses["pnl"].sum() != 0 else float("inf")

        total_return = equity_df["equity"].iloc[-1] / self.initial_capital - 1
        annual_return = (1 + total_return) ** (252 / len(equity_df)) - 1 if len(equity_df) > 0 else 0

        # 夏普比率（无风险利率3%）
        daily_rf = 0.03 / 252
        excess_returns = equity_df["returns"] - daily_rf
        sharpe = np.sqrt(252) * excess_returns.mean() / excess_returns.std() if excess_returns.std() > 0 else 0

        # 最大回撤
        equity_df["peak"] = equity_df["equity"].cummax()
        equity_df["drawdown"] = (equity_df["equity"] - equity_df["peak"]) / equity_df["peak"]
        max_drawdown = equity_df["drawdown"].min()

        # Sortino
        downside = equity_df["returns"][equity_df["returns"] < 0]
        sortino = np.sqrt(252) * excess_returns.mean() / downside.std() if len(downside) > 0 and downside.std() > 0 else 0

        # Calmar
        calmar = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0

        # 按形态统计
        pattern_stats = {}
        for pat in trades_df["pattern"].unique():
            pat_trades = trades_df[trades_df["pattern"] == pat]
            pat_wins = pat_trades[pat_trades["pnl"] > 0]
            pattern_stats[pat] = {
                "count": len(pat_trades),
                "win_rate": len(pat_wins) / len(pat_trades),
                "avg_return": pat_trades["pnl_pct"].mean(),
                "avg_hold_days": pat_trades["hold_days"].mean()
            }

        # 按出场原因统计
        exit_stats = trades_df["exit_reason"].value_counts().to_dict()

        return {
            "total_trades": len(trades_df),
            "win_rate": win_rate,
            "avg_win_pct": avg_win,
            "avg_loss_pct": avg_loss,
            "profit_factor": profit_factor,
            "total_return": total_return,
            "annual_return": annual_return,
            "sharpe": sharpe,
            "sortino": sortino,
            "calmar": calmar,
            "max_drawdown": max_drawdown,
            "avg_hold_days": trades_df["hold_days"].mean(),
            "pattern_stats": pattern_stats,
            "exit_stats": exit_stats,
            "trades_df": trades_df,
            "equity_df": equity_df
        }

    def _empty_metrics(self):
        return {
            "total_trades": 0, "win_rate": 0, "avg_win_pct": 0,
            "avg_loss_pct": 0, "profit_factor": 0, "total_return": 0,
            "annual_return": 0, "sharpe": 0, "sortino": 0, "calmar": 0,
            "max_drawdown": 0, "avg_hold_days": 0,
            "pattern_stats": {}, "exit_stats": {},
            "trades_df": pd.DataFrame(), "equity_df": pd.DataFrame()
        }


def print_metrics(metrics):
    """打印绩效指标"""
    print("\n" + "=" * 60)
    print("  回测绩效报告")
    print("=" * 60)
    print(f"  总交易次数:     {metrics['total_trades']}")
    print(f"  胜率:           {metrics['win_rate']:.2%}")
    print(f"  平均盈利:       {metrics['avg_win_pct']:.2%}")
    print(f"  平均亏损:       {metrics['avg_loss_pct']:.2%}")
    print(f"  盈亏比:         {metrics['profit_factor']:.2f}")
    print(f"  总收益率:       {metrics['total_return']:.2%}")
    print(f"  年化收益率:     {metrics['annual_return']:.2%}")
    print(f"  夏普比率:       {metrics['sharpe']:.2f}")
    print(f"  Sortino:        {metrics['sortino']:.2f}")
    print(f"  Calmar:         {metrics['calmar']:.2f}")
    print(f"  最大回撤:       {metrics['max_drawdown']:.2%}")
    print(f"  平均持仓天数:   {metrics['avg_hold_days']:.1f}")
    print("-" * 60)
    print("  各形态表现:")
    for pat, stat in metrics["pattern_stats"].items():
        print(f"    {pat}: {stat['count']}次, 胜率{stat['win_rate']:.1%}, "
              f"均收益{stat['avg_return']:.2%}, 持仓{stat['avg_hold_days']:.0f}天")
    print("-" * 60)
    print("  出场原因分布:")
    for reason, cnt in metrics["exit_stats"].items():
        print(f"    {reason}: {cnt}次")
    print("=" * 60)
