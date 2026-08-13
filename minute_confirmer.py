# -*- coding: utf-8 -*-
"""
60分钟级别二次确认模块
日线突破后，用60分钟K线做二次确认，过滤假突破
"""
import warnings
warnings.filterwarnings("ignore")
import os
import numpy as np
import pandas as pd
import akshare as ak
from datetime import datetime, timedelta


class MinuteLevelConfirmer:
    """60分钟级别确认器"""

    def __init__(self, cache_dir="data_cache/minute"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def get_60min_data(self, code, start_date, end_date):
        """
        获取60分钟K线数据
        code: 股票代码（如000001）
        start_date, end_date: 'YYYYMMDD'格式
        返回: DataFrame [datetime, open, high, low, close, volume]
        """
        cache_file = os.path.join(self.cache_dir, f"{code}_60min_{start_date}_{end_date}.parquet")

        # 尝试从缓存读取
        if os.path.exists(cache_file):
            try:
                df = pd.read_parquet(cache_file)
                return df
            except Exception:
                pass

        # 从东方财富获取
        try:
            # akshare的分钟数据接口
            df = ak.stock_zh_a_hist_min_em(
                symbol=code,
                period="60",
                start_date=start_date,
                end_date=end_date,
                adjust="qfq"
            )
            if df is None or df.empty:
                return pd.DataFrame()

            # 标准化列名
            col_map = {}
            for col in df.columns:
                col_lower = col.lower()
                if "时间" in col or "datetime" in col_lower or "date" in col_lower:
                    col_map[col] = "datetime"
                elif "开盘" in col or "open" in col_lower:
                    col_map[col] = "open"
                elif "最高" in col or "high" in col_lower:
                    col_map[col] = "high"
                elif "最低" in col or "low" in col_lower:
                    col_map[col] = "low"
                elif "收盘" in col or "close" in col_lower:
                    col_map[col] = "close"
                elif "成交量" in col or "volume" in col_lower:
                    col_map[col] = "volume"
            df = df.rename(columns=col_map)

            needed = ["datetime", "open", "high", "low", "close", "volume"]
            df = df[[c for c in needed if c in df.columns]]
            df["datetime"] = pd.to_datetime(df["datetime"])
            df = df.sort_values("datetime").reset_index(drop=True)

            # 缓存
            try:
                df.to_parquet(cache_file, index=False)
            except Exception:
                pass

            return df

        except Exception as e:
            # 备用：新浪接口
            try:
                df = ak.stock_zh_a_minute(symbol=f"sh{code}" if code.startswith("6") else f"sz{code}", period="60")
                if df is None or df.empty:
                    return pd.DataFrame()
                df.columns = [c.lower() for c in df.columns]
                if "day" in df.columns:
                    df = df.rename(columns={"day": "datetime"})
                df["datetime"] = pd.to_datetime(df["datetime"])
                for col in ["open", "high", "low", "close", "volume"]:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors="coerce")
                df = df.sort_values("datetime").reset_index(drop=True)
                try:
                    df.to_parquet(cache_file, index=False)
                except Exception:
                    pass
                return df
            except Exception as e2:
                print(f"    获取{code} 60分钟数据失败: {e2}")
                return pd.DataFrame()

    def confirm_breakout(self, code, breakout_date, neckline, daily_df=None):
        """
        60分钟级别确认突破
        code: 股票代码
        breakout_date: 日线突破日期 (Timestamp or 'YYYY-MM-DD')
        neckline: 颈线价格
        返回: dict {confirmed: bool, reason: str, details: dict}
        """
        breakout_date = pd.Timestamp(breakout_date)

        # 获取突破日前后的60分钟数据
        start = (breakout_date - timedelta(days=10)).strftime("%Y%m%d")
        end = (breakout_date + timedelta(days=5)).strftime("%Y%m%d")

        df_60 = self.get_60min_data(code, start, end)
        if df_60.empty:
            return {"confirmed": True, "reason": "无60分钟数据，默认确认", "details": {}}

        # 筛选突破日及之后的60分钟K线
        breakout_day_str = breakout_date.strftime("%Y-%m-%d")
        df_breakout = df_60[df_60["datetime"].dt.strftime("%Y-%m-%d") >= breakout_day_str]

        if df_breakout.empty:
            return {"confirmed": True, "reason": "突破日无60分钟数据，默认确认", "details": {}}

        details = {}
        scores = []

        # === 确认条件1：60分钟收盘价站上颈线 ===
        # 取突破日的最后一根60分钟K线
        breakout_day_bars = df_60[df_60["datetime"].dt.strftime("%Y-%m-%d") == breakout_day_str]
        if not breakout_day_bars.empty:
            last_bar = breakout_day_bars.iloc[-1]
            close_60 = last_bar["close"]
            details["60min_close"] = close_60
            details["60min_close_vs_neckline"] = (close_60 - neckline) / neckline

            if close_60 > neckline * 1.005:  # 60分钟收盘站上颈线0.5%以上
                scores.append(("60min_close_above", 1.0))
            elif close_60 > neckline:
                scores.append(("60min_close_above", 0.5))
            else:
                scores.append(("60min_close_above", -1.0))

            # === 确认条件2：60分钟量能确认 ===
            if len(breakout_day_bars) >= 2:
                breakout_vol = breakout_day_bars["volume"].mean()
                # 前5根60分钟K线的平均量
                prev_bars = df_60[df_60["datetime"] < breakout_day_bars.iloc[0]["datetime"]].tail(10)
                if not prev_bars.empty:
                    prev_vol = prev_bars["volume"].mean()
                    vol_ratio_60 = breakout_vol / prev_vol if prev_vol > 0 else 1
                    details["60min_vol_ratio"] = vol_ratio_60
                    if vol_ratio_60 > 1.5:
                        scores.append(("60min_volume", 1.0))
                    elif vol_ratio_60 > 1.0:
                        scores.append(("60min_volume", 0.5))
                    else:
                        scores.append(("60min_volume", -0.5))

        # === 确认条件3：60分钟趋势（短期均线多头） ===
        if len(df_breakout) >= 5:
            recent = df_breakout.head(8)
            if len(recent) >= 5:
                ma5 = recent["close"].tail(5).mean()
                ma10 = recent["close"].tail(10).mean() if len(recent) >= 10 else recent["close"].mean()
                details["60min_ma5"] = ma5
                details["60min_ma10"] = ma10
                if ma5 > ma10:
                    scores.append(("60min_trend", 0.5))
                else:
                    scores.append(("60min_trend", -0.5))

        # === 确认条件4：突破后不快速跌回颈线下 ===
        # 看突破日后3根60分钟K线
        after_breakout = df_breakout.head(4)
        if len(after_breakout) >= 3:
            min_close_after = after_breakout["close"].min()
            details["min_close_after_breakout"] = min_close_after
            if min_close_after > neckline * 0.99:
                scores.append(("no_fallback", 1.0))
            elif min_close_after > neckline * 0.97:
                scores.append(("no_fallback", 0.0))
            else:
                scores.append(("no_fallback", -1.0))

        # 综合评分
        total_score = sum(s[1] for s in scores)
        details["confirmation_score"] = total_score
        details["sub_scores"] = {name: score for name, score in scores}

        # 确认阈值：总分 >= 0.5 视为确认
        confirmed = total_score >= 0.5
        reason = f"60分钟确认得分{total_score:.1f}" + ("（确认）" if confirmed else "（不确认）")

        return {"confirmed": confirmed, "reason": reason, "details": details}


if __name__ == "__main__":
    # 测试
    confirmer = MinuteLevelConfirmer()

    # 测试获取数据
    print("测试获取平安银行(000001) 60分钟数据...")
    df = confirmer.get_60min_data("000001", "20240101", "20240301")
    if not df.empty:
        print(f"  获取到 {len(df)} 根60分钟K线")
        print(df.head())
    else:
        print("  获取失败")
