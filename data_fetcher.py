# -*- coding: utf-8 -*-
"""
数据获取模块 - 多源容错（东方财富/新浪/腾讯）
"""
import akshare as ak
import pandas as pd
import numpy as np
import time
import os
import warnings
warnings.filterwarnings("ignore")

CACHE_DIR = os.path.join(os.path.dirname(__file__), "data_cache")
os.makedirs(CACHE_DIR, exist_ok=True)


def _normalize_stock_df(df, source="em"):
    """标准化不同数据源的列名"""
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy()
    # 统一列名为小写
    df.columns = [str(c).strip().lower() for c in df.columns]

    # 东方财富格式
    col_map_em = {
        "日期": "date", "开盘": "open", "收盘": "close", "最高": "high",
        "最低": "low", "成交量": "volume", "成交额": "amount",
        "振幅": "amplitude", "涨跌幅": "pct_chg", "涨跌额": "change",
        "换手率": "turnover"
    }
    # 新浪格式
    col_map_sina = {
        "date": "date", "open": "open", "high": "high", "low": "low",
        "close": "close", "volume": "volume", "outstanding_share": "outstanding",
        "turnover": "turnover"
    }

    # 尝试映射
    for old, new in {**col_map_em, **col_map_sina}.items():
        if old in df.columns and new not in df.columns:
            df = df.rename(columns={old: new})

    # 确保核心列存在
    required = ["date", "open", "high", "low", "close", "volume"]
    for col in required:
        if col not in df.columns:
            return pd.DataFrame()

    # 补充缺失列
    if "amount" not in df.columns:
        df["amount"] = df["close"] * df["volume"]
    if "turnover" not in df.columns:
        df["turnover"] = np.nan

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df = df[["date", "open", "high", "low", "close", "volume", "amount", "turnover"]]

    # 类型转换
    for col in ["open", "high", "low", "close", "volume", "amount"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["close"])

    return df


def get_stock_list():
    """获取A股全部股票列表"""
    cache_file = os.path.join(CACHE_DIR, "stock_list.csv")
    if os.path.exists(cache_file):
        mtime = os.path.getmtime(cache_file)
        if time.time() - mtime < 86400:
            return pd.read_csv(cache_file, dtype={"code": str})

    for attempt in range(3):
        try:
            df = ak.stock_info_a_code_name()
            df.columns = [c.lower() for c in df.columns]
            df = df[~df["name"].str.contains("ST|退|N|C", na=False)]
            df = df[~df["code"].str.startswith(("8", "4"))]
            df = df.reset_index(drop=True)
            df.to_csv(cache_file, index=False)
            return df
        except Exception as e:
            if attempt < 2:
                time.sleep(2)
            else:
                print(f"  [WARN] 获取股票列表失败: {e}")
    return pd.DataFrame(columns=["code", "name"])


def get_stock_daily_em(code, start_date, end_date, adjust="qfq"):
    """东方财富数据源"""
    try:
        df = ak.stock_zh_a_hist(
            symbol=code, period="daily",
            start_date=start_date, end_date=end_date, adjust=adjust
        )
        return _normalize_stock_df(df, "em")
    except Exception:
        return pd.DataFrame()


def get_stock_daily_sina(code, start_date, end_date, adjust="qfq"):
    """新浪数据源"""
    try:
        # 新浪需要带市场前缀
        prefix = "sh" if code.startswith(("6", "9")) else "sz"
        symbol = f"{prefix}{code}"
        df = ak.stock_zh_a_daily(
            symbol=symbol, start_date=start_date, end_date=end_date, adjust=adjust
        )
        return _normalize_stock_df(df, "sina")
    except Exception:
        return pd.DataFrame()


def get_stock_daily(code, start_date, end_date, adjust="qfq"):
    """
    获取单只股票日线数据（多源容错）
    优先级：缓存 > 东方财富 > 新浪
    """
    cache_file = os.path.join(CACHE_DIR, f"{code}_{start_date}_{end_date}_{adjust}.parquet")
    if os.path.exists(cache_file):
        try:
            return pd.read_parquet(cache_file)
        except Exception:
            pass

    # 尝试东方财富
    df = get_stock_daily_em(code, start_date, end_date, adjust)
    if df is None or df.empty:
        # 重试一次
        time.sleep(1)
        df = get_stock_daily_em(code, start_date, end_date, adjust)

    # 尝试新浪
    if df is None or df.empty:
        time.sleep(0.5)
        df = get_stock_daily_sina(code, start_date, end_date, adjust)

    if df is None or df.empty:
        print(f"  [WARN] {code} 所有数据源均失败")
        return pd.DataFrame()

    try:
        df.to_parquet(cache_file, index=False)
    except Exception:
        pass
    return df


def get_index_daily(symbol="sh000001", start_date="20180101", end_date="20260801"):
    """获取指数日线数据"""
    cache_file = os.path.join(CACHE_DIR, f"index_{symbol}_{start_date}_{end_date}.parquet")
    if os.path.exists(cache_file):
        try:
            return pd.read_parquet(cache_file)
        except Exception:
            pass

    for attempt in range(3):
        try:
            df = ak.stock_zh_index_daily(symbol=symbol)
            if df is None or df.empty:
                return pd.DataFrame()
            df["date"] = pd.to_datetime(df["date"])
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            df = df[(df["date"] >= start_dt) & (df["date"] <= end_dt)]
            df = df.sort_values("date").reset_index(drop=True)
            df.to_parquet(cache_file, index=False)
            return df
        except Exception as e:
            if attempt < 2:
                time.sleep(2)
            else:
                print(f"  [WARN] 获取指数{symbol}失败: {e}")
    return pd.DataFrame()


def compute_technical_indicators(df):
    """计算技术指标"""
    if df is None or df.empty:
        return df
    df = df.copy()
    for p in [5, 10, 20, 60, 120]:
        df[f"ma{p}"] = df["close"].rolling(p).mean()

    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift(1)).abs()
    low_close = (df["low"] - df["close"].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["atr14"] = tr.rolling(14).mean()

    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["rsi14"] = 100 - (100 / (1 + rs))

    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()
    df["macd_dif"] = ema12 - ema26
    df["macd_dea"] = df["macd_dif"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = 2 * (df["macd_dif"] - df["macd_dea"])

    df["vol_ratio_5"] = df["volume"] / df["volume"].rolling(5).mean()
    df["vol_ma3"] = df["volume"].rolling(3).mean().shift(1)
    return df


def get_weekly_data(df_daily):
    """日线转周线"""
    if df_daily is None or df_daily.empty:
        return pd.DataFrame()
    df_w = df_daily.set_index("date").resample("W").agg({
        "open": "first", "high": "max", "low": "min",
        "close": "last", "volume": "sum", "amount": "sum"
    }).dropna().reset_index()
    return df_w
