# -*- coding: utf-8 -*-
"""
从已有交易记录快速训练ML形态评分模型
"""
import warnings
warnings.filterwarnings("ignore")
import os
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import joblib

import config as C
from data_fetcher import compute_technical_indicators, get_stock_daily
from pattern_recognizer import find_resistance_levels


def extract_features_for_trade(df, entry_idx, neckline, pattern_type, index_df=None):
    """从一笔交易的入场点提取特征"""
    features = {}
    row = df.iloc[entry_idx]

    # 价格位置
    features["price_vs_ma5"] = (row["close"] - row.get("ma5", row["close"])) / row["close"]
    features["price_vs_ma20"] = (row["close"] - row.get("ma20", row["close"])) / row["close"]
    features["price_vs_ma60"] = (row["close"] - row.get("ma60", row["close"])) / row["close"]
    features["ma5_vs_ma20"] = (row.get("ma5", 0) - row.get("ma20", 0)) / max(row.get("ma20", 1), 0.01)
    features["ma20_vs_ma60"] = (row.get("ma20", 0) - row.get("ma60", 0)) / max(row.get("ma60", 1), 0.01)

    # 突破强度
    features["breakout_strength"] = (row["close"] - neckline) / neckline if neckline > 0 else 0

    # 量能
    vol_ma3 = df.iloc[max(0, entry_idx-3):entry_idx]["volume"].mean()
    features["vol_ratio"] = row["volume"] / vol_ma3 if vol_ma3 > 0 else 1
    features["vol_vs_ma5"] = row["volume"] / row.get("vol_ma5", row["volume"])

    # 动量
    features["rsi_14"] = row.get("rsi_14", 50)
    features["macd_hist"] = row.get("macd_hist", 0)
    features["macd_dif"] = row.get("macd_dif", 0)

    # 波动率
    features["atr_ratio"] = row.get("atr_14", 0) / row["close"] if row["close"] > 0 else 0

    # 近期收益
    if entry_idx >= 20:
        features["ret_20d"] = (row["close"] - df.iloc[entry_idx-20]["close"]) / df.iloc[entry_idx-20]["close"]
        features["ret_5d"] = (row["close"] - df.iloc[entry_idx-5]["close"]) / df.iloc[entry_idx-5]["close"]
    else:
        features["ret_20d"] = 0
        features["ret_5d"] = 0

    # 量能趋势
    if entry_idx >= 10:
        vol_5 = df.iloc[entry_idx-4:entry_idx+1]["volume"].mean()
        vol_20 = df.iloc[max(0,entry_idx-19):entry_idx+1]["volume"].mean()
        features["vol_trend"] = vol_5 / vol_20 if vol_20 > 0 else 1
    else:
        features["vol_trend"] = 1

    # 市场环境
    if index_df is not None and len(index_df) > 0:
        idx_mask = index_df["date"] <= row["date"]
        if idx_mask.any():
            idx_row = index_df[idx_mask].iloc[-1]
            idx_close = idx_row["close"]
            idx_ma60 = index_df[idx_mask]["close"].rolling(60).mean().iloc[-1] if idx_mask.sum() >= 60 else idx_close
            features["market_vs_ma60"] = (idx_close - idx_ma60) / idx_ma60 if idx_ma60 > 0 else 0
            if idx_mask.sum() >= 20:
                features["market_ret_20d"] = (idx_close - index_df[idx_mask].iloc[-20]["close"]) / index_df[idx_mask].iloc[-20]["close"]
            else:
                features["market_ret_20d"] = 0
        else:
            features["market_vs_ma60"] = 0
            features["market_ret_20d"] = 0
    else:
        features["market_vs_ma60"] = 0
        features["market_ret_20d"] = 0

    # 形态类型
    features["pattern_type"] = pattern_type

    return features


def train_from_trades(trades_csv="results/trades_hs300.csv",
                      index_df=None,
                      model_path="results/ml_pattern_model.pkl"):
    """从交易记录训练ML模型"""
    print("加载交易记录...")
    trades = pd.read_csv(trades_csv)
    trades["entry_date"] = pd.to_datetime(trades["entry_date"])
    print(f"  共 {len(trades)} 笔交易")

    print("提取特征...")
    all_features = []
    cache = {}

    for i, trade in trades.iterrows():
        if i % 100 == 0:
            print(f"  处理进度: {i}/{len(trades)}")

        code = str(trade["code"]).zfill(6)
        entry_date = trade["entry_date"]

        # 加载股票数据（缓存）
        if code not in cache:
            try:
                df = get_stock_daily(code, C.DATA_START_DATE, C.DATA_END_DATE, C.ADJUST_TYPE)
                df = compute_technical_indicators(df)
                cache[code] = df
            except Exception:
                cache[code] = None
                continue

        df = cache[code]
        if df is None or len(df) == 0:
            continue

        # 找到入场日索引
        date_mask = df["date"] <= entry_date
        if not date_mask.any():
            continue
        entry_idx = date_mask.sum() - 1

        # 估算颈线（用入场价减去突破强度，近似）
        neckline = trade["entry_price"] / 1.01  # 假设1%突破缓冲

        # 提取特征
        feat = extract_features_for_trade(df, entry_idx, neckline, trade["pattern"], index_df)
        feat["label"] = 1 if trade["pnl"] > 0 else 0
        feat["pnl_pct"] = trade["pnl_pct"]
        feat["code"] = code
        feat["entry_date"] = entry_date
        all_features.append(feat)

    df_train = pd.DataFrame(all_features)
    print(f"  有效样本: {len(df_train)}, 胜率: {df_train['label'].mean():.2%}")

    # 准备特征
    pattern_dummies = pd.get_dummies(df_train["pattern_type"], prefix="ptype")
    df_train = pd.concat([df_train, pattern_dummies], axis=1)

    feature_cols = [
        "price_vs_ma5", "price_vs_ma20", "price_vs_ma60",
        "ma5_vs_ma20", "ma20_vs_ma60",
        "breakout_strength", "vol_ratio", "vol_vs_ma5",
        "rsi_14", "macd_hist", "macd_dif",
        "atr_ratio", "ret_20d", "ret_5d", "vol_trend",
        "market_vs_ma60", "market_ret_20d"
    ]
    for col in df_train.columns:
        if col.startswith("ptype_"):
            feature_cols.append(col)

    feature_cols = [c for c in feature_cols if c in df_train.columns]
    X = df_train[feature_cols].fillna(0).values.astype(np.float32)
    y = df_train["label"].values

    # 训练
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print("\n训练XGBoost...")
    model = xgb.XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0,
        random_state=42, eval_metric="logloss"
    )
    model.fit(X_train, y_train)

    # 评估
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_prob)

    print(f"\n  测试集结果:")
    print(f"    准确率: {acc:.4f}")
    print(f"    精确率: {prec:.4f}")
    print(f"    召回率: {rec:.4f}")
    print(f"    AUC: {auc:.4f}")

    # 特征重要性
    importance = model.feature_importances_
    feat_imp = sorted(zip(feature_cols, importance), key=lambda x: x[1], reverse=True)
    print(f"\n  Top 10 重要特征:")
    for feat, imp in feat_imp[:10]:
        print(f"    {feat}: {imp:.4f}")

    # 保存
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump({"model": model, "feature_cols": feature_cols, "accuracy": acc, "auc": auc}, model_path)
    print(f"\n  模型已保存: {model_path}")

    return {"accuracy": acc, "precision": prec, "recall": rec, "auc": auc, "n_samples": len(df_train)}


# === 在线评分函数 ===
_model_cache = None
_feature_cols_cache = None

def load_ml_model(model_path="results/ml_pattern_model.pkl"):
    global _model_cache, _feature_cols_cache
    if _model_cache is not None:
        return _model_cache, _feature_cols_cache
    if os.path.exists(model_path):
        data = joblib.load(model_path)
        _model_cache = data["model"]
        _feature_cols_cache = data["feature_cols"]
        return _model_cache, _feature_cols_cache
    return None, None


def score_signal(df, entry_idx, neckline, pattern_type, index_df=None,
                 model_path="results/ml_pattern_model.pkl"):
    """对一个突破信号评分（0-100），与训练特征一致"""
    model, feature_cols = load_ml_model(model_path)
    if model is None:
        return 50.0  # 默认分

    feat = extract_features_for_trade(df, entry_idx, neckline, pattern_type, index_df)
    feat_df = pd.DataFrame([feat])

    # one-hot形态类型
    pattern_dummies = pd.get_dummies(feat_df["pattern_type"], prefix="ptype")
    feat_df = pd.concat([feat_df, pattern_dummies], axis=1)

    # 确保所有特征列存在
    for col in feature_cols:
        if col not in feat_df.columns:
            feat_df[col] = 0

    X = feat_df[feature_cols].fillna(0).values.astype(np.float32)
    prob = model.predict_proba(X)[0, 1]
    return prob * 100


if __name__ == "__main__":
    from data_fetcher import get_index_daily
    index_df = get_index_daily(C.INDEX_CODE, C.DATA_START_DATE, C.DATA_END_DATE)
    train_from_trades(index_df=index_df)
