# -*- coding: utf-8 -*-
"""
机器学习形态评分模块
从历史形态突破信号中提取特征，训练XGBoost分类器预测胜率
用预测概率替代人工规则的质量评分
"""
import warnings
warnings.filterwarnings("ignore")
import os
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, classification_report
import joblib

import config as C
from data_fetcher import compute_technical_indicators
from pattern_recognizer import detect_all_patterns, find_resistance_levels


class MLPatternScorer:
    """机器学习形态评分器"""

    def __init__(self, model_path="results/ml_pattern_model.pkl"):
        self.model_path = model_path
        self.model = None
        self.feature_cols = None
        self.fitted = False

    def extract_signal_features(self, df, pattern, breakout_idx, neckline, index_df=None):
        """
        从一个突破信号中提取特征
        df: 股票日线数据（含技术指标）
        pattern: 形态信息dict
        breakout_idx: 突破日在df中的索引
        neckline: 颈线价格
        index_df: 指数数据（可选，用于市场特征）
        返回: feature dict
        """
        features = {}
        row = df.iloc[breakout_idx]

        # === 1. 形态特征 ===
        features["pattern_type"] = pattern.get("pattern_type", "unknown")
        features["quality_score"] = pattern.get("quality_score", 50)
        features["pattern_days"] = pattern.get("pattern_days", 0)
        features["neckline_touches"] = pattern.get("touch_count", 0)

        # 形态深度（从最低点到颈线的幅度）
        if "min_price" in pattern and neckline > 0:
            features["pattern_depth"] = (neckline - pattern["min_price"]) / neckline
        else:
            features["pattern_depth"] = 0

        # === 2. 突破特征 ===
        features["breakout_strength"] = (row["close"] - neckline) / neckline
        features["vol_ratio"] = row.get("vol_ratio_5", 1.0)
        features["vol_vs_ma3"] = row["volume"] / row.get("vol_ma3", row["volume"])

        # === 3. 价格位置特征 ===
        features["price_vs_ma5"] = (row["close"] - row.get("ma5", row["close"])) / row["close"]
        features["price_vs_ma20"] = (row["close"] - row.get("ma20", row["close"])) / row["close"]
        features["price_vs_ma60"] = (row["close"] - row.get("ma60", row["close"])) / row["close"]
        features["ma5_vs_ma20"] = (row.get("ma5", 0) - row.get("ma20", 0)) / row.get("ma20", 1)
        features["ma20_vs_ma60"] = (row.get("ma20", 0) - row.get("ma60", 0)) / row.get("ma60", 1)

        # === 4. 动量特征 ===
        features["rsi_14"] = row.get("rsi_14", 50)
        features["macd_hist"] = row.get("macd_hist", 0)
        features["macd_dif"] = row.get("macd_dif", 0)
        features["macd_dea"] = row.get("macd_dea", 0)

        # === 5. 波动率特征 ===
        features["atr_14"] = row.get("atr_14", 0)
        features["atr_ratio"] = row.get("atr_14", 0) / row["close"] if row["close"] > 0 else 0

        # 近20日收益率
        if breakout_idx >= 20:
            features["ret_20d"] = (row["close"] - df.iloc[breakout_idx - 20]["close"]) / df.iloc[breakout_idx - 20]["close"]
            features["ret_5d"] = (row["close"] - df.iloc[breakout_idx - 5]["close"]) / df.iloc[breakout_idx - 5]["close"]
        else:
            features["ret_20d"] = 0
            features["ret_5d"] = 0

        # === 6. 量能趋势 ===
        if breakout_idx >= 10:
            vol_5 = df.iloc[breakout_idx - 4:breakout_idx + 1]["volume"].mean()
            vol_20 = df.iloc[breakout_idx - 19:breakout_idx + 1]["volume"].mean()
            features["vol_trend"] = vol_5 / vol_20 if vol_20 > 0 else 1
        else:
            features["vol_trend"] = 1

        # === 7. 市场环境特征 ===
        if index_df is not None and len(index_df) > 0:
            idx_date = row["date"]
            idx_mask = index_df["date"] <= idx_date
            if idx_mask.any():
                idx_row = index_df[idx_mask].iloc[-1]
                idx_close = idx_row["close"]
                idx_ma60 = index_df[idx_mask]["close"].rolling(60).mean().iloc[-1] if idx_mask.sum() >= 60 else idx_close
                features["market_vs_ma60"] = (idx_close - idx_ma60) / idx_ma60 if idx_ma60 > 0 else 0
                # 市场近20日收益
                if idx_mask.sum() >= 20:
                    idx_20ago = index_df[idx_mask].iloc[-20]["close"]
                    features["market_ret_20d"] = (idx_close - idx_20ago) / idx_20ago
                else:
                    features["market_ret_20d"] = 0
            else:
                features["market_vs_ma60"] = 0
                features["market_ret_20d"] = 0
        else:
            features["market_vs_ma60"] = 0
            features["market_ret_20d"] = 0

        return features

    def build_training_data(self, stock_data_dict, index_df=None, forward_days=10):
        """
        从所有股票中构建训练数据
        forward_days: 用未来N天收益作为标签
        返回: DataFrame of features + label
        """
        all_features = []

        for code, df in stock_data_dict.items():
            if len(df) < 150:
                continue

            df = compute_technical_indicators(df.copy())
            if df is None or len(df) < 150:
                continue

            # 找阻力位
            resistance_levels = find_resistance_levels(df)
            if not resistance_levels:
                continue

            # 对每个阻力位检测形态和突破
            for res_price, first_idx, last_idx, touch_count in resistance_levels:
                if last_idx >= len(df) - forward_days - 5:
                    continue

                # 在阻力位附近检测形态
                lookback_start = max(0, last_idx - C.PATTERN_LOOKBACK)
                window_df = df.iloc[lookback_start:last_idx + 1].reset_index(drop=True)
                patterns = detect_all_patterns(window_df)

                if not patterns:
                    continue

                # 找突破日
                for pattern in patterns:
                    # 找突破日（在last_idx之后的120天内）
                    search_end = min(len(df), last_idx + 120)
                    for i in range(last_idx + 1, search_end):
                        if i >= len(df) - forward_days:
                            break
                        row_i = df.iloc[i]
                        prev_close = df.iloc[i - 1]["close"]
                        vol_ma3 = df.iloc[max(0, i - 3):i]["volume"].mean()

                        # 突破条件
                        if (row_i["close"] > res_price * (1 + C.BREAKOUT_CLOSE_BUFFER) and
                            prev_close <= res_price and
                            row_i["volume"] > vol_ma3 * C.BREAKOUT_VOL_RATIO):

                            # 提取特征
                            feat = self.extract_signal_features(df, pattern, i, res_price, index_df)
                            feat["code"] = code
                            feat["breakout_date"] = row_i["date"]

                            # 标签：未来forward_days收益 > 0
                            if i + forward_days < len(df):
                                future_ret = (df.iloc[i + forward_days]["close"] - row_i["close"]) / row_i["close"]
                                feat["label"] = 1 if future_ret > 0 else 0
                                feat["forward_return"] = future_ret
                                all_features.append(feat)
                            break  # 每个形态只取第一个突破日

        if not all_features:
            return pd.DataFrame()

        df_train = pd.DataFrame(all_features)
        return df_train

    def prepare_features(self, df):
        """准备特征矩阵：处理类别变量，选择数值特征"""
        df = df.copy()

        # 形态类型one-hot
        if "pattern_type" in df.columns:
            pattern_dummies = pd.get_dummies(df["pattern_type"], prefix="ptype")
            df = pd.concat([df, pattern_dummies], axis=1)

        # 数值特征列
        numeric_cols = [
            "quality_score", "pattern_days", "neckline_touches", "pattern_depth",
            "breakout_strength", "vol_ratio", "vol_vs_ma3",
            "price_vs_ma5", "price_vs_ma20", "price_vs_ma60",
            "ma5_vs_ma20", "ma20_vs_ma60",
            "rsi_14", "macd_hist", "macd_dif", "macd_dea",
            "atr_14", "atr_ratio", "ret_20d", "ret_5d", "vol_trend",
            "market_vs_ma60", "market_ret_20d"
        ]
        # 加上形态类型one-hot
        for col in df.columns:
            if col.startswith("ptype_"):
                numeric_cols.append(col)

        # 只保留存在的列
        feature_cols = [c for c in numeric_cols if c in df.columns]
        self.feature_cols = feature_cols

        X = df[feature_cols].fillna(0).values.astype(np.float32)
        y = df["label"].values if "label" in df.columns else None

        return X, y, feature_cols

    def train(self, stock_data_dict, index_df=None, forward_days=10, test_size=0.2):
        """
        训练XGBoost模型
        """
        print("  构建训练数据...")
        df_train = self.build_training_data(stock_data_dict, index_df, forward_days)
        if df_train.empty:
            print("  警告：没有训练数据！")
            return None

        print(f"  样本数: {len(df_train)}, 正样本率: {df_train['label'].mean():.2%}")

        X, y, feature_cols = self.prepare_features(df_train)

        # 划分训练/测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

        # 训练XGBoost
        print("  训练XGBoost...")
        self.model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            eval_metric="logloss",
            use_label_encoder=False
        )
        self.model.fit(X_train, y_train)

        # 评估
        y_pred = self.model.predict(X_test)
        y_prob = self.model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_prob)

        print(f"\n  模型评估（测试集）:")
        print(f"    准确率: {acc:.4f}")
        print(f"    精确率: {prec:.4f}")
        print(f"    召回率: {rec:.4f}")
        print(f"    AUC: {auc:.4f}")

        # 特征重要性
        importance = self.model.feature_importances_
        feat_imp = sorted(zip(feature_cols, importance), key=lambda x: x[1], reverse=True)
        print(f"\n  Top 10 重要特征:")
        for feat, imp in feat_imp[:10]:
            print(f"    {feat}: {imp:.4f}")

        self.fitted = True
        self._feature_importance = feat_imp

        # 保存模型
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        joblib.dump({
            "model": self.model,
            "feature_cols": self.feature_cols,
            "accuracy": acc,
            "auc": auc
        }, self.model_path)
        print(f"\n  模型已保存到 {self.model_path}")

        return {"accuracy": acc, "precision": prec, "recall": rec, "auc": auc}

    def load_model(self):
        """加载已训练模型"""
        if os.path.exists(self.model_path):
            data = joblib.load(self.model_path)
            self.model = data["model"]
            self.feature_cols = data["feature_cols"]
            self.fitted = True
            return True
        return False

    def score(self, df, pattern, breakout_idx, neckline, index_df=None):
        """
        对一个突破信号评分（0-100分，表示预测胜率）
        """
        if not self.fitted:
            if not self.load_model():
                return pattern.get("quality_score", 50)

        feat = self.extract_signal_features(df, pattern, breakout_idx, neckline, index_df)
        feat_df = pd.DataFrame([feat])

        # 准备特征（需要one-hot）
        if "pattern_type" in feat_df.columns:
            pattern_dummies = pd.get_dummies(feat_df["pattern_type"], prefix="ptype")
            feat_df = pd.concat([feat_df, pattern_dummies], axis=1)

        # 确保所有特征列都存在
        for col in self.feature_cols:
            if col not in feat_df.columns:
                feat_df[col] = 0

        X = feat_df[self.feature_cols].fillna(0).values.astype(np.float32)
        prob = self.model.predict_proba(X)[0, 1]
        return prob * 100  # 转换为0-100分


if __name__ == "__main__":
    from run_backtest import get_hs300_constituents, fetch_universe_data
    from data_fetcher import get_index_daily

    print("获取数据...")
    stocks = get_hs300_constituents()
    stock_codes = stocks["code"].tolist()[:80]
    index_df = get_index_daily(C.INDEX_CODE, C.DATA_START_DATE, C.DATA_END_DATE)
    stock_data = fetch_universe_data(stock_codes, C.DATA_START_DATE, C.DATA_END_DATE)
    print(f"有效股票: {len(stock_data)} 只")

    print("\n训练ML形态评分模型...")
    scorer = MLPatternScorer()
    results = scorer.train(stock_data, index_df, forward_days=10)
