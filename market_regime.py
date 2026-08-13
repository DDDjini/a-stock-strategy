# -*- coding: utf-8 -*-
"""
市场状态识别模块
使用高斯混合模型(GMM)识别牛市/熊市/震荡市三种状态
不同状态下调整策略参数
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture


class MarketRegimeDetector:
    """市场状态检测器"""

    def __init__(self, n_regimes=3, lookback=60):
        self.n_regimes = n_regimes
        self.lookback = lookback
        self.gmm = None
        self.regime_labels = {}  # {state_id: "bull"/"bear"/"range"}
        self.fitted = False

    def compute_features(self, index_df):
        """
        计算市场特征
        index_df: 包含 date, close, high, low, volume 的DataFrame
        返回: features DataFrame
        """
        df = index_df.copy()
        df = df.sort_values("date").reset_index(drop=True)

        # 收益率特征
        df["ret_5d"] = df["close"].pct_change(5)
        df["ret_20d"] = df["close"].pct_change(20)
        df["ret_60d"] = df["close"].pct_change(60)

        # 波动率特征
        df["vol_20d"] = df["close"].pct_change().rolling(20).std() * np.sqrt(252)
        df["vol_60d"] = df["close"].pct_change().rolling(60).std() * np.sqrt(252)

        # 趋势特征
        df["ma20"] = df["close"].rolling(20).mean()
        df["ma60"] = df["close"].rolling(60).mean()
        df["ma120"] = df["close"].rolling(120).mean()
        df["trend_strength"] = (df["ma20"] - df["ma60"]) / df["ma60"]
        df["price_vs_ma60"] = (df["close"] - df["ma60"]) / df["ma60"]

        # 高低点特征（ADX近似）
        df["high_low_ratio"] = (df["high"] - df["low"]) / df["close"]
        df["hl_ma20"] = df["high_low_ratio"].rolling(20).mean()

        # 量能特征
        df["vol_ma20"] = df["volume"].rolling(20).mean()
        df["volume_trend"] = df["volume"] / df["vol_ma20"] - 1

        feature_cols = ["ret_20d", "ret_60d", "vol_20d", "trend_strength",
                        "price_vs_ma60", "hl_ma20"]
        df_features = df[["date"] + feature_cols].dropna().reset_index(drop=True)
        return df_features

    def fit(self, index_df):
        """训练GMM模型"""
        features = self.compute_features(index_df)
        X = features.drop(columns=["date"]).values

        # 标准化
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        X_scaled = (X - self.mean_) / self.std_

        # 训练GMM
        self.gmm = GaussianMixture(
            n_components=self.n_regimes,
            covariance_type="full",
            random_state=42,
            max_iter=500,
            n_init=5
        )
        self.gmm.fit(X_scaled)

        # 根据各状态的平均收益率和波动率标注牛/熊/震荡
        state_means = self.gmm.means_
        # ret_20d是第0列，vol_20d是第2列（标准化后）
        # 反标准化看原始值
        orig_means = state_means * self.std_ + self.mean_

        state_info = []
        for i in range(self.n_regimes):
            avg_ret = orig_means[i, 0]  # ret_20d
            avg_vol = orig_means[i, 2]  # vol_20d
            avg_trend = orig_means[i, 3]  # trend_strength
            state_info.append({
                "state": i,
                "avg_ret": avg_ret,
                "avg_vol": avg_vol,
                "avg_trend": avg_trend
            })

        # 排序标注：收益最高=牛市，收益最低=熊市，中间=震荡
        sorted_by_ret = sorted(state_info, key=lambda x: x["avg_ret"], reverse=True)
        self.regime_labels[sorted_by_ret[0]["state"]] = "bull"
        self.regime_labels[sorted_by_ret[-1]["state"]] = "bear"
        if self.n_regimes >= 3:
            self.regime_labels[sorted_by_ret[1]["state"]] = "range"

        self.fitted = True
        self._features = features
        return self.regime_labels

    def predict(self, index_df):
        """预测每个交易日的市场状态"""
        if not self.fitted:
            raise ValueError("模型未训练，请先调用fit()")

        features = self.compute_features(index_df)
        X = features.drop(columns=["date"]).values
        X_scaled = (X - self.mean_) / self.std_

        states = self.gmm.predict(X_scaled)
        probs = self.gmm.predict_proba(X_scaled)

        result = features[["date"]].copy()
        result["state"] = states
        result["regime"] = [self.regime_labels[s] for s in states]
        result["confidence"] = probs.max(axis=1)

        # 添加各状态概率
        for i in range(self.n_regimes):
            label = self.regime_labels.get(i, f"state_{i}")
            result[f"prob_{label}"] = probs[:, i]

        return result

    def get_regime_on_date(self, regime_df, date):
        """获取指定日期的市场状态"""
        date = pd.Timestamp(date)
        mask = regime_df["date"] <= date
        if not mask.any():
            return "range"  # 默认震荡
        row = regime_df[mask].iloc[-1]
        return row["regime"]


def get_regime_params(regime):
    """
    根据市场状态返回策略参数调整
    返回: dict of parameter overrides
    """
    if regime == "bull":
        return {
            "quality_threshold": 40,       # 牛市放松形态质量要求
            "vol_ratio": 0.8,              # 降低量能要求
            "max_positions": 15,           # 增加持仓
            "single_position": 0.08,       # 单仓略降（分散）
            "atr_stop_mult": 2.5,          # 放宽止损（给趋势更多空间）
            "trendline_buffer": 0.015,     # 趋势线跌破缓冲加大
            "use_market_filter": False,    # 牛市不需要大盘过滤（本身就是牛市）
        }
    elif regime == "bear":
        return {
            "quality_threshold": 65,       # 熊市严格筛选
            "vol_ratio": 1.5,              # 要求更强量能
            "max_positions": 5,            # 减少持仓
            "single_position": 0.10,       # 集中但少持仓
            "atr_stop_mult": 1.5,          # 收紧止损
            "trendline_buffer": 0.005,     # 趋势线跌破缓冲减小（快跑）
            "use_market_filter": True,     # 严格大盘过滤
        }
    else:  # range
        return {
            "quality_threshold": 50,       # 默认
            "vol_ratio": 1.0,
            "max_positions": 10,
            "single_position": 0.10,
            "atr_stop_mult": 2.0,
            "trendline_buffer": 0.01,
            "use_market_filter": True,
        }


if __name__ == "__main__":
    # 测试
    from data_fetcher import get_index_daily
    import config as C

    print("获取上证指数数据...")
    index_df = get_index_daily(C.INDEX_CODE, C.DATA_START_DATE, C.DATA_END_DATE)
    print(f"  {len(index_df)} 个交易日")

    print("\n训练市场状态识别模型...")
    detector = MarketRegimeDetector(n_regimes=3, lookback=60)
    labels = detector.fit(index_df)
    print(f"  状态标注: {labels}")

    print("\n预测市场状态...")
    regime_df = detector.predict(index_df)
    print(f"  状态分布:")
    print(regime_df["regime"].value_counts())

    print(f"\n  最近10天:")
    print(regime_df.tail(10)[["date", "regime", "confidence"]].to_string(index=False))

    # 保存
    regime_df.to_csv("results/market_regime.csv", index=False, encoding="utf-8-sig")
    print("\n  已保存到 results/market_regime.csv")
