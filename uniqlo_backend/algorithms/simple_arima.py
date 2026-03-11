#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
改进的简单 ARIMA 实现
不依赖 statsmodels，使用统计方法进行时间序列预测
"""

import numpy as np
import pandas as pd
from typing import Dict, List


class ImprovedSimpleARIMA:
    """
    改进的简单 ARIMA 实现
    
    使用统计方法分析趋势、季节性和自相关性
    不依赖 statsmodels，避免崩溃风险
    """

    def __init__(self, series: pd.Series, params: Dict[str, int]):
        self.series = series
        self.params = params
        self.p = params.get('p', 2)
        self.d = params.get('d', 1)
        self.q = params.get('q', 2)

        # 计算序列统计特性
        self._analyze_series()

        # 拟合模型
        self.fitted_values = self._fit()

        # 添加属性
        self.aic = self._calculate_aic()
        self.bic = self._calculate_bic()

    def _analyze_series(self):
        """分析序列特性"""
        values = self.series.values

        # 1. 计算趋势（线性回归）
        x = np.arange(len(values))
        coeffs = np.polyfit(x, values, 1)
        self.trend_slope = coeffs[0]
        self.trend_intercept = coeffs[1]
        self.trend = coeffs[0] * x + coeffs[1]

        # 2. 计算去除趋势后的残差
        self.residuals = values - self.trend

        # 3. 计算季节性（7天周期）
        self.seasonal_period = 7
        seasonal = np.zeros(self.seasonal_period)
        for i in range(self.seasonal_period):
            indices = np.arange(i, len(values), self.seasonal_period)
            if len(indices) > 0:
                seasonal[i] = np.mean(self.residuals[indices])
        self.seasonal = np.tile(seasonal, len(values) // self.seasonal_period + 1)[:len(values)]

        # 4. 计算自相关系数（AR部分）
        self._calculate_autocorrelation()

        # 5. 计算移动平均系数（MA部分）
        self._calculate_ma_coefficients()

    def _calculate_autocorrelation(self):
        """计算自相关系数"""
        n = len(self.residuals)
        mean = np.mean(self.residuals)
        var = np.var(self.residuals)

        if var == 0:
            self.autocorr = np.zeros(min(self.p + 1, n))
            return

        autocorr = np.correlate(self.residuals - mean, self.residuals - mean, mode='full')
        autocorr = autocorr[n-1:n+self.p]
        self.autocorr = autocorr / (var * n)

    def _calculate_ma_coefficients(self):
        """计算移动平均系数"""
        # 使用残差序列的移动平均
        window = self.q
        if window > 0:
            self.ma_coeffs = np.convolve(self.residuals, np.ones(window)/window, mode='same')
        else:
            self.ma_coeffs = np.zeros_like(self.residuals)

    def _fit(self):
        """拟合模型"""
        # 组合预测：趋势 + 季节性 + AR + MA
        values = self.series.values

        # 基础预测：趋势 + 季节性
        base = self.trend + self.seasonal

        # AR 调整
        ar_adjustment = np.zeros(len(values))
        for lag in range(1, min(self.p + 1, len(self.residuals))):
            weight = self.autocorr[lag] if lag < len(self.autocorr) else 0
            ar_adjustment[lag:] += weight * self.residuals[lag:]

        # MA 调整
        ma_adjustment = self.ma_coeffs * 0.3

        # 拟合值
        fitted = base + ar_adjustment * 0.5 + ma_adjustment * 0.3

        return fitted

    def _calculate_aic(self):
        """计算 AIC"""
        n = len(self.series)
        residuals = self.series.values - self.fitted_values
        rss = np.sum(residuals ** 2)

        if rss == 0:
            return 0

        k = self.p + self.d + self.q + 1  # 参数数量
        aic = n * np.log(rss / n) + 2 * k
        return max(0, aic)  # 确保非负

    def _calculate_bic(self):
        """计算 BIC"""
        n = len(self.series)
        residuals = self.series.values - self.fitted_values
        rss = np.sum(residuals ** 2)

        if rss == 0:
            return 0

        k = self.p + self.d + self.q + 1
        bic = n * np.log(rss / n) + k * np.log(n)
        return max(0, bic)

    def predict(self, steps: int):
        """预测未来值"""
        n = len(self.series)
        predictions = np.zeros(steps)

        for i in range(steps):
            future_idx = n + i
            x = future_idx

            # 趋势预测
            trend_pred = self.trend_slope * x + self.trend_intercept

            # 季节性预测
            seasonal_idx = i % self.seasonal_period
            seasonal_pred = np.mean(self.seasonal[seasonal_idx::self.seasonal_period]) if seasonal_idx < 7 else 0

            # AR 预测（基于最近的历史残差）
            ar_pred = 0
            if len(self.residuals) > 0:
                recent_residuals = self.residuals[-self.p:] if len(self.residuals) >= self.p else self.residuals
                weights = self.autocorr[1:len(recent_residuals)+1] if len(self.autocorr) > 1 else np.ones(len(recent_residuals)) * 0.3
                ar_pred = np.sum(recent_residuals * weights[:len(recent_residuals)]) / len(recent_residuals) if len(recent_residuals) > 0 else 0

            # 组合预测
            predictions[i] = trend_pred + seasonal_pred + ar_pred * 0.5

        return predictions


# 保持原有接口兼容
class SimpleARIMAFit:
    """简单 ARIMA 实现 (当 statsmodels 不可用时) - 使用改进版本"""

    def __init__(self, series: pd.Series, params: Dict[str, int]):
        # 使用改进的实现
        self._improved = ImprovedSimpleARIMA(series, params)
        self.series = series
        self.params = params
        self.fitted_values = self._improved.fitted_values
        self.aic = self._improved.aic
        self.bic = self._improved.bic

    def predict(self, steps: int):
        return self._improved.predict(steps)
