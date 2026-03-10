#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
纯 numpy 实现 ARIMA
完全不依赖 scipy 优化器，避免 OpenBLAS 崩溃问题

Author: Uniqlo Analysis System
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple


class PureNumpyARIMA:
    """
    纯 numpy 实现的 ARIMA(p,d,q)
    使用最小二乘法进行参数估计，完全避免第三方优化器
    """

    def __init__(self, series: np.ndarray, order: Tuple[int, int, int] = (1, 1, 1)):
        """
        初始化 ARIMA 模型
        """
        self.order = order
        self.p, self.d, self.q = order

        # 存储原始序列
        self.original_series = series.copy() if isinstance(series, np.ndarray) else series.values.copy()

        # 差分序列
        self.series = self._difference(self.original_series, self.d)

        # 参数
        self.ar_params = None
        self.ma_params = None
        self.sigma2 = None

        # 拟合结果
        self.fitted_values = None
        self.aic = None
        self.bic = None

        # 拟合
        if len(self.series) > self.p + self.q + 5:
            self._fit()

    def _difference(self, series: np.ndarray, d: int) -> np.ndarray:
        """差分操作"""
        diff = series.copy()
        for _ in range(d):
            diff = np.diff(diff)
        return diff

    def _inverse_difference(self, forecasts: np.ndarray, original: np.ndarray, d: int) -> np.ndarray:
        """逆差分"""
        if d == 0:
            return forecasts

        result = forecasts.copy()
        n_forecast = len(forecasts)

        inv_diff = np.zeros(n_forecast)
        if len(original) > 0:
            inv_diff[0] = forecasts[0] + original[-1]
        else:
            inv_diff[0] = forecasts[0]

        for i in range(1, n_forecast):
            inv_diff[i] = forecasts[i] + inv_diff[i - 1]

        return inv_diff

    def _fit(self):
        """使用最小二乘法拟合 AR 模型"""
        y = self.series
        n = len(y)

        if n <= self.p:
            self.ar_params = np.zeros(self.p)
            self.ma_params = np.zeros(self.q) if self.q > 0 else np.array([])
            self.sigma2 = np.var(y) if len(y) > 0 else 1.0
            self.fitted_values = y.copy()
            return

        # 构建 AR 矩阵 (Yule-Walker 方程)
        # y[t] = a1*y[t-1] + a2*y[t-2] + ... + ap*y[t-p] + e[t]
        X = np.zeros((n - self.p, self.p))
        for i in range(self.p):
            X[:, i] = y[self.p - 1 - i:n - 1 - i]

        y_train = y[self.p:]

        # 最小二乘解
        try:
            # 使用伪逆
            XtX = X.T @ X
            XtX_inv = np.linalg.pinv(XtX)
            self.ar_params = XtX_inv @ (X.T @ y_train)
        except:
            self.ar_params = np.zeros(self.p)

        # 计算拟合值
        fitted = np.zeros(n)
        fitted[:self.p] = y[:self.p]
        for t in range(self.p, n):
            ar_part = np.sum(self.ar_params * y[t - self.p:t][::-1])
            fitted[t] = ar_part

        self.fitted_values = fitted

        # 残差
        residuals = y[self.p:] - fitted[self.p:]
        self.sigma2 = np.var(residuals) if len(residuals) > 0 else 1.0

        # 简化 MA 参数
        self.ma_params = np.zeros(self.q) if self.q > 0 else np.array([])

        # 计算 AIC/BIC
        n_params = self.p + self.q + 1
        rss = np.sum(residuals ** 2) if len(residuals) > 0 else 1.0

        if rss > 0 and n > n_params:
            self.aic = n * np.log(rss / n + 1e-10) + 2 * n_params
            self.bic = n * np.log(rss / n + 1e-10) + n_params * np.log(n)
        else:
            self.aic = 0
            self.bic = 0

    def predict(self, steps: int = 1) -> np.ndarray:
        """预测未来值"""
        y = self.series
        n = len(y)

        predictions = np.zeros(steps)

        # 使用最后 p 个值作为起始
        last_values = list(y[-self.p:]) if len(y) >= self.p else list(y)

        for h in range(steps):
            # AR 预测
            ar_part = np.sum(self.ar_params * last_values[-self.p:][::-1])
            predictions[h] = ar_part

            # 更新历史
            last_values.append(predictions[h])

        # 逆差分
        if self.d > 0:
            predictions = self._inverse_difference(predictions, self.original_series, self.d)

        return predictions

    def get_forecast(self, steps: int = 1):
        """获取预测结果（兼容接口）"""
        predictions = self.predict(steps)

        class ForecastResult:
            def __init__(self, pred, model):
                self.predicted_mean = pred
                self.model = model

            def conf_int(self, alpha=0.05):
                std = np.sqrt(model.sigma2) if model.sigma2 else 1
                z = 1.96  # 95% 置信度
                n = len(self.predicted_mean)
                lower = self.predicted_mean - z * std * np.sqrt(np.arange(1, n + 1) * 0.5)
                upper = self.predicted_mean + z * std * np.sqrt(np.arange(1, n + 1) * 0.5)
                return np.column_stack([lower, upper])

        return ForecastResult(predictions, self)


def fit_arima(series: np.ndarray, order: Tuple[int, int, int] = (1, 1, 1)) -> PureNumpyARIMA:
    """拟合 ARIMA 模型"""
    return PureNumpyARIMA(series, order)
