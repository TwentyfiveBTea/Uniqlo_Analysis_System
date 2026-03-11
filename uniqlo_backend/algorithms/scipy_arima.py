#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于 scipy 的 ARIMA 实现
替代 statsmodels，避免崩溃问题

Author: Uniqlo Analysis System
"""

# 修复 OpenBLAS 多线程崩溃问题
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class ScipyARIMA:
    """
    基于 scipy 的 ARIMA(p,d,q) 实现
    
    使用最大似然估计 (MLE) 进行参数估计
    支持趋势和季节性分析
    """

    def __init__(self, series: np.ndarray, order: Tuple[int, int, int] = (1, 1, 1)):
        """
        初始化 ARIMA 模型
        
        Args:
            series: 时间序列数据
            order: (p, d, q) - 自回归阶数, 差分阶数, 移动平均阶数
        """
        self.order = order
        self.p, self.d, self.q = order
        
        # 存储原始序列
        self.original_series = series.copy() if isinstance(series, np.ndarray) else series.values.copy()
        
        # 差分序列
        self.series = self._difference(self.original_series, self.d)
        
        # 参数
        self.ar_params = None  # AR 系数
        self.ma_params = None  # MA 系数
        self.sigma2 = None  # 噪声方差
        
        # 拟合结果
        self.fitted_values = None
        self.residuals = None
        self.aic = None
        self.bic = None
        self.loglikelihood = None
        
        # 拟合模型
        if len(self.series) > max(self.p, self.q) + 10:
            self._fit()

    def _difference(self, series: np.ndarray, d: int) -> np.ndarray:
        """差分操作"""
        diff = series.copy()
        for _ in range(d):
            diff = np.diff(diff)
        return diff

    def _inverse_difference(self, forecasts: np.ndarray, original: np.ndarray, d: int) -> np.ndarray:
        """差分的逆操作"""
        # 对于 d=1: forecast[i] = forecast[i] + original[i-1]
        # 对于 d>1: 递归应用
        if d == 0:
            return forecasts
        
        result = forecasts.copy()
        n_orig = len(original)
        n_forecast = len(forecasts)
        
        # 逆差分
        inv_diff = np.zeros(n_forecast)
        if n_orig > 0:
            inv_diff[0] = forecasts[0] + original[-1]
        else:
            inv_diff[0] = forecasts[0]
        
        for i in range(1, n_forecast):
            inv_diff[i] = forecasts[i] + inv_diff[i-1]
        
        return inv_diff

    def _fit(self):
        """拟合 ARMA(p,q) 模型"""
        y = self.series
        n = len(y)
        
        if n <= self.p + self.q:
            logger.warning("Insufficient data for ARIMA fitting")
            self._use_default_fit()
            return
        
        # 初始化参数
        n_params = self.p + self.q + 1  # AR + MA + sigma2
        initial_params = np.zeros(n_params)
        
        # 使用 Yule-Walker 估计初始化 AR 参数
        if self.p > 0:
            # 计算样本自协方差
            autocov = np.array([np.cov(y[:-i] if i > 0 else y, y[i:] if i > 0 else y)[0, 1] for i in range(self.p + 1)])
            # Yule-Walker 方程
            try:
                gamma = autocov[:-1]
                gamma_mat = np.array([[autocov[abs(i-j)] for j in range(self.p)] for i in range(self.p)])
                initial_params[:self.p] = np.linalg.solve(gamma_mat, gamma)
            except:
                initial_params[:self.p] = np.zeros(self.p)
        
        # 初始 MA 系数设为小的随机值
        initial_params[self.p:self.p+self.q] = 0.1
        
        # 初始方差
        initial_params[-1] = np.var(y)
        
        # 确保方差为正
        initial_params[-1] = max(initial_params[-1], 0.01)
        
        # 使用拟牛顿法优化
        try:
            result = minimize(
                self._neg_log_likelihood,
                initial_params,
                method='L-BFGS-B',
                options={'maxiter': 500, 'disp': False}
            )
            
            if result.success:
                params = result.x
                self.ar_params = params[:self.p]
                self.ma_params = params[self.p:self.p+self.q]
                self.sigma2 = max(params[-1], 0.01)
                self.loglikelihood = -result.fun
            else:
                logger.warning("Optimization did not converge, using default fit")
                self._use_default_fit()
        except Exception as e:
            logger.warning(f"Fitting failed: {e}, using default fit")
            self._use_default_fit()
        
        # 计算拟合值和残差
        self._calculate_fitted()
        
        # 计算 AIC/BIC
        self._calculate_ic()

    def _use_default_fit(self):
        """使用默认/简单拟合"""
        y = self.series
        n = len(y)
        
        # 简单 AR 拟合
        if self.p > 0 and n > self.p:
            X = np.column_stack([y[i:n-self.p+i] for i in range(self.p)])
            y_train = y[self.p:]
            try:
                self.ar_params = np.linalg.lstsq(X, y_train, rcond=None)[0]
            except:
                self.ar_params = np.zeros(self.p)
        else:
            self.ar_params = np.zeros(self.p)
        
        self.ma_params = np.zeros(self.q) if self.q > 0 else np.array([])
        self.sigma2 = np.var(y) if len(y) > 0 else 1.0

    def _neg_log_likelihood(self, params: np.ndarray) -> float:
        """负对数似然函数"""
        ar = params[:self.p]
        ma = params[self.p:self.p+self.q]
        sigma2 = max(params[-1], 0.01)
        
        y = self.series
        n = len(y)
        
        if n <= max(self.p, self.q):
            return 1e10
        
        # 初始化残差
        residuals = np.zeros(n)
        
        # 使用滤波器计算残差
        for t in range(max(self.p, self.q), n):
            # AR 部分
            ar_part = np.sum(ar * y[t-self.p:t][::-1]) if self.p > 0 else 0
            # MA 部分 (简化)
            ma_part = np.sum(ma * residuals[t-self.q:t][::-1]) if self.q > 0 else 0
            residuals[t] = y[t] - ar_part - ma_part
        
        # 忽略初始部分
        valid_resid = residuals[max(self.p, self.q):]
        
        # 对数似然
        ll = -0.5 * len(valid_resid) * np.log(2 * np.pi * sigma2) - 0.5 * np.sum(valid_resid**2) / sigma2
        
        return -ll

    def _calculate_fitted(self):
        """计算拟合值"""
        y = self.series
        n = len(y)
        
        if n == 0:
            self.fitted_values = np.array([])
            self.residuals = np.array([])
            return
        
        fitted = np.zeros(n)
        residuals = np.zeros(n)
        
        for t in range(max(self.p, self.q), n):
            # AR 部分
            ar_part = np.sum(self.ar_params * y[t-self.p:t][::-1]) if self.p > 0 else 0
            # MA 部分
            ma_part = np.sum(self.ma_params * residuals[t-self.q:t][::-1]) if self.q > 0 else 0
            fitted[t] = ar_part + ma_part
            residuals[t] = y[t] - fitted[t]
        
        # 初始值设为观测值
        fitted[:max(self.p, self.q)] = y[:max(self.p, self.q)]
        residuals[:max(self.p, self.q)] = 0
        
        self.fitted_values = fitted
        self.residuals = residuals

    def _calculate_ic(self):
        """计算信息准则"""
        n = len(self.series)
        k = self.p + self.q + 1  # 参数数量
        
        if self.loglikelihood is not None and n > k:
            self.aic = 2 * k - 2 * self.loglikelihood
            self.bic = k * np.log(n) - 2 * self.loglikelihood
        else:
            # 使用近似计算
            if self.sigma2 is not None and self.sigma2 > 0:
                rss = np.sum(self.residuals**2)
                self.aic = n * np.log(rss/n + 1e-10) + 2 * k
                self.bic = n * np.log(rss/n + 1e-10) + k * np.log(n)
            else:
                self.aic = 0
                self.bic = 0

    def predict(self, steps: int = 1) -> np.ndarray:
        """
        预测未来值
        
        Args:
            steps: 预测步数
            
        Returns:
            预测值数组
        """
        forecasts = np.zeros(steps)
        
        # 组合原始序列和预测
        combined = list(self.series) + [0] * steps
        
        for h in range(steps):
            t = len(self.series) + h
            
            # AR 部分
            ar_part = 0
            if self.p > 0 and t > 0:
                ar_start = max(0, t - self.p)
                ar_values = combined[ar_start:t][::-1]
                ar_part = np.sum(self.ar_params[:len(ar_values)] * ar_values)
            
            # MA 部分
            ma_part = 0
            if self.q > 0 and t > 0:
                # 需要历史残差，这里简化处理
                ma_part = 0
            
            forecasts[h] = ar_part + ma_part
        
        # 逆差分 (从差分空间转回原始空间)
        if self.d > 0:
            forecasts = self._inverse_difference(forecasts, self.original_series, self.d)
        
        return forecasts

    def get_forecast(self, steps: int = 1):
        """
        获取预测结果（兼容 statsmodels 接口）

        Returns:
            ForecastResult 对象
        """
        predictions = self.predict(steps)

        class ForecastResult:
            def __init__(self, pred, model):
                self.predicted_mean = pred
                self.model = model

            def conf_int(self, alpha=0.05):
                std = np.sqrt(model.sigma2) if model.sigma2 else 1
                z = stats.norm.ppf(1 - alpha/2)
                n = len(self.predicted_mean)
                lower = self.predicted_mean - z * std * np.sqrt(np.arange(1, n+1))
                upper = self.predicted_mean + z * std * np.sqrt(np.arange(1, n+1))
                return np.column_stack([lower, upper])

        return ForecastResult(predictions, self)


def fit_arima(series: np.ndarray, order: Tuple[int, int, int] = (1, 1, 1)) -> ScipyARIMA:
    """
    拟合 ARIMA 模型
    
    Args:
        series: 时间序列数据
        order: (p, d, q) 模型阶数
        
    Returns:
        拟合后的 ScipyARIMA 模型
    """
    return ScipyARIMA(series, order)
