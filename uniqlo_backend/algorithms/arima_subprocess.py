#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ARIMA 子进程运行器
用于在独立子进程中运行 statsmodels ARIMA，避免主进程崩溃
"""

import sys
import json
import warnings
warnings.filterwarnings('ignore')

def run_arima_in_subprocess(series_data, order, task='fit_predict'):
    """
    在子进程中运行 statsmodels ARIMA
    """
    from statsmodels.tsa.arima.model import ARIMA
    import numpy as np

    series = np.array(series_data)

    if task == 'params':
        # 参数选择任务
        from statsmodels.tsa.stattools import adfuller
        import pandas as pd

        def check_stationarity(s):
            result = adfuller(s.dropna())
            return result[1] < 0.05

        # 确定差分阶数
        d = 1
        diff = series.copy()
        for i in range(2):
            if not check_stationarity(pd.Series(diff)):
                d = i + 1
                diff = np.diff(diff)

        # 简单参数搜索
        best_aic = float('inf')
        best_params = (1, d, 1)
        param_candidates = [(1, d, 0), (1, d, 1), (2, d, 0), (2, d, 1), (1, d, 2)]

        for p, d_test, q in param_candidates:
            try:
                model = ARIMA(series, order=(p, d_test, q))
                fitted = model.fit(method='innovations_mle')
                if fitted.aic < best_aic:
                    best_aic = fitted.aic
                    best_params = (p, d_test, q)
            except:
                continue

        return {'params': list(best_params), 'aic': best_aic}

    else:
        # 拟合和预测任务
        model = ARIMA(series, order=order)
        fitted = model.fit()

        return {
            'fittedvalues': fitted.fittedvalues.tolist(),
            'aic': fitted.aic,
            'bic': fitted.bic,
            'forecast': fitted.forecast(30).tolist()
        }


if __name__ == '__main__':
    # 从命令行参数读取
    # 格式: python arima_subprocess.py <series_json> <order_json> <task> <output_file>
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('series_json', help='JSON string of series data')
    parser.add_argument('order_json', help='JSON string of order (p,d,q)')
    parser.add_argument('task', help='Task: params or fit_predict')
    parser.add_argument('output_file', help='Output file path')
    args = parser.parse_args()

    import json
    series_data = json.loads(args.series_json)
    order = tuple(json.loads(args.order_json))
    task = args.task

    try:
        result = run_arima_in_subprocess(series_data, order, task)
        with open(args.output_file, 'w') as f:
            json.dump({'success': True, 'result': result}, f)
    except Exception as e:
        with open(args.output_file, 'w') as f:
            json.dump({'success': False, 'error': str(e)}, f)
