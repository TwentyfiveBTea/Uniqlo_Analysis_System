#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Uniqlo Analysis System - Flask RESTful API
============================================
基于 Hadoop 的优衣库订单数据分析与可视化系统后端 API

Related to Research: System Implementation
研究内容：系统后端服务架构与API接口设计

功能说明：
1. ARIMA 销量预测接口
2. K-means 用户画像聚类接口
3. Decision Tree 区域销售分析接口
4. Apriori 商品关联规则接口
5. 数据聚合统计接口

Author: Uniqlo Analysis System
"""

import os
import sys
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional

# 添加算法模块路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, jsonify, request
from flask_cors import CORS

# 导入算法模块
from algorithms.arima_trend import run_arima_forecast, ARIMATrendForecaster
from algorithms.user_portrait_kmeans import run_user_clustering, KMeansUserProfiler
from algorithms.distribution_tree import run_decision_tree_analysis, DistributionDecisionTree
from algorithms.market_basket_apriori import run_apriori_analysis, AprioriAnalyzer

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 初始化 Flask 应用
app = Flask(__name__)
CORS(app)  # 启用跨域请求

# 配置
app.config['JSON_AS_ASCII'] = False
app.config['JSON_SORT_KEYS'] = False


# ============================================================================
# 健康检查接口
# ============================================================================

@app.route('/api/health', methods=['GET'])
def health_check():
    """健康检查接口"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'service': 'Uniqlo Analysis System API'
    })


# ============================================================================
# 数据概览接口
# ============================================================================

@app.route('/api/overview', methods=['GET'])
def get_overview():
    """
    获取数据概览
    
    返回聚合统计数据用于前端展示
    """
    try:
        # 导入数据处理模块
        from data_utils import get_overview as get_data_overview
        
        # 获取实际数据概览
        overview = get_data_overview()
        
        return jsonify(overview)
    
    except Exception as e:
        logger.error(f"Error getting overview: {e}")
        return jsonify({'error': str(e)}), 500


# ============================================================================
# ARIMA 销量预测接口
# ============================================================================

@app.route('/api/arima/forecast', methods=['GET', 'POST'])
def arima_forecast():
    """
    ARIMA 销量预测接口
    
    Query Parameters (GET) or Body Parameters (POST):
    - category: 商品品类 (可选)
    - region: 地区 (可选)
    - forecast_days: 预测天数，默认30天
    
    Returns:
    - historical_data: 历史数据
    - forecast_data: 预测数据
    - metrics: 模型评估指标
    - model_params: 模型参数
    - confidence_interval: 置信区间
    """
    try:
        # 获取请求参数
        if request.method == 'POST':
            params = request.get_json() or {}
        else:
            params = request.args.to_dict()
        
        category = params.get('category')
        region = params.get('region')
        forecast_days = int(params.get('forecast_days', 30))
        
        logger.info(f"ARIMA forecast request: category={category}, region={region}, days={forecast_days}")
        
        # 运行预测
        result = run_arima_forecast(
            category=category,
            region=region,
            forecast_days=forecast_days
        )
        
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"Error in ARIMA forecast: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/arima/categories', methods=['GET'])
def get_arima_categories():
    """获取可用的预测品类"""
    from data_utils import load_all_years, get_categories
    
    all_data = load_all_years()
    categories_list = get_categories(all_data)
    
    # 映射到API格式
    categories = []
    for cat in categories_list:
        seasonality = 'All Year'
        if cat in ['T恤', '裙子', '运动']:
            seasonality = 'Summer'
        elif cat in ['羽绒服/大衣', '大衣']:
            seasonality = 'Winter'
        elif cat in ['毛衣', '外套/夹克', '夹克']:
            seasonality = 'Autumn/Winter'
        
        categories.append({
            'id': cat,
            'name': cat,
            'seasonality': seasonality
        })
    
    # 如果没有品类，返回默认
    if not categories:
        categories = [
            {'id': 'T恤', 'name': 'T恤', 'seasonality': 'Summer'},
            {'id': '牛仔裤', 'name': '牛仔裤', 'seasonality': 'All Year'},
            {'id': '毛衣', 'name': '毛衣', 'seasonality': 'Winter'}
        ]
    
    return jsonify(categories)


# ============================================================================
# K-means 用户画像聚类接口
# ============================================================================

@app.route('/api/kmeans/clustering', methods=['GET', 'POST'])
def kmeans_clustering():
    """
    K-means 用户画像聚类接口
    
    Query Parameters (GET) or Body Parameters (POST):
    - n_clusters: 聚类数量 (可选，默认自动确定)
    
    Returns:
    - clusters: 用户群体列表
    - metrics: 聚类评估指标
    - optimal_k: 最优聚类数
    - user_assignments: 用户分配记录
    - feature_importance: 特征重要性
    """
    try:
        # 获取请求参数
        if request.method == 'POST':
            params = request.get_json() or {}
        else:
            params = request.args.to_dict()
        
        n_clusters = params.get('n_clusters')
        if n_clusters:
            n_clusters = int(n_clusters)
        
        logger.info(f"K-means clustering request: n_clusters={n_clusters}")
        
        # 运行聚类
        result = run_user_clustering(n_clusters=n_clusters)
        
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"Error in K-means clustering: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/kmeans/user/<customer_id>', methods=['GET'])
def get_user_cluster(customer_id: str):
    """
    获取指定用户的画像标签
    
    Path Parameters:
    - customer_id: 客户ID
    
    Returns:
    - customer_info: 客户信息
    - cluster_label: 所属群体标签
    - cluster_description: 群体描述
    """
    try:
        # 实际应用中应从数据库查询
        # 这里返回模拟数据
        cluster_labels = ['VIP高价值用户', '潜力用户', '活跃用户', '沉睡用户', '普通用户']
        
        result = {
            'customer_id': customer_id,
            'cluster_label': cluster_labels[hash(customer_id) % len(cluster_labels)],
            'cluster_description': '该用户群体具有较高的消费频次和客单价，是核心价值用户群体。',
            'total_orders': 28,
            'total_spend': 12580.00,
            'avg_order_value': 449.29,
            'purchase_frequency': 2.5,
            'favorite_category': 'T-Shirt',
            'last_purchase_date': '2024-03-15'
        }
        
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"Error getting user cluster: {e}")
        return jsonify({'error': str(e)}), 500


# ============================================================================
# Decision Tree 区域销售分析接口
# ============================================================================

@app.route('/api/decisiontree/analysis', methods=['GET', 'POST'])
def decision_tree_analysis():
    """
    Decision Tree 区域销售分析接口
    
    Query Parameters (GET) or Body Parameters (POST):
    - max_depth: 决策树最大深度 (可选，默认5)
    - min_samples_split: 最小分裂样本数 (可选，默认20)
    
    Returns:
    - metrics: 模型评估指标
    - feature_importance: 特征重要性
    - rules: 决策规则列表
    - insights: 铺货建议
    - prediction_sample: 预测示例
    """
    try:
        # 获取请求参数
        if request.method == 'POST':
            params = request.get_json() or {}
        else:
            params = request.args.to_dict()
        
        max_depth = int(params.get('max_depth', 5))
        min_samples_split = int(params.get('min_samples_split', 20))
        
        logger.info(f"Decision tree analysis request: max_depth={max_depth}")

        # 运行分析，传递参数
        result = run_decision_tree_analysis(max_depth=max_depth, min_samples_split=min_samples_split)
        
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"Error in decision tree analysis: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/decisiontree/insights', methods=['GET'])
def get_distribution_insights():
    """
    获取铺货建议列表
    
    Query Parameters:
    - region: 地区筛选 (可选)
    - season: 季节筛选 (可选)
    - priority: 优先级筛选 (可选: 高/中/低)
    
    Returns:
    - insights: 铺货建议列表
    """
    try:
        region = request.args.get('region')
        season = request.args.get('season')
        priority = request.args.get('priority')
        
        # 运行分析获取建议
        result = run_decision_tree_analysis()
        
        insights = result.get('insights', [])
        
        # 筛选
        if region:
            insights = [i for i in insights if i['region'] == region]
        if season:
            insights = [i for i in insights if i['season'] == season]
        if priority:
            insights = [i for i in insights if i['priority'] == priority]
        
        return jsonify({'insights': insights})
    
    except Exception as e:
        logger.error(f"Error getting insights: {e}")
        return jsonify({'error': str(e)}), 500


# ============================================================================
# Apriori 商品关联规则接口
# ============================================================================

@app.route('/api/apriori/analysis', methods=['GET', 'POST'])
def apriori_analysis():
    """
    Apriori 商品关联规则挖掘接口
    
    Query Parameters (GET) or Body Parameters (POST):
    - min_support: 最小支持度 (可选，默认0.01)
    - min_confidence: 最小置信度 (可选，默认0.5)
    
    Returns:
    - frequent_itemsets: 频繁项集
    - association_rules: 关联规则
    - min_support: 使用的最小支持度
    - min_confidence: 使用的最小置信度
    - total_transactions: 交易总数
    - execution_time: 执行时间
    """
    try:
        # 获取请求参数
        if request.method == 'POST':
            params = request.get_json() or {}
        else:
            params = request.args.to_dict()
        
        min_support = float(params.get('min_support', 0.01))
        min_confidence = float(params.get('min_confidence', 0.5))
        
        logger.info(f"Apriori analysis request: min_support={min_support}, min_confidence={min_confidence}")
        
        # 运行分析
        result = run_apriori_analysis(
            min_support=min_support,
            min_confidence=min_confidence
        )
        
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"Error in Apriori analysis: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/apriori/rules', methods=['GET'])
def get_association_rules():
    """
    获取关联规则列表
    
    Query Parameters:
    - min_lift: 最小提升度筛选 (可选)
    - limit: 返回数量限制 (可选，默认20)
    
    Returns:
    - rules: 关联规则列表
    """
    try:
        min_lift = request.args.get('min_lift', type=float)
        limit = int(request.args.get('limit', 20))
        
        # 运行分析
        result = run_apriori_analysis()
        
        rules = result.get('association_rules', [])
        
        # 筛选
        if min_lift:
            rules = [r for r in rules if r['lift'] >= min_lift]
        
        # 限制数量
        rules = rules[:limit]
        
        return jsonify({'rules': rules})
    
    except Exception as e:
        logger.error(f"Error getting association rules: {e}")
        return jsonify({'error': str(e)}), 500


# ============================================================================
# 数据聚合统计接口
# ============================================================================

@app.route('/api/stats/sales-by-date', methods=['GET'])
def get_sales_by_date():
    """
    获取按日期聚合的销售数据
    
    Query Parameters:
    - start_date: 开始日期 (可选)
    - end_date: 结束日期 (可选)
    - year: 年份筛选 (可选, 如 2023, 2024, 2025)
    
    Returns:
    - data: 日期销售数据列表
    """
    try:
        from data_utils import load_all_years, aggregate_by_date
        
        year = request.args.get('year', type=int)
        
        # 加载数据
        all_data = load_all_years()
        
        # 按年份筛选
        if year:
            all_data = all_data[all_data['year'] == year]
        
        # 按日期聚合
        daily_data = aggregate_by_date(all_data)
        
        # 转换为API响应格式
        data = []
        for _, row in daily_data.iterrows():
            date_str = row['order_date'].strftime('%Y-%m-%d') if hasattr(row['order_date'], 'strftime') else str(row['order_date'])
            data.append({
                'date': date_str,
                'orders': int(row['order_count']),
                'revenue': round(float(row['sales_amount']), 2),
                'profit': round(float(row['profit']), 2),
                'customers': int(row['customer_count']),
                'avg_order_value': round(float(row['sales_amount']) / max(row['order_count'], 1), 2)
            })
        
        return jsonify({'data': data})
    
    except Exception as e:
        logger.error(f"Error getting sales by date: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/stats/sales-by-region', methods=['GET'])
def get_sales_by_region():
    """
    获取按地区聚合的销售数据
    
    Query Parameters:
    - year: 年份筛选 (可选)
    
    Returns:
    - data: 地区销售数据列表
    """
    try:
        from data_utils import load_all_years, aggregate_by_region
        
        year = request.args.get('year', type=int)
        
        # 加载数据
        all_data = load_all_years()
        
        # 按年份筛选
        if year:
            all_data = all_data[all_data['year'] == year]
        
        # 按地区聚合
        region_data = aggregate_by_region(all_data)
        
        # 转换为API响应格式
        data = []
        for _, row in region_data.iterrows():
            data.append({
                'region': row['city'],
                'orders': int(row['order_count']),
                'revenue': round(float(row['sales_amount']), 2),
                'customers': int(row['customer_count']),
                'profit': round(float(row['profit']), 2)
            })
        
        return jsonify({'data': data})
    
    except Exception as e:
        logger.error(f"Error getting sales by region: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/stats/sales-by-category', methods=['GET'])
def get_sales_by_category():
    """
    获取按品类聚合的销售数据
    
    Query Parameters:
    - year: 年份筛选 (可选)
    
    Returns:
    - data: 品类销售数据列表
    """
    try:
        from data_utils import load_all_years, aggregate_by_category
        
        year = request.args.get('year', type=int)
        
        # 加载数据
        all_data = load_all_years()
        
        # 按年份筛选
        if year:
            all_data = all_data[all_data['year'] == year]
        
        # 按品类聚合
        category_data = aggregate_by_category(all_data)
        
        # 转换为API响应格式
        data = []
        for _, row in category_data.iterrows():
            data.append({
                'category': row['category'],
                'orders': int(row['order_count']),
                'revenue': round(float(row['sales_amount']), 2),
                'profit': round(float(row['profit']), 2),
                'avg_price': round(float(row['sales_amount']) / max(row['order_count'], 1), 2)
            })
        
        return jsonify({'data': data})
    
    except Exception as e:
        logger.error(f"Error getting sales by category: {e}")
        return jsonify({'error': str(e)}), 500


# ============================================================================
# 数据Schema接口
# ============================================================================

@app.route('/api/schema', methods=['GET'])
def get_data_schema():
    """
    获取数据Schema定义
    
    Returns:
    - schema: 数据Schema定义
    """
    from data_schema import (
        ORDER_SCHEMA,
        USER_BEHAVIOR_SCHEMA,
        SALES_TIMESERIES_SCHEMA,
        TRANSACTION_BASKET_SCHEMA,
        REGIONAL_SALES_SCHEMA
    )
    
    return jsonify({
        'order_schema': ORDER_SCHEMA,
        'user_behavior_schema': USER_BEHAVIOR_SCHEMA,
        'sales_timeseries_schema': SALES_TIMESERIES_SCHEMA,
        'transaction_basket_schema': TRANSACTION_BASKET_SCHEMA,
        'regional_sales_schema': REGIONAL_SALES_SCHEMA
    })


# ============================================================================
# 错误处理
# ============================================================================

@app.errorhandler(404)
def not_found(error):
    """404 错误处理"""
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    """500 错误处理"""
    return jsonify({'error': 'Internal server error'}), 500


# ============================================================================
# 数据筛选接口
# ============================================================================

@app.route('/api/filters', methods=['GET'])
def get_filter_options():
    """
    获取数据筛选选项
    
    Returns:
        - categories: 产品类别列表
        - cities: 城市列表
        - genders: 性别群体列表
        - age_groups: 年龄群体列表
        - channels: 渠道列表
        - years: 可用年份列表
    """
    try:
        from data_utils import get_all_data, get_categories, get_cities, get_genders, get_age_groups, get_channels
        
        all_data = get_all_data()
        
        return jsonify({
            'categories': get_categories(all_data),
            'cities': get_cities(all_data),
            'genders': get_genders(all_data),
            'age_groups': get_age_groups(all_data),
            'channels': get_channels(all_data),
            'years': [2023, 2024, 2025]
        })
    
    except Exception as e:
        logger.error(f"Error getting filter options: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/stats/sales-by-month', methods=['GET'])
def get_sales_by_month():
    """
    获取按月份聚合的销售数据
    
    Query Parameters:
    - year: 年份筛选 (可选)
    
    Returns:
    - data: 月份销售数据列表
    """
    try:
        from data_utils import load_all_years, aggregate_by_month
        
        year = request.args.get('year', type=int)
        
        all_data = load_all_years()
        
        if year:
            all_data = all_data[all_data['year'] == year]
        
        monthly_data = aggregate_by_month(all_data)
        
        data = []
        for _, row in monthly_data.iterrows():
            data.append({
                'month': row['year_month'],
                'orders': int(row['order_count']),
                'revenue': round(float(row['sales_amount']), 2),
                'profit': round(float(row['profit']), 2),
                'customers': int(row['customer_count'])
            })
        
        return jsonify({'data': data})
    
    except Exception as e:
        logger.error(f"Error getting sales by month: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/stats/sales-by-gender', methods=['GET'])
def get_sales_by_gender():
    """
    获取按性别群体聚合的销售数据
    
    Query Parameters:
    - year: 年份筛选 (可选)
    
    Returns:
    - data: 性别销售数据列表
    """
    try:
        from data_utils import load_all_years, aggregate_by_gender
        
        year = request.args.get('year', type=int)
        
        all_data = load_all_years()
        
        if year:
            all_data = all_data[all_data['year'] == year]
        
        gender_data = aggregate_by_gender(all_data)
        
        data = []
        for _, row in gender_data.iterrows():
            data.append({
                'gender': row['gender'],
                'orders': int(row['order_count']),
                'revenue': round(float(row['sales_amount']), 2),
                'profit': round(float(row['profit']), 2),
                'customers': int(row['customer_count'])
            })
        
        return jsonify({'data': data})
    
    except Exception as e:
        logger.error(f"Error getting sales by gender: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/stats/sales-by-age', methods=['GET'])
def get_sales_by_age():
    """
    获取按年龄群体聚合的销售数据
    
    Query Parameters:
    - year: 年份筛选 (可选)
    
    Returns:
    - data: 年龄群体销售数据列表
    """
    try:
        from data_utils import load_all_years, aggregate_by_age_group
        
        year = request.args.get('year', type=int)
        
        all_data = load_all_years()
        
        if year:
            all_data = all_data[all_data['year'] == year]
        
        age_data = aggregate_by_age_group(all_data)
        
        data = []
        for _, row in age_data.iterrows():
            data.append({
                'age_group': row['age_group'],
                'orders': int(row['order_count']),
                'revenue': round(float(row['sales_amount']), 2),
                'profit': round(float(row['profit']), 2),
                'customers': int(row['customer_count'])
            })
        
        return jsonify({'data': data})
    
    except Exception as e:
        logger.error(f"Error getting sales by age: {e}")
        return jsonify({'error': str(e)}), 500


# ============================================================================
# 入口函数
# ============================================================================

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    debug = os.environ.get('FLASK_DEBUG', 'True').lower() == 'true'
    
    print("\n" + "="*60)
    print("Uniqlo Analysis System - Backend API")
    print("="*60)
    print(f"Starting server on port {port}")
    print(f"Debug mode: {debug}")
    print("\nAvailable endpoints:")
    print("  - GET  /api/health              - Health check")
    print("  - GET  /api/overview            - Data overview")
    print("  - GET  /api/arima/forecast      - ARIMA sales forecast")
    print("  - GET  /api/kmeans/clustering   - K-means user clustering")
    print("  - GET  /api/decisiontree/analysis - Decision tree analysis")
    print("  - GET  /api/apriori/analysis    - Apriori association rules")
    print("  - GET  /api/stats/*             - Statistics endpoints")
    print("  - GET  /api/schema              - Data schema")
    print("="*60 + "\n")
    
    app.run(host='0.0.0.0', port=port, debug=debug)
