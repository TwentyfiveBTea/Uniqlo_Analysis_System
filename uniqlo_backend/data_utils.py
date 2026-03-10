#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据加载和处理工具模块
=====================
本模块提供基于pandas的数据加载、处理和转换功能，
用于适配实际的CSV算法模块

数据数据格式与各来源：
- 2023_uniqlo.csv
- 2024_uniqlo.csv
- 2025_uniqlo.csv

Author: Uniqlo Analysis System
"""

import os
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 数据目录
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data', 'raw')

# 中文列名到英文列名的映射
COLUMN_MAPPING = {
    '商店ID': 'store_id',
    '门店所在城市': 'city',
    '渠道': 'channel',
    '性别群体': 'gender',
    '年龄群体': 'age_group',
    '产品类别': 'category',
    '客户数量': 'customer_count',
    '销售金额': 'sales_amount',
    '订单数量': 'order_count',
    '购买的产品数量': 'product_count',
    '成本': 'cost',
    '单价': 'unit_price',
    '利润': 'profit',
    '订单日期': 'order_date',
    '星期': 'weekday'
}

# 英文列名到中文的映射（用于显示）
REVERSE_MAPPING = {v: k for k, v in COLUMN_MAPPING.items()}


def load_csv_data(filepath: str) -> pd.DataFrame:
    """
    从CSV文件加载订单数据
    
    Args:
        filepath: CSV文件路径
        
    Returns:
        pandas DataFrame
    """
    logger.info(f"Loading data from: {filepath}")
    
    if not os.path.exists(filepath):
        logger.error(f"File not found: {filepath}")
        return pd.DataFrame()
    
    try:
        df = pd.read_csv(filepath, encoding='utf-8')
        
        # 重命名列为英文
        df = df.rename(columns=COLUMN_MAPPING)
        
        # 解析日期
        df['order_date'] = pd.to_datetime(df['order_date'], format='%Y年%m月%d日', errors='coerce')
        
        # 提取年月
        df['year'] = df['order_date'].dt.year
        df['month'] = df['order_date'].dt.month
        df['day'] = df['order_date'].dt.day
        
        # 添加季节字段
        def get_season(month):
            if month in [3, 4, 5]:
                return '春季'
            elif month in [6, 7, 8]:
                return '夏季'
            elif month in [9, 10, 11]:
                return '秋季'
            else:
                return '冬季'
        
        df['season'] = df['month'].apply(get_season)
        
        # 确保数值列为数值类型
        numeric_cols = ['customer_count', 'sales_amount', 'order_count', 
                       'product_count', 'cost', 'unit_price', 'profit']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        logger.info(f"Loaded {len(df)} records")
        return df
        
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return pd.DataFrame()


def load_all_years() -> pd.DataFrame:
    """
    加载所有年份的数据
    
    Returns:
        合并后的DataFrame
    """
    all_dfs = []
    
    for year in [2023, 2024, 2025]:
        filepath = os.path.join(DATA_DIR, f'{year}_uniqlo.csv')
        if os.path.exists(filepath):
            df = load_csv_data(filepath)
            if not df.empty:
                all_dfs.append(df)
                logger.info(f"Loaded {len(df)} records from {year}")
        else:
            logger.warning(f"File not found: {filepath}")
    
    if all_dfs:
        result = pd.concat(all_dfs, ignore_index=True)
        logger.info(f"Total loaded: {len(result)} records")
        return result
    else:
        return pd.DataFrame()


def load_year_data(year: int) -> pd.DataFrame:
    """
    加载指定年份的数据
    
    Args:
        year: 年份 (2023, 2024, 2025)
        
    Returns:
        指定年份的DataFrame
    """
    filepath = os.path.join(DATA_DIR, f'{year}_uniqlo.csv')
    return load_csv_data(filepath)


def get_unique_values(df: pd.DataFrame, field: str) -> List[str]:
    """
    获取某字段的所有唯一值
    
    Args:
        df: DataFrame
        field: 字段名 (category, city, gender, age_group, channel)
        
    Returns:
        唯一值列表
    """
    if df.empty or field not in df.columns:
        return []
    return sorted(df[field].dropna().unique().tolist())


def aggregate_by_date(df: pd.DataFrame) -> pd.DataFrame:
    """
    按日期聚合销售数据
    
    Args:
        df: 订单DataFrame
        
    Returns:
        聚合后的DataFrame
    """
    if df.empty:
        return pd.DataFrame()
    
    agg_df = df.groupby('order_date').agg({
        'sales_amount': 'sum',
        'order_count': 'sum',
        'customer_count': 'sum',
        'profit': 'sum',
        'product_count': 'sum'
    }).reset_index()
    
    return agg_df


def aggregate_by_category(df: pd.DataFrame) -> pd.DataFrame:
    """
    按产品类别聚合销售数据
    
    Args:
        df: 订单DataFrame
        
    Returns:
        聚合后的DataFrame
    """
    if df.empty:
        return pd.DataFrame()
    
    agg_df = df.groupby('category').agg({
        'sales_amount': 'sum',
        'order_count': 'sum',
        'customer_count': 'sum',
        'profit': 'sum'
    }).reset_index()
    
    return agg_df.sort_values('sales_amount', ascending=False)


def aggregate_by_region(df: pd.DataFrame) -> pd.DataFrame:
    """
    按地区聚合销售数据
    
    Args:
        df: 订单DataFrame
        
    Returns:
        聚合后的DataFrame
    """
    if df.empty:
        return pd.DataFrame()
    
    agg_df = df.groupby('city').agg({
        'sales_amount': 'sum',
        'order_count': 'sum',
        'customer_count': 'sum',
        'profit': 'sum'
    }).reset_index()
    
    return agg_df.sort_values('sales_amount', ascending=False)


def aggregate_by_month(df: pd.DataFrame) -> pd.DataFrame:
    """
    按月份聚合销售数据
    
    Args:
        df: 订单DataFrame
        
    Returns:
        聚合后的DataFrame
    """
    if df.empty:
        return pd.DataFrame()
    
    df_copy = df.copy()
    df_copy['year_month'] = df_copy['order_date'].dt.to_period('M')
    
    agg_df = df_copy.groupby('year_month').agg({
        'sales_amount': 'sum',
        'order_count': 'sum',
        'customer_count': 'sum',
        'profit': 'sum'
    }).reset_index()
    
    agg_df['year_month'] = agg_df['year_month'].astype(str)
    
    return agg_df


def aggregate_by_gender(df: pd.DataFrame) -> pd.DataFrame:
    """
    按性别群体聚合
    
    Args:
        df: 订单DataFrame
        
    Returns:
        聚合后的DataFrame
    """
    if df.empty:
        return pd.DataFrame()
    
    agg_df = df.groupby('gender').agg({
        'sales_amount': 'sum',
        'order_count': 'sum',
        'customer_count': 'sum',
        'profit': 'sum'
    }).reset_index()
    
    return agg_df.sort_values('sales_amount', ascending=False)


def aggregate_by_age_group(df: pd.DataFrame) -> pd.DataFrame:
    """
    按年龄群体聚合
    
    Args:
        df: 订单DataFrame
        
    Returns:
        聚合后的DataFrame
    """
    if df.empty:
        return pd.DataFrame()
    
    agg_df = df.groupby('age_group').agg({
        'sales_amount': 'sum',
        'order_count': 'sum',
        'customer_count': 'sum',
        'profit': 'sum'
    }).reset_index()
    
    return agg_df.sort_values('sales_amount', ascending=False)


def get_sales_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    获取销售汇总统计
    
    Args:
        df: 订单DataFrame
        
    Returns:
        汇总统计数据
    """
    if df.empty:
        return {
            'total_orders': 0,
            'total_revenue': 0,
            'total_customers': 0,
            'total_profit': 0,
            'avg_order_value': 0
        }
    
    total_revenue = df['sales_amount'].sum()
    total_orders = df['order_count'].sum()
    total_customers = df['customer_count'].sum()
    total_profit = df['profit'].sum()
    
    return {
        'total_orders': int(total_orders),
        'total_revenue': round(total_revenue, 2),
        'total_customers': int(total_customers),
        'total_profit': round(total_profit, 2),
        'avg_order_value': round(total_revenue / total_orders, 2) if total_orders > 0 else 0
    }


def filter_data(
    df: pd.DataFrame,
    year: Optional[int] = None,
    category: Optional[str] = None,
    city: Optional[str] = None,
    gender: Optional[str] = None,
    age_group: Optional[str] = None,
    channel: Optional[str] = None
) -> pd.DataFrame:
    """
    根据条件筛选数据
    
    Args:
        df: 订单DataFrame
        year: 年份筛选
        category: 产品类别筛选
        city: 城市筛选
        gender: 性别筛选
        age_group: 年龄群体筛选
        channel: 渠道筛选
        
    Returns:
        筛选后的DataFrame
    """
    result = df.copy()
    
    if year is not None:
        result = result[result['year'] == year]
    
    if category is not None:
        result = result[result['category'] == category]
    
    if city is not None:
        result = result[result['city'] == city]
    
    if gender is not None:
        result = result[result['gender'] == gender]
    
    if age_group is not None:
        result = result[result['age_group'] == age_group]
    
    if channel is not None:
        result = result[result['channel'] == channel]
    
    return result


def get_categories(df: pd.DataFrame) -> List[str]:
    """获取所有产品类别"""
    return get_unique_values(df, 'category')


def get_cities(df: pd.DataFrame) -> List[str]:
    """获取所有城市"""
    return get_unique_values(df, 'city')


def get_genders(df: pd.DataFrame) -> List[str]:
    """获取所有性别群体"""
    return get_unique_values(df, 'gender')


def get_age_groups(df: pd.DataFrame) -> List[str]:
    """获取所有年龄群体"""
    return get_unique_values(df, 'age_group')


def get_channels(df: pd.DataFrame) -> List[str]:
    """获取所有渠道"""
    return get_unique_values(df, 'channel')


# ============================================================================
# 便捷函数
# ============================================================================

def get_all_data() -> pd.DataFrame:
    """获取所有年份的数据"""
    return load_all_years()


def get_data_for_year(year: int) -> pd.DataFrame:
    """获取指定年份的数据"""
    return load_year_data(year)


def get_overview() -> Dict[str, Any]:
    """
    获取数据概览
    
    Returns:
        概览数据
    """
    all_data = load_all_years()
    
    # 按年份汇总
    yearly_data = {}
    for year in [2023, 2024, 2025]:
        year_df = all_data[all_data['year'] == year]
        if not year_df.empty:
            yearly_data[str(year)] = get_sales_summary(year_df)
    
    # 按类别汇总
    category_summary = aggregate_by_category(all_data)
    top_categories = category_summary.head(5).to_dict('records')
    
    # 按地区汇总
    region_summary = aggregate_by_region(all_data)
    top_regions = region_summary.head(5).to_dict('records')
    
    return {
        'total_orders': int(all_data['order_count'].sum()),
        'total_revenue': round(all_data['sales_amount'].sum(), 2),
        'total_customers': int(all_data['customer_count'].sum()),
        'total_profit': round(all_data['profit'].sum(), 2),
        'yearly_summary': yearly_data,
        'top_categories': top_categories,
        'top_regions': top_regions
    }


if __name__ == "__main__":
    # 测试数据加载
    print("测试数据加载...")
    
    all_data = get_all_data()
    print(f"总记录数: {len(all_data)}")
    
    summary = get_sales_summary(all_data)
    print(f"\n销售汇总: {summary}")
    
    print(f"\n产品类别: {get_categories(all_data)}")
    print(f"城市: {get_cities(all_data)}")
    print(f"性别群体: {get_genders(all_data)}")
    print(f"年龄群体: {get_age_groups(all_data)}")
