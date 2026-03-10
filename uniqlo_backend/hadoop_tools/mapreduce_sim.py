#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MapReduce Simulation Module - 模拟 MapReduce 离线数据汇总
============================================================
本模块模拟 Apache Hadoop MapReduce 的离线数据汇总功能

Related to Research: Chapter 3 - Data Processing and Storage
研究内容：基于 Hadoop 的数据聚合与统计分析

功能说明：
1. 按区域聚合：统计各地区销售数据
2. 按日期聚合：统计每日销售趋势
3. 按品类聚合：统计各品类销售情况
4. 多维度聚合：区域+日期+品类组合统计
5. 输出聚合结果供后续分析使用

注意：本脚本为模拟实现，实际生产环境需对接真实 MapReduce 组件
"""

import os
import json
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from collections import defaultdict
from dataclasses import dataclass, asdict
import pandas as pd
import numpy as np

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class AggregationResult:
    """聚合结果数据类"""
    dimension: str
    groups: List[Dict[str, Any]]
    total_records: int
    execution_time: float
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class MapReduceSimulator:
    """
    MapReduce 模拟器
    
    模拟 MapReduce 的核心功能：
    - Map 阶段：数据分组映射
    - Reduce 阶段：数据聚合汇总
    - 支持多种聚合维度
    """
    
    def __init__(self, input_dir: str = "./data/cleaned", output_dir: str = "./data/aggregated"):
        """
        初始化 MapReduce 模拟器
        
        Args:
            input_dir: 清洗后数据输入目录
            output_dir: 聚合结果输出目录
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"MapReduceSimulator initialized: {input_dir} -> {output_dir}")
    
    def load_cleaned_data(self, filepath: str = None) -> pd.DataFrame:
        """
        加载清洗后的数据
        
        Args:
            filepath: 数据文件路径，若为 None 则加载目录下所有 Parquet 文件
        
        Returns:
            pandas DataFrame
        """
        logger.info("Loading cleaned data...")
        
        if filepath:
            if filepath.endswith('.parquet'):
                df = pd.read_parquet(filepath)
            elif filepath.endswith('.csv'):
                df = pd.read_csv(filepath)
            else:
                df = pd.read_json(filepath)
        else:
            # 加载目录下所有 Parquet 文件
            all_files = []
            for root, dirs, files in os.walk(self.input_dir):
                for file in files:
                    if file.endswith('.parquet'):
                        all_files.append(os.path.join(root, file))
            
            if not all_files:
                logger.warning(f"No parquet files found in {self.input_dir}")
                return pd.DataFrame()
            
            df = pd.concat([pd.read_parquet(f) for f in all_files], ignore_index=True)
        
        logger.info(f"Loaded {len(df)} records")
        return df
    
    # =========================================================================
    # Map 函数 - 数据分组映射
    # =========================================================================
    
    def map_by_region(self, record: Dict[str, Any]) -> tuple:
        """
        Map: 按区域分组
        
        Args:
            record: 单条订单记录
        
        Returns:
            (region, record)
        """
        region = record.get('Region', 'Unknown')
        return (region, record)
    
    def map_by_date(self, record: Dict[str, Any]) -> tuple:
        """
        Map: 按日期分组
        
        Args:
            record: 单条订单记录
        
        Returns:
            (date, record)
        """
        date = record.get('Order_Date', 'Unknown')
        return (date, record)
    
    def map_by_category(self, record: Dict[str, Any]) -> tuple:
        """
        Map: 按品类分组
        
        Args:
            record: 单条订单记录
        
        Returns:
            (category, record)
        """
        category = record.get('Category', 'Unknown')
        return (category, record)
    
    def map_by_region_date(self, record: Dict[str, Any]) -> tuple:
        """
        Map: 按区域+日期分组
        
        Args:
            record: 单条订单记录
        
        Returns:
            (region_date_key, record)
        """
        region = record.get('Region', 'Unknown')
        date = record.get('Order_Date', 'Unknown')
        key = f"{region}_{date}"
        return (key, record)
    
    def map_by_region_category(self, record: Dict[str, Any]) -> tuple:
        """
        Map: 按区域+品类分组
        
        Args:
            record: 单条订单记录
        
        Returns:
            (region_category_key, record)
        """
        region = record.get('Region', 'Unknown')
        category = record.get('Category', 'Unknown')
        key = f"{region}_{category}"
        return (key, record)
    
    # =========================================================================
    # Reduce 函数 - 数据聚合汇总
    # =========================================================================
    
    def reduce_sales(self, key: str, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Reduce: 销售数据聚合
        
        Args:
            key: 分组键
            records: 该分组下的所有记录
        
        Returns:
            聚合结果
        """
        total_amount = sum(r.get('Total_Amount', 0) for r in records)
        total_quantity = sum(r.get('Quantity', 0) for r in records)
        order_count = len(records)
        
        return {
            'key': key,
            'total_orders': order_count,
            'total_quantity': int(total_quantity),
            'total_amount': float(total_amount),
            'avg_order_value': float(total_amount / order_count) if order_count > 0 else 0
        }
    
    # =========================================================================
    # 高级聚合功能
    # =========================================================================
    
    def aggregate_by_dimensions(self, df: pd.DataFrame, 
                                dimensions: List[str]) -> pd.DataFrame:
        """
        多维度聚合
        
        Args:
            df: 输入数据框
            dimensions: 聚合维度列表
        
        Returns:
            聚合结果数据框
        """
        logger.info(f"Aggregating by dimensions: {dimensions}")
        
        # 构建聚合表达式
        agg_dict = {
            'Order_ID': 'count',
            'Quantity': 'sum',
            'Total_Amount': 'sum'
        }
        
        # 执行分组聚合
        result = df.groupby(dimensions).agg(agg_dict).reset_index()
        
        # 重命名列
        result.rename(columns={
            'Order_ID': 'order_count',
            'Quantity': 'total_quantity',
            'Total_Amount': 'total_amount'
        }, inplace=True)
        
        # 计算平均订单金额
        result['avg_order_value'] = result['total_amount'] / result['order_count']
        
        logger.info(f"Aggregation complete: {len(result)} groups")
        
        return result
    
    # =========================================================================
    # 预定义的聚合任务
    # =========================================================================
    
    def aggregate_by_region(self, df: pd.DataFrame) -> AggregationResult:
        """
        按区域聚合销售数据
        
        Args:
            df: 输入数据框
        
        Returns:
            聚合结果
        """
        start_time = datetime.now()
        
        # 按 Region 分组聚合
        result = self.aggregate_by_dimensions(df, ['Region'])
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        return AggregationResult(
            dimension='Region',
            groups=result.to_dict('records'),
            total_records=len(df),
            execution_time=execution_time
        )
    
    def aggregate_by_date(self, df: pd.DataFrame) -> AggregationResult:
        """
        按日期聚合销售数据
        
        Args:
            df: 输入数据框
        
        Returns:
            聚合结果
        """
        start_time = datetime.now()
        
        # 按 Order_Date 分组聚合
        result = self.aggregate_by_dimensions(df, ['Order_Date'])
        
        # 按日期排序
        result = result.sort_values('Order_Date')
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        return AggregationResult(
            dimension='Date',
            groups=result.to_dict('records'),
            total_records=len(df),
            execution_time=execution_time
        )
    
    def aggregate_by_category(self, df: pd.DataFrame) -> AggregationResult:
        """
        按品类聚合销售数据
        
        Args:
            df: 输入数据框
        
        Returns:
            聚合结果
        """
        start_time = datetime.now()
        
        # 按 Category 分组聚合
        result = self.aggregate_by_dimensions(df, ['Category'])
        
        # 按销售额排序
        result = result.sort_values('total_amount', ascending=False)
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        return AggregationResult(
            dimension='Category',
            groups=result.to_dict('records'),
            total_records=len(df),
            execution_time=execution_time
        )
    
    def aggregate_by_region_date(self, df: pd.DataFrame) -> AggregationResult:
        """
        按区域+日期聚合销售数据
        
        Args:
            df: 输入数据框
        
        Returns:
            聚合结果
        """
        start_time = datetime.now()
        
        # 按 Region + Order_Date 分组聚合
        result = self.aggregate_by_dimensions(df, ['Region', 'Order_Date'])
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        return AggregationResult(
            dimension='Region_Date',
            groups=result.to_dict('records'),
            total_records=len(df),
            execution_time=execution_time
        )
    
    def aggregate_by_region_category(self, df: pd.DataFrame) -> AggregationResult:
        """
        按区域+品类聚合销售数据
        
        Args:
            df: 输入数据框
        
        Returns:
            聚合结果
        """
        start_time = datetime.now()
        
        # 按 Region + Category 分组聚合
        result = self.aggregate_by_dimensions(df, ['Region', 'Category'])
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        return AggregationResult(
            dimension='Region_Category',
            groups=result.to_dict('records'),
            total_records=len(df),
            execution_time=execution_time
        )
    
    def aggregate_by_season_category(self, df: pd.DataFrame) -> AggregationResult:
        """
        按季节+品类聚合销售数据
        
        Args:
            df: 输入数据框
        
        Returns:
            聚合结果
        """
        start_time = datetime.now()
        
        # 按 Season + Category 分组聚合
        result = self.aggregate_by_dimensions(df, ['Season', 'Category'])
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        return AggregationResult(
            dimension='Season_Category',
            groups=result.to_dict('records'),
            total_records=len(df),
            execution_time=execution_time
        )
    
    # =========================================================================
    # 输出功能
    # =========================================================================
    
    def save_aggregation_result(self, result: AggregationResult, 
                                  filename: str = None) -> str:
        """
        保存聚合结果
        
        Args:
            result: 聚合结果对象
            filename: 输出文件名
        
        Returns:
            输出文件路径
        """
        if filename is None:
            filename = f"agg_{result.dimension.lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        output_path = os.path.join(self.output_dir, filename)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result.to_dict(), f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved aggregation result: {output_path}")
        
        return output_path
    
    def save_all_results(self, results: List[AggregationResult]) -> List[str]:
        """
        保存所有聚合结果
        
        Args:
            results: 聚合结果列表
        
        Returns:
            输出文件路径列表
        """
        output_files = []
        
        for result in results:
            filepath = self.save_aggregation_result(result)
            output_files.append(filepath)
        
        return output_files


def run_mapreduce_pipeline(input_file: str = None, 
                           output_dir: str = "./data/aggregated") -> Dict[str, Any]:
    """
    运行完整的 MapReduce 聚合管道
    
    Args:
        input_file: 输入文件路径
        output_dir: 输出目录
    
    Returns:
        处理结果信息
    """
    mr = MapReduceSimulator(output_dir=output_dir)
    
    # 加载数据
    df = mr.load_cleaned_data(input_file)
    
    if df.empty:
        logger.warning("No data to aggregate")
        return {'error': 'No data available'}
    
    # 执行各种聚合任务
    results = []
    
    # 1. 按区域聚合
    region_result = mr.aggregate_by_region(df)
    results.append(region_result)
    
    # 2. 按日期聚合
    date_result = mr.aggregate_by_date(df)
    results.append(date_result)
    
    # 3. 按品类聚合
    category_result = mr.aggregate_by_category(df)
    results.append(category_result)
    
    # 4. 按区域+日期聚合
    region_date_result = mr.aggregate_by_region_date(df)
    results.append(region_date_result)
    
    # 5. 按区域+品类聚合
    region_category_result = mr.aggregate_by_region_category(df)
    results.append(region_category_result)
    
    # 6. 按季节+品类聚合
    season_category_result = mr.aggregate_by_season_category(df)
    results.append(season_category_result)
    
    # 保存所有结果
    output_files = mr.save_all_results(results)
    
    return {
        'input_records': len(df),
        'aggregations': len(results),
        'output_files': output_files,
        'summary': {
            r.dimension: {
                'groups': len(r.groups),
                'execution_time': r.execution_time
            }
            for r in results
        }
    }


# ============================================================================
# 入口函数 - 供外部调用
# ============================================================================

def main():
    """主函数 - 测试 MapReduce 聚合流程"""
    
    # 检查是否有清洗后的数据
    test_data_file = "./data/cleaned/orders_cleaned_20240315.parquet"
    
    # 如果没有测试数据，创建模拟数据
    if not os.path.exists(test_data_file):
        logger.info("Creating sample data for testing...")
        os.makedirs("./data/cleaned", exist_ok=True)
        
        # 创建样本数据
        sample_data = []
        regions = ['Shanghai', 'Beijing', 'Guangzhou', 'Shenzhen', 'Hangzhou']
        categories = ['T-Shirt', 'Pants', 'Jacket', 'Coat', 'Dress']
        seasons = ['Spring', 'Summer', 'Autumn', 'Winter']
        
        for i in range(1000):
            sample_data.append({
                'Order_ID': f"ORD-202403{i%30+1:02d}-{i:05d}",
                'Source': 'POS',
                'Region': regions[i % len(regions)],
                'Order_Date': f"2024-03-{(i % 30) + 1:02d}",
                'Season': seasons[i % len(seasons)],
                'Category': categories[i % len(categories)],
                'Product_Code': f"SKU-{i % 100:05d}",
                'Quantity': (i % 5) + 1,
                'Unit_Price': (i % 50 + 50) * 2.0,
                'Discount': (i % 10) * 5.0,
                'Total_Amount': ((i % 50 + 50) * 2.0 - (i % 10) * 5.0) * ((i % 5) + 1)
            })
        
        df = pd.DataFrame(sample_data)
        df.to_parquet(test_data_file, index=False)
    
    # 运行聚合管道
    result = run_mapreduce_pipeline(test_data_file)
    
    print(f"\n{'='*60}")
    print("MapReduce Aggregation Complete")
    print(f"{'='*60}")
    print(f"Input records: {result.get('input_records', 'N/A')}")
    print(f"Aggregations performed: {result.get('aggregations', 'N/A')}")
    print(f"Output files: {result.get('output_files', [])}")
    
    if 'summary' in result:
        print("\nSummary:")
        for dim, stats in result['summary'].items():
            print(f"  {dim}: {stats['groups']} groups, {stats['execution_time']:.3f}s")


if __name__ == "__main__":
    main()
