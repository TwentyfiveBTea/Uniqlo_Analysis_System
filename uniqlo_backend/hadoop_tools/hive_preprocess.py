#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hive Preprocess Module - 模拟 Hive 数据清洗与 Parquet 转换
============================================================
本模块模拟 Apache Hive 的数据清洗逻辑，处理异常值并将数据转换为 Parquet 格式存储

Related to Research: Chapter 3 - Data Processing and Storage
研究内容：基于 Hadoop 的数据清洗与预处理

功能说明：
1. 数据质量检查与异常值检测
2. 缺失值处理
3. 数据类型转换与标准化
4. 数据去重
5. 输出 Parquet 格式文件

注意：本脚本为模拟实现，实际生产环境需对接真实 Hive/ Spark 组件
"""

import os
import json
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
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
class DataQualityReport:
    """数据质量报告"""
    total_records: int = 0
    valid_records: int = 0
    dropped_records: int = 0
    missing_values: Dict[str, int] = None
    outliers: Dict[str, int] = None
    duplicates: int = 0
    
    def __post_init__(self):
        if self.missing_values is None:
            self.missing_values = {}
        if self.outliers is None:
            self.outliers = {}
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class HivePreprocessor:
    """
    Hive 数据预处理模拟器
    
    模拟 Hive 的数据清洗功能：
    - 数据质量检查
    - 异常值检测与处理
    - 缺失值填充
    - 数据类型转换
    - Parquet 格式输出
    """
    
    # 数据质量规则配置
    QUALITY_RULES = {
        'Quantity': {'min': 1, 'max': 100, 'type': 'int'},
        'Unit_Price': {'min': 0, 'max': 10000, 'type': 'float'},
        'Discount': {'min': 0, 'max': 1000, 'type': 'float'},
        'Total_Amount': {'min': 0, 'max': 100000, 'type': 'float'},
    }
    
    # 必填字段
    REQUIRED_FIELDS = [
        'Order_ID', 'Source', 'Region', 'Order_Date', 
        'Season', 'Category', 'Product_Code'
    ]
    
    # 枚举字段
    ENUM_FIELDS = {
        'Source': ['ERP', 'POS', 'ONLINE_EC', 'APP', 'MINI_PROGRAM'],
        'Season': ['Spring', 'Summer', 'Autumn', 'Winter'],
        'Category': [
            'T-Shirt', 'Shirt', 'Pants', 'Jacket', 'Coat', 
            'Dress', 'Skirt', 'Sweater', 'Hoodie', 'Innerwear',
            'Accessories', 'Shoes', 'Bags'
        ],
        'Size': ['XS', 'S', 'M', 'L', 'XL', 'XXL', 'Free'],
        'Age_Group': ['<18', '18-25', '26-35', '36-45', '46-55', '>55'],
        'Gender': ['Male', 'Female', 'Unknown']
    }
    
    def __init__(self, input_dir: str = "./data/raw", output_dir: str = "./data/cleaned"):
        """
        初始化 Hive 预处理器
        
        Args:
            input_dir: 原始数据输入目录
            output_dir: 清洗后数据输出目录
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.quality_report = DataQualityReport()
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"HivePreprocessor initialized: {input_dir} -> {output_dir}")
    
    def load_raw_data(self, filepath: str) -> pd.DataFrame:
        """
        加载原始数据
        
        Args:
            filepath: 原始数据文件路径
        
        Returns:
            pandas DataFrame
        """
        logger.info(f"Loading raw data from: {filepath}")
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
        
        # 根据文件类型加载
        if filepath.endswith('.json'):
            df = pd.read_json(filepath)
        elif filepath.endswith('.csv'):
            df = pd.read_csv(filepath)
        elif filepath.endswith('.parquet'):
            df = pd.read_parquet(filepath)
        else:
            raise ValueError(f"Unsupported file format: {filepath}")
        
        logger.info(f"Loaded {len(df)} records")
        return df
    
    def check_missing_values(self, df: pd.DataFrame) -> Dict[str, int]:
        """
        检查缺失值
        
        Args:
            df: 输入数据框
        
        Returns:
            各字段缺失值统计
        """
        missing = {}
        for col in df.columns:
            missing_count = df[col].isna().sum()
            if missing_count > 0:
                missing[col] = int(missing_count)
        
        self.quality_report.missing_values = missing
        return missing
    
    def check_outliers(self, df: pd.DataFrame) -> Dict[str, int]:
        """
        检测异常值 (基于业务规则)
        
        Args:
            df: 输入数据框
        
        Returns:
            各字段异常值统计
        """
        outliers = {}
        
        for field, rules in self.QUALITY_RULES.items():
            if field in df.columns:
                if rules['type'] == 'int':
                    # 数值型字段异常检测
                    invalid_count = ((df[field] < rules['min']) | 
                                    (df[field] > rules['max'])).sum()
                else:
                    invalid_count = 0
                
                if invalid_count > 0:
                    outliers[field] = int(invalid_count)
        
        self.quality_report.outliers = outliers
        return outliers
    
    def validate_enum_fields(self, df: pd.DataFrame) -> int:
        """
        验证枚举字段有效性
        
        Args:
            df: 输入数据框
        
        Returns:
            无效记录数
        """
        invalid_count = 0
        
        for field, valid_values in self.ENUM_FIELDS.items():
            if field in df.columns:
                # 检查非空值是否在枚举范围内
                mask = df[field].notna() & ~df[field].isin(valid_values)
                invalid_count += mask.sum()
        
        return int(invalid_count)
    
    def clean_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, DataQualityReport]:
        """
        执行完整的数据清洗流程
        
        Args:
            df: 原始数据框
        
        Returns:
            清洗后的数据框, 数据质量报告
        """
        logger.info("Starting data cleaning process...")
        
        self.quality_report.total_records = len(df)
        original_count = len(df)
        
        # Step 1: 检查并记录缺失值
        missing = self.check_missing_values(df)
        logger.info(f"Missing values: {missing}")
        
        # Step 2: 处理必填字段缺失 - 删除这些记录
        for field in self.REQUIRED_FIELDS:
            if field in df.columns:
                df = df[df[field].notna()]
        
        dropped = original_count - len(df)
        self.quality_report.dropped_records = dropped
        logger.info(f"Dropped records due to missing required fields: {dropped}")
        
        # Step 3: 处理数值字段异常值
        outliers = self.check_outliers(df)
        logger.info(f"Outliers detected: {outliers}")
        
        for field, rules in self.QUALITY_RULES.items():
            if field in df.columns:
                # 将异常值设为 NaN，后续填充
                df.loc[(df[field] < rules['min']) | 
                       (df[field] > rules['max']), field] = np.nan
        
        # Step 4: 填充缺失值
        # 数值型字段用中位数填充
        numeric_fields = ['Quantity', 'Unit_Price', 'Discount', 'Total_Amount']
        for field in numeric_fields:
            if field in df.columns:
                median_val = df[field].median()
                df[field] = df[field].fillna(median_val)
        
        # 枚举字段用 'Unknown' 填充
        for field in self.ENUM_FIELDS.keys():
            if field in df.columns:
                df[field] = df[field].fillna('Unknown')
        
        # Step 5: 数据类型转换
        df = self._normalize_data_types(df)
        
        # Step 6: 去重
        before_dedup = len(df)
        df = df.drop_duplicates(subset=['Order_ID'], keep='first')
        self.quality_report.duplicates = before_dedup - len(df)
        
        # Step 7: 添加清洗时间戳
        df['cleaned_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        self.quality_report.valid_records = len(df)
        
        logger.info(f"Data cleaning complete: {len(df)} valid records")
        
        return df, self.quality_report
    
    def _normalize_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        标准化数据类型
        
        Args:
            df: 输入数据框
        
        Returns:
            标准化后的数据框
        """
        # 日期字段转换
        date_fields = ['Order_Date', 'Last_Purchase_Date', 'Transaction_Date']
        for field in date_fields:
            if field in df.columns:
                df[field] = pd.to_datetime(df[field], errors='coerce')
                df[field] = df[field].dt.strftime('%Y-%m-%d')
        
        # 数值字段确保正确类型
        numeric_fields = ['Quantity', 'Unit_Price', 'Discount', 'Total_Amount',
                         'Sales_Volume', 'Sales_Amount']
        for field in numeric_fields:
            if field in df.columns:
                df[field] = pd.to_numeric(df[field], errors='coerce')
        
        return df
    
    def save_to_parquet(self, df: pd.DataFrame, filename: str = None) -> str:
        """
        保存为 Parquet 格式
        
        Args:
            df: 数据框
            filename: 输出文件名
        
        Returns:
            输出文件路径
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"orders_cleaned_{timestamp}.parquet"
        
        output_path = os.path.join(self.output_dir, filename)
        
        # 保存为 Parquet 格式
        df.to_parquet(output_path, index=False, engine='pyarrow')
        
        logger.info(f"Saved {len(df)} records to Parquet: {output_path}")
        
        return output_path
    
    def save_quality_report(self, report: DataQualityReport, filename: str = None) -> str:
        """
        保存数据质量报告
        
        Args:
            report: 质量报告对象
            filename: 报告文件名
        
        Returns:
            报告文件路径
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"quality_report_{timestamp}.json"
        
        output_path = os.path.join(self.output_dir, filename)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report.to_dict(), f, ensure_ascii=False, indent=2)
        
        logger.info(f"Quality report saved: {output_path}")
        
        return output_path


def run_hive_pipeline(input_file: str, output_dir: str = "./data/cleaned") -> Dict[str, Any]:
    """
    运行完整的 Hive 数据清洗管道
    
    Args:
        input_file: 输入文件路径
        output_dir: 输出目录
    
    Returns:
        处理结果信息
    """
    preprocessor = HivePreprocessor(output_dir=output_dir)
    
    # 加载数据
    df = preprocessor.load_raw_data(input_file)
    
    # 清洗数据
    cleaned_df, report = preprocessor.clean_data(df)
    
    # 保存清洗后的数据
    output_file = preprocessor.save_to_parquet(cleaned_df)
    
    # 保存质量报告
    report_file = preprocessor.save_quality_report(report)
    
    return {
        'input_file': input_file,
        'output_file': output_file,
        'quality_report': report_file,
        'stats': report.to_dict()
    }


# ============================================================================
# 入口函数 - 供外部调用
# ============================================================================

def main():
    """主函数 - 测试数据清洗流程"""
    
    # 测试数据
    test_data = [
        {
            "Order_ID": "ORD-20240315-000001",
            "Source": "POS",
            "Region": "Shanghai",
            "Order_Date": "2024-03-15",
            "Season": "Spring",
            "Category": "T-Shirt",
            "Product_Code": "SKU-123456",
            "Quantity": 2,
            "Unit_Price": 99.00,
            "Discount": 10.00,
            "Total_Amount": 188.00,
        },
        {
            "Order_ID": "ORD-20240315-000002",
            "Source": "INVALID_SOURCE",  # 异常值测试
            "Region": "Beijing",
            "Order_Date": "2024-03-15",
            "Season": "Spring",
            "Category": "Pants",
            "Product_Code": "SKU-234567",
            "Quantity": 1,
            "Unit_Price": 199.00,
            "Discount": 0.00,
            "Total_Amount": 199.00,
        }
    ]
    
    # 创建测试数据文件
    os.makedirs("./data/raw", exist_ok=True)
    test_file = "./data/raw/test_orders.json"
    with open(test_file, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, ensure_ascii=False)
    
    # 运行清洗管道
    result = run_hive_pipeline(test_file, output_dir="./data/cleaned")
    
    print(f"\n{'='*60}")
    print("Hive Data Preprocessing Complete")
    print(f"{'='*60}")
    print(f"Input: {result['input_file']}")
    print(f"Output: {result['output_file']}")
    print(f"Quality Report: {result['quality_report']}")
    print(f"Stats: {json.dumps(result['stats'], indent=2)}")


if __name__ == "__main__":
    main()
