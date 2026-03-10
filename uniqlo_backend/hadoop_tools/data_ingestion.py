#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data Ingestion Module - 模拟 Flume 数据抓取
==========================================
本模块模拟 Apache Flume 从多源订单数据系统抓取数据并存入 HDFS 的逻辑

Related to Research: Chapter 3 - Data Sources and Collection
研究内容：多源订单数据采集与预处理

功能说明：
1. 支持从 ERP、POS、线上电商平台等多数据源抓取订单数据
2. 模拟 Flume 的 Source → Channel → Sink 架构
3. 数据验证与格式化
4. 输出用于后续 Hive 清洗的原始数据

注意：本脚本为模拟实现，实际生产环境需对接真实 Flume 组件
"""

import os
import json
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataSource(Enum):
    """数据源类型枚举"""
    ERP = "ERP"           # 线下门店ERP系统
    POS = "POS"           # 门店POS收银系统
    ONLINE_EC = "ONLINE_EC"  # 线上电商平台
    APP = "APP"           # 优衣库APP
    MINI_PROGRAM = "MINI_PROGRAM"  # 微信小程序


@dataclass
class OrderRecord:
    """
    订单数据记录结构
    对应 ORDER_SCHEMA 定义的所有字段
    """
    Order_ID: str
    Source: str
    Region: str
    Order_Date: str
    Season: str
    Category: str
    Product_Code: str
    Store_Code: Optional[str] = None
    Order_Time: Optional[str] = None
    Product_Name: Optional[str] = None
    Color: Optional[str] = None
    Size: Optional[str] = None
    Quantity: int = 1
    Unit_Price: float = 0.0
    Discount: float = 0.0
    Total_Amount: float = 0.0
    Payment_Method: Optional[str] = None
    Customer_ID: Optional[str] = None
    Age_Group: Optional[str] = None
    Gender: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {k: v for k, v in asdict(self).items() if v is not None}

    def validate(self) -> tuple[bool, str]:
        """
        验证数据完整性
        Returns: (is_valid, error_message)
        """
        required_fields = ['Order_ID', 'Source', 'Region', 'Order_Date', 
                          'Season', 'Category', 'Product_Code']
        
        for field in required_fields:
            if not getattr(self, field, None):
                return False, f"Missing required field: {field}"
        
        if self.Quantity < 1:
            return False, "Quantity must be at least 1"
        
        if self.Unit_Price < 0:
            return False, "Unit_Price cannot be negative"
        
        return True, ""


class FlumeSimulator:
    """
    Flume 数据抓取模拟器
    
    模拟 Flume 的核心组件：
    - Source: 数据源接入
    - Channel: 数据缓冲
    - Sink: 数据输出(HDFS)
    """
    
    def __init__(self, output_dir: str = "./data/raw"):
        """
        初始化 Flume 模拟器
        
        Args:
            output_dir: 输出目录 (模拟 HDFS 存储路径)
        """
        self.output_dir = output_dir
        self.channel_buffer: List[OrderRecord] = []
        self.stats = {
            'total_processed': 0,
            'valid_records': 0,
            'invalid_records': 0,
            'by_source': {}
        }
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"FlumeSimulator initialized with output_dir: {output_dir}")
    
    def ingest_from_source(self, source: DataSource, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        从指定数据源抓取数据
        
        Args:
            source: 数据源类型
            data: 原始订单数据列表
        
        Returns:
            处理统计信息
        """
        logger.info(f"Starting ingestion from source: {source.value}")
        
        valid_count = 0
        invalid_count = 0
        
        for record_data in data:
            try:
                # 创建订单记录
                record = OrderRecord(
                    Order_ID=record_data.get('Order_ID'),
                    Source=source.value,
                    Region=record_data.get('Region', ''),
                    Store_Code=record_data.get('Store_Code'),
                    Order_Date=record_data.get('Order_Date', ''),
                    Order_Time=record_data.get('Order_Time'),
                    Season=record_data.get('Season', ''),
                    Category=record_data.get('Category', ''),
                    Product_Code=record_data.get('Product_Code', ''),
                    Product_Name=record_data.get('Product_Name'),
                    Color=record_data.get('Color'),
                    Size=record_data.get('Size'),
                    Quantity=record_data.get('Quantity', 1),
                    Unit_Price=record_data.get('Unit_Price', 0.0),
                    Discount=record_data.get('Discount', 0.0),
                    Total_Amount=record_data.get('Total_Amount', 0.0),
                    Payment_Method=record_data.get('Payment_Method'),
                    Customer_ID=record_data.get('Customer_ID'),
                    Age_Group=record_data.get('Age_Group'),
                    Gender=record_data.get('Gender')
                )
                
                # 验证数据
                is_valid, error_msg = record.validate()
                if is_valid:
                    self.channel_buffer.append(record)
                    valid_count += 1
                else:
                    logger.warning(f"Invalid record {record.Order_ID}: {error_msg}")
                    invalid_count += 1
                    
            except Exception as e:
                logger.error(f"Error processing record: {e}")
                invalid_count += 1
        
        # 更新统计
        self.stats['total_processed'] += len(data)
        self.stats['valid_records'] += valid_count
        self.stats['invalid_records'] += invalid_count
        
        if source.value not in self.stats['by_source']:
            self.stats['by_source'][source.value] = {'valid': 0, 'invalid': 0}
        self.stats['by_source'][source.value]['valid'] += valid_count
        self.stats['by_source'][source.value]['invalid'] += invalid_count
        
        logger.info(f"Ingestion complete: {valid_count} valid, {invalid_count} invalid")
        
        return {
            'source': source.value,
            'total': len(data),
            'valid': valid_count,
            'invalid': invalid_count
        }
    
    def flush_to_hdfs(self, filename: str = None) -> str:
        """
        将缓冲数据写入 HDFS (模拟)
        
        Args:
            filename: 输出文件名
        
        Returns:
            输出文件路径
        """
        if not self.channel_buffer:
            logger.warning("No data in channel to flush")
            return ""
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"orders_raw_{timestamp}.json"
        
        output_path = os.path.join(self.output_dir, filename)
        
        # 写入 JSON 格式数据
        with open(output_path, 'w', encoding='utf-8') as f:
            records = [record.to_dict() for record in self.channel_buffer]
            json.dump(records, f, ensure_ascii=False, indent=2)
        
        record_count = len(self.channel_buffer)
        
        # 清空缓冲区
        self.channel_buffer.clear()
        
        logger.info(f"Flushed {record_count} records to: {output_path}")
        
        return output_path
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取采集统计信息"""
        return self.stats


def load_data_from_file(filepath: str) -> List[Dict[str, Any]]:
    """
    从文件加载订单数据
    
    Args:
        filepath: 数据文件路径
    
    Returns:
        订单数据列表
    """
    logger.info(f"Loading data from: {filepath}")
    
    if not os.path.exists(filepath):
        logger.warning(f"File not found: {filepath}")
        return []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        if filepath.endswith('.json'):
            data = json.load(f)
        elif filepath.endswith('.csv'):
            # 简单 CSV 解析
            import csv
            reader = csv.DictReader(f)
            data = list(reader)
        else:
            logger.error(f"Unsupported file format: {filepath}")
            return []
    
    logger.info(f"Loaded {len(data)} records")
    return data


def run_ingestion_pipeline(data_sources: List[tuple], output_dir: str = "./data/raw") -> Dict[str, Any]:
    """
    运行完整的数据采集管道
    
    Args:
        data_sources: 数据源列表 [(source_type, filepath), ...]
        output_dir: 输出目录
    
    Returns:
        采集统计信息
    """
    flume = FlumeSimulator(output_dir)
    
    results = []
    
    for source_type, filepath in data_sources:
        # 解析数据源类型
        try:
            source = DataSource(source_type)
        except ValueError:
            logger.error(f"Unknown source type: {source_type}")
            continue
        
        # 加载数据
        data = load_data_from_file(filepath)
        if data:
            result = flume.ingest_from_source(source, data)
            results.append(result)
    
    # 写入 HDFS
    output_file = flume.flush_to_hdfs()
    
    return {
        'sources_processed': results,
        'overall_stats': flume.get_statistics(),
        'output_file': output_file
    }


# ============================================================================
# 入口函数 - 供外部调用
# ============================================================================

def main():
    """主函数 - 测试数据采集流程"""
    
    # 示例数据 (实际使用时替换为真实数据文件路径)
    sample_data = [
        {
            "Order_ID": "ORD-20240315-000001",
            "Region": "Shanghai",
            "Store_Code": "SH-001",
            "Order_Date": "2024-03-15",
            "Order_Time": "14:30:25",
            "Season": "Spring",
            "Category": "T-Shirt",
            "Product_Code": "SKU-123456",
            "Product_Name": "UT Crew Neck T-Shirt",
            "Color": "White",
            "Size": "M",
            "Quantity": 2,
            "Unit_Price": 99.00,
            "Discount": 10.00,
            "Total_Amount": 188.00,
            "Payment_Method": "Alipay",
            "Customer_ID": "CUST-2024001",
            "Age_Group": "26-35",
            "Gender": "Male"
        }
    ]
    
    # 初始化采集器
    flume = FlumeSimulator(output_dir="./data/raw")
    
    # 模拟从多个数据源采集
    sources = [
        (DataSource.POS, sample_data),
        (DataSource.ONLINE_EC, sample_data),
    ]
    
    for source, data in sources:
        flume.ingest_from_source(source, data)
    
    # 写入 HDFS
    output_path = flume.flush_to_hdfs()
    
    print(f"\n{'='*60}")
    print("Data Ingestion Complete")
    print(f"{'='*60}")
    print(f"Output file: {output_path}")
    print(f"Statistics: {json.dumps(flume.get_statistics(), indent=2)}")


if __name__ == "__main__":
    main()
