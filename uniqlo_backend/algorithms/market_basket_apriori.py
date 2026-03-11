#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Market Basket Apriori Module - 商品关联规则挖掘
===============================================
本模块实现 Apriori 算法，挖掘商品组合关联规则

Related to Research: Chapter 7 - Product Association Analysis
研究内容：基于 Apriori 的商品关联规则挖掘

功能说明：
1. 交易数据预处理
2. 频繁项集生成
3. 关联规则挖掘
4. 支持度、置信度、提升度计算
5. 规则排序与筛选
6. 可视化结果输出

Author: Uniqlo Analysis System
"""

import os
import sys
import json
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional, Set, Tuple
from dataclasses import dataclass, asdict
import pandas as pd
import numpy as np
from collections import Counter, defaultdict
from itertools import combinations

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class AssociationRule:
    """关联规则"""
    antecedent: List[str]    # 前项
    consequent: List[str]    # 后项
    support: float           # 支持度
    confidence: float       # 置信度
    lift: float             # 提升度
    leverage: float         # 杠杆率
    conviction: float        # 确信度
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class FrequentItemset:
    """频繁项集"""
    items: Set[str]
    support: float
    item_count: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'items': list(self.items),
            'support': self.support,
            'item_count': self.item_count
        }


@dataclass
class AprioriResult:
    """Apriori 分析结果"""
    frequent_itemsets: List[FrequentItemset]
    association_rules: List[AssociationRule]
    min_support: float
    min_confidence: float
    total_transactions: int
    execution_time: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'frequent_itemsets': [f.to_dict() for f in self.frequent_itemsets],
            'association_rules': [r.to_dict() for r in self.association_rules],
            'min_support': self.min_support,
            'min_confidence': self.min_confidence,
            'total_transactions': self.total_transactions,
            'execution_time': self.execution_time
        }


class AprioriAnalyzer:
    """
    Apriori 关联规则分析模型
    
    用于挖掘：
    - 商品购买组合规律
    - 交叉销售机会
    - 促销策略优化
    - 商品陈列优化
    """
    
    def __init__(self, data_dir: str = "./data"):
        """
        初始化 Apriori 分析器
        
        Args:
            data_dir: 数据目录
        """
        self.data_dir = data_dir
        self.transactions = []
        self.item_counts = Counter()
        
        logger.info("AprioriAnalyzer initialized")
    
    def load_transaction_data(self, filepath: str = None) -> List[List[str]]:
        """
        加载交易数据
        
        Args:
            filepath: 数据文件路径
        
        Returns:
            交易列表 (每条交易是商品列表)
        """
        logger.info("Loading transaction data...")
        
        if filepath and os.path.exists(filepath):
            if filepath.endswith('.json'):
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    transactions = [item.get('Items', []) for item in data]
            elif filepath.endswith('.csv'):
                df = pd.read_csv(filepath)
                transactions = df['Items'].apply(lambda x: x.split(',')).tolist()
            else:
                transactions = self._generate_sample_transactions()
        else:
            # 生成示例交易数据
            logger.warning("No transaction data found, generating sample data")
            transactions = self._generate_sample_transactions()
        
        # 过滤空交易
        self.transactions = [t for t in transactions if t]
        
        logger.info(f"Loaded {len(self.transactions)} transactions")
        
        # 统计商品频次
        for trans in self.transactions:
            self.item_counts.update(trans)
        
        return self.transactions
    
    def _generate_sample_transactions(self) -> List[List[str]]:
        """
        生成示例交易数据 (实际使用时替换为真实数据)
        
        Returns:
            示例交易列表
        """
        np.random.seed(42)
        
        # 定义商品池
        categories = {
            'T-Shirt': ['UT Crew Neck', 'UT Graphic', 'AIRism棉混纺'],
            'Pants': ['牛仔裤', '休闲裤', '卡其裤'],
            'Jacket': ['法兰绒衬衫', '连帽开衫', '棒球外套'],
            'Coat': ['羽绒服', '大衣', '棉服'],
            'Dress': ['连衣裙', '半身裙'],
            'Sweater': ['针织衫', '羊绒衫'],
            'Accessories': ['帽子', '围巾', '袜子']
        }
        
        # 生成10000条交易
        transactions = []
        
        for _ in range(10000):
            # 随机选择交易中的商品数量
            n_items = np.random.choice([1, 2, 3, 4, 5], p=[0.15, 0.35, 0.30, 0.15, 0.05])
            
            # 随机选择品类
            selected_categories = np.random.choice(list(categories.keys()), 
                                                  size=min(n_items, len(categories)),
                                                  replace=False)
            
            # 从每个品类中随机选择商品
            items = []
            for cat in selected_categories:
                item = np.random.choice(categories[cat])
                items.append(item)
            
            transactions.append(items)
        
        return transactions
    
    def _get_support(self, itemset: Set[str]) -> float:
        """
        计算项集支持度
        
        Args:
            itemset: 项集
        
        Returns:
            支持度
        """
        # 支持 DataFrame 和 list 两种格式
        if isinstance(self.transactions, pd.DataFrame):
            # DataFrame 格式：检查每行是否包含所有项
            items_list = list(itemset)
            if all(col in self.transactions.columns for col in items_list):
                # 统计包含所有项的交易数
                mask = self.transactions[items_list].all(axis=1)
                count = int(mask.sum())
            else:
                count = 0
            return count / len(self.transactions) if len(self.transactions) > 0 else 0
        else:
            # list 格式
            count = sum(1 for trans in self.transactions if itemset.issubset(set(trans)))
            return count / len(self.transactions) if self.transactions else 0
    
    def _generate_candidates(self, prev_frequent: List[Set[str]], k: int) -> List[Set[str]]:
        """
        生成候选项集
        
        Args:
            prev_frequent: 前一轮频繁项集
            k: 项集大小
        
        Returns:
            候选项集列表
        """
        candidates = []
        n = len(prev_frequent)
        
        for i in range(n):
            for j in range(i + 1, n):
                # 合并两个项集
                union = prev_frequent[i] | prev_frequent[j]
                
                if len(union) == k:
                    # 检查所有 k-1 子集是否频繁
                    is_valid = True
                    for item in union:
                        subset = union - {item}
                        if subset not in prev_frequent:
                            is_valid = False
                            break
                    
                    if is_valid and union not in candidates:
                        candidates.append(union)
        
        return candidates
    
    def find_frequent_itemsets(self, min_support: float = 0.01, 
                              max_length: int = 5) -> List[FrequentItemset]:
        """
        挖掘频繁项集
        
        Args:
            min_support: 最小支持度
            max_length: 最大项集长度
        
        Returns:
            频繁项集列表
        """
        logger.info(f"Finding frequent itemsets with min_support={min_support}")
        
        all_frequent = []
        
        # 1-项集
        current_frequent = []
        for item, count in self.item_counts.items():
            support = count / len(self.transactions)
            if support >= min_support:
                itemset = {item}
                current_frequent.append(itemset)
                all_frequent.append(FrequentItemset(
                    items=itemset,
                    support=support,
                    item_count=count
                ))
        
        # 迭代生成 k-项集
        k = 2
        while current_frequent and k <= max_length:
            logger.info(f"Generating {k}-itemsets...")
            
            # 生成候选项集
            candidates = self._generate_candidates(current_frequent, k)
            
            # 计算支持度并筛选
            current_frequent = []
            for candidate in candidates:
                support = self._get_support(candidate)
                if support >= min_support:
                    current_frequent.append(candidate)
                    count = sum(1 for trans in self.transactions 
                               if candidate.issubset(set(trans)))
                    all_frequent.append(FrequentItemset(
                        items=candidate,
                        support=support,
                        item_count=count
                    ))
            
            k += 1
        
        # 按支持度排序
        all_frequent.sort(key=lambda x: x.support, reverse=True)
        
        logger.info(f"Found {len(all_frequent)} frequent itemsets")
        
        return all_frequent
    
    def _calculate_rule_metrics(self, antecedent: Set[str], 
                                consequent: Set[str]) -> Tuple[float, float, float, float, float]:
        """
        计算关联规则指标
        
        Args:
            antecedent: 前项
            consequent: 后项
        
        Returns:
            (support, confidence, lift, leverage, conviction)
        """
        # 前项支持度
        ant_support = self._get_support(antecedent)
        
        # 前后项并集支持度
        union = antecedent | consequent
        union_support = self._get_support(union)
        
        # 置信度
        confidence = union_support / ant_support if ant_support > 0 else 0
        
        # 后项支持度
        cons_support = self._get_support(consequent)
        
        # 提升度
        lift = confidence / cons_support if cons_support > 0 else 0
        
        # 杠杆率
        leverage = union_support - (ant_support * cons_support)
        
        # 确信度
        if confidence < 1:
            conviction = (1 - cons_support) / (1 - confidence) if (1 - confidence) > 0 else 0
        else:
            conviction = float('inf')
        
        return union_support, confidence, lift, leverage, conviction
    
    def generate_rules(self, frequent_itemsets: List[FrequentItemset],
                      min_confidence: float = 0.5) -> List[AssociationRule]:
        """
        生成关联规则
        
        Args:
            frequent_itemsets: 频繁项集
            min_confidence: 最小置信度
        
        Returns:
            关联规则列表
        """
        logger.info(f"Generating rules with min_confidence={min_confidence}")
        
        rules = []
        
        # 从频繁项集生成规则
        for itemset in frequent_itemsets:
            items = list(itemset.items)
            
            if len(items) < 2:
                continue
            
            # 生成所有可能的规则组合
            for i in range(1, len(items)):
                for ant in combinations(items, i):
                    ant = set(ant)
                    cons = itemset.items - ant
                    
                    if not cons:
                        continue
                    
                    # 计算规则指标
                    support, confidence, lift, leverage, conviction = \
                        self._calculate_rule_metrics(ant, cons)
                    
                    # 筛选满足最小置信度的规则
                    if confidence >= min_confidence:
                        rule = AssociationRule(
                            antecedent=list(ant),
                            consequent=list(cons),
                            support=support,
                            confidence=confidence,
                            lift=lift,
                            leverage=leverage,
                            conviction=conviction
                        )
                        rules.append(rule)
        
        # 按提升度排序
        rules.sort(key=lambda x: x.lift, reverse=True)
        
        logger.info(f"Generated {len(rules)} association rules")
        
        return rules
    
    def analyze(self, min_support: float = 0.01, 
               min_confidence: float = 0.5,
               max_item_length: int = 5) -> AprioriResult:
        """
        执行完整分析
        
        Args:
            min_support: 最小支持度
            min_confidence: 最小置信度
            max_item_length: 最大项集长度
        
        Returns:
            分析结果
        """
        import time
        start_time = time.time()
        
        # 加载数据
        self.load_transaction_data()
        
        # 挖掘频繁项集
        frequent_itemsets = self.find_frequent_itemsets(min_support, max_item_length)
        
        # 生成关联规则
        rules = self.generate_rules(frequent_itemsets, min_confidence)
        
        execution_time = time.time() - start_time
        
        logger.info(f"Apriori analysis complete in {execution_time:.2f}s")
        
        return AprioriResult(
            frequent_itemsets=frequent_itemsets,
            association_rules=rules,
            min_support=min_support,
            min_confidence=min_confidence,
            total_transactions=len(self.transactions),
            execution_time=execution_time
        )


def run_apriori_analysis(min_support: float = 0.01, 
                        min_confidence: float = 0.5) -> Dict[str, Any]:
    """
    运行 Apriori 分析
    
    Args:
        min_support: 最小支持度
        min_confidence: 最小置信度
    
    Returns:
        分析结果
    """
    # 导入数据处理模块
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from data_utils import load_all_years
    
    analyzer = AprioriAnalyzer()
    
    # 从实际数据构建交易数据
    all_data = load_all_years()
    
    if all_data.empty:
        logger.warning("No data available for Apriori analysis")
        return {'error': 'No data available'}

    # 按城市+日期构建交易basket（减少交易数量以提高性能）
    transactions = []

    for (order_date, city), group in all_data.groupby(['order_date', 'city']):
        categories = group['category'].dropna().unique().tolist()
        if categories and len(categories) > 0:
            transactions.append(categories)

    # 限制交易数量以提高性能（随机采样或取最近的）
    max_transactions = 50000
    if len(transactions) > max_transactions:
        transactions = transactions[:max_transactions]
        logger.info(f"Limited transactions to {max_transactions} for performance")

    logger.info(f"Using {len(transactions)} transactions for Apriori analysis")
    
    # 转换格式为DataFrame（每行一个交易，每列一个商品）
    # 首先对transactions进行编码
    if transactions:
        # 获取所有唯一品类
        all_categories = set()
        for t in transactions:
            all_categories.update(t)
        
        # 创建交易编码
        encoded_transactions = []
        for t in transactions:
            transaction = {cat: 1 for cat in t}
            encoded_transactions.append(transaction)
        
        # 转换为DataFrame
        df = pd.DataFrame(encoded_transactions).fillna(0)
        
        # 设置到analyzer
        analyzer.transactions = df
        analyzer.all_items = list(all_categories)

        # 初始化 item_counts（从 DataFrame 计算）
        from collections import Counter
        analyzer.item_counts = Counter()
        for col in df.columns:
            analyzer.item_counts[col] = int(df[col].sum())

        # 直接运行分析
        import time
        start_time = time.time()

        # 挖掘频繁项集
        frequent_itemsets = analyzer.find_frequent_itemsets(min_support, max_length=3)

        # 只处理项集大小<=3的频繁项集（减少规则生成时间）
        filtered_itemsets = [item for item in frequent_itemsets if len(item.items) <= 3]

        # 生成关联规则
        rules = analyzer.generate_rules(filtered_itemsets, min_confidence)

        # 按提升度排序
        rules = sorted(rules, key=lambda x: x.lift, reverse=True)

        execution_time = time.time() - start_time

        logger.info(f"Apriori analysis complete in {execution_time:.2f}s")

        # 构建返回结果 - 返回更多规则和频繁项集
        result = {
            'frequent_itemsets': [
                {
                    'items': list(itemset.items),
                    'support': float(itemset.support)
                }
                for itemset in frequent_itemsets[:50]  # 增加返回数量
            ],
            'association_rules': [
                {
                    'antecedent': list(rule.antecedent),
                    'consequent': list(rule.consequent),
                    'support': float(rule.support),
                    'confidence': float(rule.confidence),
                    'lift': float(rule.lift)
                }
                for rule in rules[:50]  # 增加返回数量
            ],
            'min_support': min_support,
            'min_confidence': min_confidence,
            'total_transactions': len(transactions),
            'execution_time': execution_time
        }

        return result


# ============================================================================
# 入口函数
# ============================================================================

def main():
    """主函数 - 测试 Apriori 分析"""
    
    print("\n" + "="*60)
    print("Market Basket Apriori Analysis Test")
    print("="*60)
    
    # 运行分析
    result = run_apriori_analysis(min_support=0.01, min_confidence=0.5)
    
    print(f"\nParameters:")
    print(f"  Min Support: {result['min_support']}")
    print(f"  Min Confidence: {result['min_confidence']}")
    print(f"  Total Transactions: {result['total_transactions']}")
    print(f"  Execution Time: {result['execution_time']:.2f}s")
    
    print(f"\nFrequent Itemsets: {len(result['frequent_itemsets'])}")
    print("\nTop 10 Frequent Itemsets:")
    for itemset in result['frequent_itemsets'][:10]:
        print(f"  {itemset['items']}: support={itemset['support']:.4f}")
    
    print(f"\nAssociation Rules: {len(result['association_rules'])}")
    print("\nTop 10 Rules (by lift):")
    for rule in result['association_rules'][:10]:
        print(f"  {rule['antecedent']} -> {rule['consequent']}")
        print(f"    Support: {rule['support']:.4f}, Confidence: {rule['confidence']:.4f}")
        print(f"    Lift: {rule['lift']:.4f}")


if __name__ == "__main__":
    main()
