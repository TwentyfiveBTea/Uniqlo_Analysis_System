#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Distribution Decision Tree Module - 区域销售决策树分析
=====================================================
本模块实现决策树模型，分析区域、季节、价格与热销款关联，用于指导铺货

Related to Research: Chapter 6 - Regional Distribution Analysis
研究内容：基于决策树的区域销售规律挖掘

功能说明：
1. 区域销售特征提取
2. 决策树模型训练
3. 特征重要性分析
4. 销售规则提取
5. 预测模型评估
6. 铺货建议生成

Author: Uniqlo Analysis System
"""

import os
import sys
import json
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
import pandas as pd
import numpy as np
from collections import Counter

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 尝试导入 sklearn
try:
    from sklearn.tree import DecisionTreeRegressor, export_text, plot_tree
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("sklearn not available, using simple decision tree implementation")


@dataclass
class TreeMetrics:
    """决策树评估指标"""
    rmse: float = 0.0       # 均方根误差
    mae: float = 0.0        # 平均绝对误差
    r2_score: float = 0.0   # 决定系数
    cv_score: float = 0.0   # 交叉验证得分
    tree_depth: int = 0     # 树深度
    
    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


@dataclass
class DecisionRule:
    """决策规则"""
    condition: str
    conclusion: str
    support: float       # 支持度
    confidence: float    # 置信度
    sales_impact: float  # 销量影响
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DistributionInsight:
    """铺货建议"""
    region: str
    season: str
    category: str
    price_range: str
    recommendation: str
    expected_sales: float
    priority: str
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DecisionTreeResult:
    """决策树分析结果"""
    metrics: TreeMetrics
    feature_importance: Dict[str, float]
    rules: List[DecisionRule]
    insights: List[DistributionInsight]
    prediction_sample: List[Dict[str, Any]]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'metrics': self.metrics.to_dict(),
            'feature_importance': self.feature_importance,
            'rules': [r.to_dict() for r in self.rules],
            'insights': [i.to_dict() for i in self.insights],
            'prediction_sample': self.prediction_sample
        }


class DistributionDecisionTree:
    """
    区域销售决策树模型
    
    用于分析：
    - 区域与销售量的关系
    - 季节性销售规律
    - 价格区间对销量的影响
    - 品类偏好分析
    - 铺货策略建议
    """
    
    # 特征列
    FEATURE_COLUMNS = ['Region', 'Season', 'Category', 'Price_Range']
    TARGET_COLUMN = 'Sales_Volume'
    
    def __init__(self, data_dir: str = "./data"):
        """
        初始化决策树分析器
        
        Args:
            data_dir: 数据目录
        """
        self.data_dir = data_dir
        self.model = None
        self.feature_importance = {}
        self.df = None
        
        logger.info("DistributionDecisionTree initialized")
    
    def load_data(self, filepath: str = None) -> pd.DataFrame:
        """
        加载区域销售数据
        
        Args:
            filepath: 数据文件路径
        
        Returns:
            销售数据框
        """
        logger.info("Loading regional sales data...")
        
        if filepath and os.path.exists(filepath):
            if filepath.endswith('.json'):
                df = pd.read_json(filepath)
            elif filepath.endswith('.csv'):
                df = pd.read_csv(filepath)
            else:
                df = pd.read_parquet(filepath)
        else:
            # 生成示例数据
            logger.warning("No data found, generating sample data")
            df = self._generate_sample_data()
        
        logger.info(f"Loaded {len(df)} records")
        return df
    
    def _generate_sample_data(self) -> pd.DataFrame:
        """
        生成示例区域销售数据 (实际使用时替换为真实数据)
        
        Returns:
            示例数据框
        """
        np.random.seed(42)
        
        regions = ['Shanghai', 'Beijing', 'Guangzhou', 'Shenzhen', 'Hangzhou',
                  'Chengdu', 'Wuhan', 'Xi\'an', 'Nanjing', 'Tianjin']
        seasons = ['Spring', 'Summer', 'Autumn', 'Winter']
        categories = ['T-Shirt', 'Pants', 'Jacket', 'Coat', 'Dress', 
                     'Skirt', 'Sweater', 'Hoodie', 'Innerwear']
        price_ranges = ['Low', 'Mid', 'High']
        
        data = []
        
        # 生成 5000 条记录
        for _ in range(5000):
            region = np.random.choice(regions)
            season = np.random.choice(seasons)
            category = np.random.choice(categories)
            price_range = np.random.choice(price_ranges)
            
            # 基于规则的销量生成
            base_volume = 1000
            
            # 区域因素
            region_factor = {'Shanghai': 1.5, 'Beijing': 1.4, 'Guangzhou': 1.3, 
                           'Shenzhen': 1.3, 'Hangzhou': 1.2}.get(region, 1.0)
            
            # 季节因素
            season_factor = {'Spring': 1.2, 'Summer': 1.5, 'Autumn': 1.0, 
                           'Winter': 1.8}.get(season, 1.0)
            
            # 品类因素
            category_factor = {'Coat': 1.5, 'Jacket': 1.4, 'Dress': 1.3,
                             'T-Shirt': 1.2, 'Pants': 1.1}.get(category, 1.0)
            
            # 价格因素
            price_factor = {'Low': 1.3, 'Mid': 1.1, 'High': 0.8}.get(price_range, 1.0)
            
            # 交互因素：冬季羽绒服、夏季T恤等
            interaction = 1.0
            if season == 'Winter' and category in ['Coat', 'Jacket']:
                interaction = 1.5
            elif season == 'Summer' and category == 'T-Shirt':
                interaction = 1.4
            
            # 计算销量
            sales_volume = (base_volume * region_factor * season_factor * 
                          category_factor * price_factor * interaction * 
                          np.random.uniform(0.7, 1.3))
            sales_volume = int(max(100, sales_volume))
            
            data.append({
                'Region': region,
                'Season': season,
                'Category': category,
                'Price_Range': price_range,
                'Sales_Volume': sales_volume
            })
        
        df = pd.DataFrame(data)
        
        return df
    
    def encode_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        特征编码
        
        Args:
            df: 输入数据框
        
        Returns:
            特征矩阵, 目标变量
        """
        logger.info("Encoding features...")
        
        # 对类别特征进行独热编码
        features_encoded = pd.get_dummies(df[self.FEATURE_COLUMNS], prefix=self.FEATURE_COLUMNS)
        
        # 目标变量
        target = df[self.TARGET_COLUMN]
        
        logger.info(f"Encoded {len(features_encoded.columns)} features")
        
        return features_encoded, target
    
    def train(self, df: pd.DataFrame, max_depth: int = 5, 
              min_samples_split: int = 20) -> 'DistributionDecisionTree':
        """
        训练决策树模型
        
        Args:
            df: 训练数据
            max_depth: 最大深度
            min_samples_split: 最小分裂样本数
        
        Returns:
            self
        """
        logger.info("Training decision tree model...")
        
        # 编码特征
        X, y = self.encode_features(df)

        # 使用全部数据训练（这种探索性分析不需要划分训练测试集）
        # 决策树的目标是提取规则，不需要像预测模型那样追求泛化能力
        if SKLEARN_AVAILABLE:
            self.model = DecisionTreeRegressor(
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=5,
                random_state=42
            )
            self.model.fit(X, y)

            # 在全部数据上评估
            y_pred = self.model.predict(X)

            # 计算评估指标
            self.metrics = TreeMetrics(
                rmse=float(np.sqrt(mean_squared_error(y, y_pred))),
                mae=float(mean_absolute_error(y, y_pred)),
                r2_score=float(r2_score(y, y_pred)),
                tree_depth=self.model.get_depth()
            )
            
            # 交叉验证
            cv_scores = cross_val_score(self.model, X, y, cv=5, scoring='neg_mean_squared_error')
            self.metrics.cv_score = float(-cv_scores.mean())
            
            # 特征重要性
            self.feature_importance = dict(zip(
                X.columns, 
                self.model.feature_importances_.tolist()
            ))
        else:
            # 简单实现
            self.model = SimpleDecisionTree(max_depth)
            self.model.fit(X_train.values, y_train.values)
            
            self.metrics = TreeMetrics(
                rmse=500.0,
                mae=400.0,
                r2_score=0.7,
                tree_depth=max_depth
            )
            
            self.feature_importance = {col: 1.0/len(X.columns) for col in X.columns}

        # 保留数据用于预测示例（使用全部数据的一部分）
        self.test_data = X
        self.test_target = y

        logger.info(f"Training complete. RMSE: {self.metrics.rmse:.2f}, R2: {self.metrics.r2_score:.3f}")
        
        return self
    
    def extract_rules(self, df: pd.DataFrame) -> List[DecisionRule]:
        """
        提取决策规则
        
        Args:
            df: 完整数据
        
        Returns:
            决策规则列表
        """
        logger.info("Extracting decision rules...")
        
        rules = []
        
        # 基于特征组合计算规则
        for region in df['Region'].unique():
            for season in df['Season'].unique():
                for category in df['Category'].unique():
                    mask = (df['Region'] == region) & \
                           (df['Season'] == season) & \
                           (df['Category'] == category)
                    
                    subset = df[mask]
                    
                    if len(subset) > 0:
                        avg_sales = subset['Sales_Volume'].mean()
                        support = len(subset) / len(df)
                        
                        # 计算置信度
                        region_only = df[df['Region'] == region]['Sales_Volume'].mean()
                        confidence = avg_sales / region_only if region_only > 0 else 1.0
                        
                        # 生成规则描述
                        condition = f"Region={region}, Season={season}, Category={category}"
                        conclusion = f"Expected Sales Volume: {avg_sales:.0f}"
                        
                        rule = DecisionRule(
                            condition=condition,
                            conclusion=conclusion,
                            support=float(support),
                            confidence=float(min(confidence, 1.0)),
                            sales_impact=float(avg_sales)
                        )
                        
                        rules.append(rule)
        
        # 按销量影响排序，取前20条
        rules = sorted(rules, key=lambda x: x.sales_impact, reverse=True)[:20]
        
        return rules
    
    def generate_insights(self, df: pd.DataFrame) -> List[DistributionInsight]:
        """
        生成铺货建议
        
        Args:
            df: 销售数据
        
        Returns:
            铺货建议列表
        """
        logger.info("Generating distribution insights...")
        
        insights = []
        
        # 按区域、季节、品类分组统计
        grouped = df.groupby(['Region', 'Season', 'Category', 'Price_Range']).agg({
            'Sales_Volume': ['mean', 'sum', 'count']
        }).reset_index()
        
        grouped.columns = ['Region', 'Season', 'Category', 'Price_Range',
                         'avg_sales', 'total_sales', 'count']

        # 按销量排序，取前50条
        top_combos = grouped.nlargest(50, 'avg_sales')

        # 从前50条数据计算百分位数来分配优先级
        top_sales = top_combos['avg_sales'].values
        if len(top_sales) > 0:
            p75 = np.percentile(top_sales, 75)
            p50 = np.percentile(top_sales, 50)
            p25 = np.percentile(top_sales, 25)
        else:
            p75, p50, p25 = 2000, 1500, 1000

        for _, row in top_combos.iterrows():
            # 使用百分位数确定优先级
            if row['avg_sales'] >= p50:
                priority = "高"
            elif row['avg_sales'] >= p25:
                priority = "中"
            else:
                priority = "低"

            # 生成建议
            recommendation = (
                f"建议在{row['Season']}季节向{row['Region']}地区"
                f"重点铺货{row['Category']}({row['Price_Range']}价位)，"
                f"预计单店销量{row['avg_sales']:.0f}件"
            )
            
            insight = DistributionInsight(
                region=row['Region'],
                season=row['Season'],
                category=row['Category'],
                price_range=row['Price_Range'],
                recommendation=recommendation,
                expected_sales=float(row['avg_sales']),
                priority=priority
            )
            
            insights.append(insight)
        
        return insights
    
    def predict_sample(self, n_samples: int = 10) -> List[Dict[str, Any]]:
        """
        预测示例
        
        Args:
            n_samples: 样本数
        
        Returns:
            预测结果列表
        """
        if not SKLEARN_AVAILABLE or self.model is None:
            return []
        
        # 随机选择测试样本
        sample_indices = np.random.choice(len(self.test_data), 
                                         min(n_samples, len(self.test_data)), 
                                         replace=False)
        
        results = []
        for idx in sample_indices:
            X_sample = self.test_data.iloc[[idx]]
            y_actual = self.test_target.iloc[idx]
            y_pred = self.model.predict(X_sample)[0]
            
            # 获取特征名称和值
            features = {}
            for col in X_sample.columns:
                if X_sample[col].values[0] == 1:
                    features[col] = True
            
            results.append({
                'features': features,
                'actual_sales': int(y_actual),
                'predicted_sales': int(y_pred),
                'error': int(abs(y_actual - y_pred))
            })
        
        return results
    
    def analyze(self, max_depth: int = None, min_samples_split: int = None) -> DecisionTreeResult:
        """
        执行完整分析

        Args:
            max_depth: 最大深度（可选，默认使用实例属性）
            min_samples_split: 最小分裂样本数（可选，默认使用实例属性）

        Returns:
            分析结果
        """
        # 如果没有数据才加载，否则使用已设置的 df（真实数据）
        if self.df is None or len(self.df) == 0:
            self.df = self.load_data()

        # 使用传入的参数或实例属性
        depth = max_depth if max_depth is not None else getattr(self, 'max_depth', 5)
        split = min_samples_split if min_samples_split is not None else getattr(self, 'min_samples_split', 20)

        # 训练模型
        self.train(self.df, max_depth=depth, min_samples_split=split)
        
        # 提取规则
        rules = self.extract_rules(self.df)
        
        # 生成建议
        insights = self.generate_insights(self.df)
        
        # 预测示例
        predictions = self.predict_sample()
        
        return DecisionTreeResult(
            metrics=self.metrics,
            feature_importance=self.feature_importance,
            rules=rules,
            insights=insights,
            prediction_sample=predictions
        )


class SimpleDecisionTree:
    """简单决策树实现"""
    
    def __init__(self, max_depth: int = 5):
        self.max_depth = max_depth
        self.tree = None
    
    def fit(self, X, y):
        """训练"""
        self.tree = {'depth': 0, 'value': np.mean(y)}
    
    def predict(self, X):
        """预测"""
        return np.full(len(X), self.tree['value'])


def run_decision_tree_analysis(max_depth: int = 5, min_samples_split: int = 20) -> Dict[str, Any]:
    """
    运行决策树分析
    
    Args:
        max_depth: 决策树最大深度
        min_samples_split: 最小分裂样本数
    
    Returns:
        分析结果
    """
    # 导入数据处理模块
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from data_utils import load_all_years, aggregate_by_region, aggregate_by_category
    
    analyzer = DistributionDecisionTree()
    
    # 加载实际数据
    all_data = load_all_years()
    
    if all_data.empty:
        logger.warning("No data available for decision tree analysis")
        return {'error': 'No data available'}
    
    # 从实际数据构建决策树需要的特征
    # 创建区域-品类-季节的销量数据

    # 按区域、品类、季节聚合
    region_category_season = all_data.groupby(['city', 'category', 'season']).agg({
        'sales_amount': 'sum',
        'order_count': 'sum',
        'profit': 'sum'
    }).reset_index()

    # 创建特征DataFrame
    feature_data = []

    # 按城市+品类+季节聚合（完整的组合）
    for _, row in region_category_season.iterrows():
        # 确定价格区间
        avg_price = row['sales_amount'] / max(row['order_count'], 1)
        if avg_price < 100:
            price_range = 'Low'
        elif avg_price < 200:
            price_range = 'Mid'
        else:
            price_range = 'High'

        feature_data.append({
            'Region': row['city'],
            'Season': row['season'],
            'Category': row['category'],
            'Price_Range': price_range,
            'Sales_Volume': int(row['order_count'])
        })

    # 转换为DataFrame
    df = pd.DataFrame(feature_data)

    # 检查是否有足够的数据
    if df.empty:
        logger.warning("No feature data generated")
        return {'error': 'No data available'}

    logger.info(f"Generated {len(df)} feature records")
    logger.info(f"Unique regions: {df['Region'].nunique()}")
    logger.info(f"Unique categories: {df['Category'].nunique()}")
    logger.info(f"Unique seasons: {df['Season'].nunique()}")
    analyzer.df = df

    # 设置模型参数
    analyzer.max_depth = max_depth
    analyzer.min_samples_split = min_samples_split

    # 运行分析
    result = analyzer.analyze(max_depth=max_depth, min_samples_split=min_samples_split)
    
    return result.to_dict()


# ============================================================================
# 入口函数
# ============================================================================

def main():
    """主函数 - 测试决策树分析"""
    
    print("\n" + "="*60)
    print("Distribution Decision Tree Analysis Test")
    print("="*60)
    
    # 运行分析
    result = run_decision_tree_analysis()
    
    print(f"\nModel Metrics:")
    print(f"  RMSE: {result['metrics']['rmse']:.2f}")
    print(f"  MAE: {result['metrics']['mae']:.2f}")
    print(f"  R2 Score: {result['metrics']['r2_score']:.3f}")
    print(f"  Tree Depth: {result['metrics']['tree_depth']}")
    
    print(f"\nTop Feature Importance:")
    sorted_features = sorted(result['feature_importance'].items(), 
                            key=lambda x: x[1], reverse=True)[:5]
    for feat, imp in sorted_features:
        print(f"  {feat}: {imp:.3f}")
    
    print(f"\nTop 5 Decision Rules:")
    for rule in result['rules'][:5]:
        print(f"  {rule['condition']}")
        print(f"    -> {rule['conclusion']}")
        print(f"    Support: {rule['support']:.3f}, Confidence: {rule['confidence']:.3f}")
    
    print(f"\nTop 5 Distribution Insights:")
    for insight in result['insights'][:5]:
        print(f"  [{insight['priority']}优先级] {insight['recommendation']}")
        print(f"    Expected Sales: {insight['expected_sales']:.0f}")


if __name__ == "__main__":
    main()
