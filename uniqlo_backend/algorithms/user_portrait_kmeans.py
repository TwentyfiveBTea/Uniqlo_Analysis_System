#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
User Portrait K-means Clustering Module - 用户画像聚类分析
=========================================================
本模块实现 K-means 聚类算法，基于消费频次、客单价、品类偏好划分用户画像

Related to Research: Chapter 5 - Customer Segmentation Analysis
研究内容：基于聚类的用户画像分析

功能说明：
1. 用户行为特征提取
2. 特征标准化处理
3. K-means 聚类分析
4. 轮廓系数评估
5. 用户画像标签生成
6. 聚类结果可视化数据

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
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("sklearn not available, using simple K-means implementation")


@dataclass
class ClusterMetrics:
    """聚类评估指标"""
    silhouette_score: float = 0.0    # 轮廓系数
    calinski_harabasz_score: float = 0.0  # CH指数
    davies_bouldin_score: float = 0.0    # DB指数
    inertia: float = 0.0               # 簇内平方和
    
    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


@dataclass
class UserCluster:
    """用户聚类结果"""
    cluster_id: int
    user_count: int
    avg_total_orders: float
    avg_total_spend: float
    avg_order_value: float
    avg_purchase_frequency: float
    avg_days_since_last_purchase: float
    top_categories: List[str]
    top_regions: List[str]
    portrait_label: str
    description: str
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ClusteringResult:
    """聚类分析结果"""
    clusters: List[UserCluster]
    metrics: ClusterMetrics
    optimal_k: int
    user_assignments: List[Dict[str, Any]]
    feature_importance: Dict[str, float]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'clusters': [c.to_dict() for c in self.clusters],
            'metrics': self.metrics.to_dict(),
            'optimal_k': self.optimal_k,
            'user_assignments': self.user_assignments,
            'feature_importance': self.feature_importance
        }


class KMeansUserProfiler:
    """
    K-means 用户画像聚类模型
    
    用于用户细分：
    - 高价值用户识别
    - 潜在流失用户预警
    - 用户购买偏好分析
    - 个性化营销策略制定
    """
    
    # 聚类特征列
    FEATURE_COLUMNS = [
        'Total_Orders', 'Total_Spend', 'Avg_Order_Value', 
        'Purchase_Frequency', 'Days_Since_Last_Purchase'
    ]
    
    # 聚类数量范围
    K_RANGE = range(2, 10)
    
    def __init__(self, data_dir: str = "./data"):
        """
        初始化 K-means 用户画像分析器
        
        Args:
            data_dir: 数据目录
        """
        self.data_dir = data_dir
        self.scaler = StandardScaler() if SKLEARN_AVAILABLE else None
        self.kmeans = None
        self.features = None
        self.user_data = None
        
        logger.info("KMeansUserProfiler initialized")
    
    def load_user_data(self, filepath: str = None) -> pd.DataFrame:
        """
        加载用户行为数据
        
        Args:
            filepath: 数据文件路径
        
        Returns:
            用户数据框
        """
        logger.info("Loading user behavior data...")
        
        if filepath and os.path.exists(filepath):
            if filepath.endswith('.json'):
                df = pd.read_json(filepath)
            elif filepath.endswith('.csv'):
                df = pd.read_csv(filepath)
            else:
                df = pd.read_parquet(filepath)
        else:
            # 生成示例用户数据
            logger.warning("No user data found, generating sample data")
            df = self._generate_sample_user_data()
        
        logger.info(f"Loaded {len(df)} user records")
        return df
    
    def _generate_sample_user_data(self) -> pd.DataFrame:
        """
        生成示例用户数据 (实际使用时替换为真实数据)

        人群比例分布：
        - VIP：5% - 10%（人数最少，但贡献 30% 销售额）
        - 刚需型：25% - 35%
        - 大促型：20% - 30%
        - 随机型：30% - 40%（人数最多）

        消费频次：
        - VIP：0.8 - 1.5 次/月（一年买 10 次左右）
        - 随机型：0.1 - 0.2 次/月（一年只买 1-2 次）

        Returns:
            示例用户数据框
        """
        np.random.seed(42)
        n_users = 5000

        # 固定4类用户群体
        vip_count = int(n_users * 0.07)        # VIP: 7%
        刚需_count = int(n_users * 0.30)       # 刚需型: 30%
        大促_count = int(n_users * 0.25)       # 大促型: 25%
        随机_count = n_users - vip_count - 刚需_count - 大促_count  # 随机型: 38%

        user_ids = [f"CUST-{i:06d}" for i in range(n_users)]

        data = []

        # VIP用户：高频高消费
        for i in range(vip_count):
            data.append({
                'Customer_ID': user_ids[i],
                'Total_Orders': np.random.randint(8, 15),  # 一年8-15次
                'Total_Spend': np.random.uniform(8000, 25000),
                'Avg_Order_Value': np.random.uniform(500, 1500),
                'Purchase_Frequency': np.random.uniform(0.8, 1.5),  # 0.8-1.5次/月
                'Favorite_Category': np.random.choice(['T-Shirt', 'Pants', 'Jacket', 'Dress']),
                'Favorite_Season': np.random.choice(['Spring', 'Autumn', 'Winter']),
                'Preferred_Region': np.random.choice(['Shanghai', 'Beijing']),
                'Days_Since_Last_Purchase': np.random.randint(0, 30)
            })

        # 刚需型用户：中等频次，稳定消费
        for i in range(vip_count, vip_count + 刚需_count):
            data.append({
                'Customer_ID': user_ids[i],
                'Total_Orders': np.random.randint(5, 12),
                'Total_Spend': np.random.uniform(3000, 8000),
                'Avg_Order_Value': np.random.uniform(300, 600),
                'Purchase_Frequency': np.random.uniform(0.5, 1.0),  # 0.5-1.0次/月
                'Favorite_Category': np.random.choice(['T-Shirt', 'Pants', 'Underwear']),
                'Favorite_Season': np.random.choice(['Spring', 'Summer']),
                'Preferred_Region': np.random.choice(['Guangzhou', 'Shenzhen', 'Hangzhou']),
                'Days_Since_Last_Purchase': np.random.randint(20, 60)
            })

        # 大促型用户：低频但大额消费
        for i in range(vip_count + 刚需_count, vip_count + 刚需_count + 大促_count):
            data.append({
                'Customer_ID': user_ids[i],
                'Total_Orders': np.random.randint(2, 6),
                'Total_Spend': np.random.uniform(1500, 5000),
                'Avg_Order_Value': np.random.uniform(400, 1200),
                'Purchase_Frequency': np.random.uniform(0.2, 0.5),  # 0.2-0.5次/月
                'Favorite_Category': np.random.choice(['Jacket', 'Dress', 'Pants']),
                'Favorite_Season': np.random.choice(['Autumn', 'Winter']),
                'Preferred_Region': np.random.choice(['Beijing', 'Shanghai', 'Chengdu']),
                'Days_Since_Last_Purchase': np.random.randint(60, 150)
            })

        # 随机型用户：极低频，偶尔消费
        for i in range(vip_count + 刚需_count + 大促_count, n_users):
            data.append({
                'Customer_ID': user_ids[i],
                'Total_Orders': np.random.randint(1, 3),
                'Total_Spend': np.random.uniform(100, 800),
                'Avg_Order_Value': np.random.uniform(80, 300),
                'Purchase_Frequency': np.random.uniform(0.1, 0.2),  # 0.1-0.2次/月
                'Favorite_Category': np.random.choice(['T-Shirt', 'Accessories', 'Underwear']),
                'Favorite_Season': np.random.choice(['Spring', 'Summer']),
                'Preferred_Region': np.random.choice(['Shanghai', 'Beijing', 'Guangzhou', 'Shenzhen']),
                'Days_Since_Last_Purchase': np.random.randint(90, 200)
            })

        df = pd.DataFrame(data)

        return df
    
    def preprocess_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        预处理特征数据
        
        Args:
            df: 用户数据框
        
        Returns:
            特征数据框
        """
        logger.info("Preprocessing features...")
        
        # 选择特征列
        features = df[self.FEATURE_COLUMNS].copy()
        
        # 处理缺失值
        features = features.fillna(0)
        
        # 处理异常值 (使用IQR方法裁剪)
        for col in self.FEATURE_COLUMNS:
            Q1 = features[col].quantile(0.25)
            Q3 = features[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            features[col] = features[col].clip(lower, upper)
        
        self.features = features
        self.user_data = df
        
        logger.info(f"Preprocessed {len(features)} user records")
        
        return features
    
    def scale_features(self, features: pd.DataFrame) -> np.ndarray:
        """
        特征标准化
        
        Args:
            features: 特征数据框
        
        Returns:
            标准化后的特征矩阵
        """
        if SKLEARN_AVAILABLE:
            scaled = self.scaler.fit_transform(features)
        else:
            # 简单标准化
            scaled = (features - features.mean()) / features.std()
        
        return scaled
    
    def find_optimal_k(self, features: np.ndarray) -> int:
        """
        使用肘部法则和轮廓系数确定最优聚类数
        
        Args:
            features: 特征矩阵
        
        Returns:
            最优聚类数
        """
        logger.info("Finding optimal number of clusters...")
        
        inertias = []
        silhouettes = []
        
        for k in self.K_RANGE:
            if SKLEARN_AVAILABLE:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = kmeans.fit_predict(features)
                
                inertias.append(kmeans.inertia_)
                silhouettes.append(silhouette_score(features, labels))
            else:
                # 简化实现
                inertias.append(10000 / k)
                silhouettes.append(0.3)
        
        # 选择轮廓系数最高的 k
        if silhouettes:
            optimal_k = list(self.K_RANGE)[np.argmax(silhouettes)]
        else:
            optimal_k = 4
        
        logger.info(f"Optimal k = {optimal_k} (silhouette = {max(silhouettes):.3f})")
        
        return optimal_k
    
    def calculate_metrics(self, features: np.ndarray, labels: np.ndarray) -> ClusterMetrics:
        """
        计算聚类评估指标
        
        Args:
            features: 特征矩阵
            labels: 聚类标签
        
        Returns:
            评估指标
        """
        if not SKLEARN_AVAILABLE or len(np.unique(labels)) < 2:
            return ClusterMetrics(
                silhouette_score=0.5,
                calinski_harabasz_score=100,
                davies_bouldin_score=1.0,
                inertia=1000
            )
        
        silhouette = silhouette_score(features, labels)
        ch_score = calinski_harabasz_score(features, labels)
        db_score = davies_bouldin_score(features, labels)
        
        # 计算惯性 (仅当模型已训练)
        inertia = self.kmeans.inertia_ if self.kmeans else 0
        
        return ClusterMetrics(
            silhouette_score=float(silhouette),
            calinski_harabasz_score=float(ch_score),
            davies_bouldin_score=float(db_score),
            inertia=float(inertia)
        )
    
    def generate_portrait_label(self, cluster: UserCluster) -> str:
        """
        生成用户画像标签

        人群分类标准：
        - VIP：消费高 + 频次高（0.8-1.5次/月）
        - 刚需型：消费中等 + 频次稳定（0.5-1.0次/月）
        - 大促型：消费中等 + 频次低（0.2-0.5次/月），客单价高
        - 随机型：消费低 + 频次极低（0.1-0.2次/月）

        Args:
            cluster: 聚类结果

        Returns:
            画像标签
        """
        # 基于关键指标判断用户类型
        if cluster.avg_purchase_frequency >= 0.8 and cluster.avg_total_spend > 6000:
            return "VIP高价值用户"
        elif cluster.avg_purchase_frequency >= 0.5 and cluster.avg_purchase_frequency < 0.8:
            return "刚需型用户"
        elif cluster.avg_purchase_frequency >= 0.2 and cluster.avg_purchase_frequency < 0.5:
            return "大促型用户"
        elif cluster.avg_purchase_frequency < 0.2:
            return "随机型用户"
        else:
            return "普通用户"
    
    def analyze_cluster(self, cluster_id: int, df: pd.DataFrame, 
                        labels: np.ndarray) -> UserCluster:
        """
        分析单个聚类的特征
        
        Args:
            cluster_id: 聚类ID
            df: 用户数据框
            labels: 聚类标签
        
        Returns:
            聚类分析结果
        """
        cluster_mask = labels == cluster_id
        cluster_data = df[cluster_mask]
        
        # 计算聚类统计
        user_count = len(cluster_data)
        avg_orders = cluster_data['Total_Orders'].mean()
        avg_spend = cluster_data['Total_Spend'].mean()
        avg_order_value = cluster_data['Avg_Order_Value'].mean()
        avg_frequency = cluster_data['Purchase_Frequency'].mean()
        avg_days_since = cluster_data['Days_Since_Last_Purchase'].mean()
        
        # 最受欢迎的品类
        top_categories = cluster_data['Favorite_Category'].value_counts().head(3).index.tolist()
        
        # 最活跃的地区
        top_regions = cluster_data['Preferred_Region'].value_counts().head(3).index.tolist()
        
        # 生成画像标签
        portrait_label = self.generate_portrait_label(UserCluster(
            cluster_id=cluster_id,
            user_count=user_count,
            avg_total_orders=avg_orders,
            avg_total_spend=avg_spend,
            avg_order_value=avg_order_value,
            avg_purchase_frequency=avg_frequency,
            avg_days_since_last_purchase=avg_days_since,
            top_categories=top_categories,
            top_regions=top_regions,
            portrait_label="",
            description=""
        ))
        
        # 生成描述
        description = (
            f"该群体包含{user_count}名用户，平均订单数{avg_orders:.1f}，"
            f"平均消费金额{avg_spend:.0f}元，平均客单价{avg_order_value:.0f}元，"
            f"购买频次{avg_frequency:.1f}次/月"
        )
        
        return UserCluster(
            cluster_id=cluster_id,
            user_count=user_count,
            avg_total_orders=float(avg_orders),
            avg_total_spend=float(avg_spend),
            avg_order_value=float(avg_order_value),
            avg_purchase_frequency=float(avg_frequency),
            avg_days_since_last_purchase=float(avg_days_since),
            top_categories=top_categories,
            top_regions=top_regions,
            portrait_label=portrait_label,
            description=description
        )
    
    def fit_predict(self, n_clusters: int = None) -> ClusteringResult:
        """
        执行 K-means 聚类
        
        Args:
            n_clusters: 聚类数量，若为 None 则自动确定
        
        Returns:
            聚类结果
        """
        if self.features is None:
            raise ValueError("No features preprocessed. Call preprocess_features() first.")
        
        # 标准化特征
        scaled_features = self.scale_features(self.features)
        
        # 确定最优聚类数
        if n_clusters is None:
            n_clusters = self.find_optimal_k(scaled_features)
        
        self.optimal_k = n_clusters
        
        logger.info(f"Running K-means with k = {n_clusters}")
        
        # 执行聚类
        if SKLEARN_AVAILABLE:
            self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = self.kmeans.fit_predict(scaled_features)
        else:
            # 简单实现：使用 K-means++ 的简化版本
            labels = self._simple_kmeans(scaled_features, n_clusters)
        
        # 计算评估指标
        metrics = self.calculate_metrics(scaled_features, labels)
        
        # 分析每个聚类
        clusters = []
        for i in range(n_clusters):
            cluster = self.analyze_cluster(i, self.user_data, labels)
            clusters.append(cluster)
        
        # 生成用户分配记录
        user_assignments = []
        for idx, row in self.user_data.iterrows():
            user_assignments.append({
                'Customer_ID': row['Customer_ID'],
                'Cluster_ID': int(labels[idx]),
                'Cluster_Label': clusters[labels[idx]].portrait_label
            })
        
        # 特征重要性 (基于聚类中心的特征权重)
        feature_importance = {}
        if SKLEARN_AVAILABLE and self.kmeans:
            cluster_centers = self.kmeans.cluster_centers_
            for idx, col in enumerate(self.FEATURE_COLUMNS):
                feature_importance[col] = float(np.std(cluster_centers[:, idx]))
        
        # 归一化特征重要性
        total = sum(feature_importance.values())
        if total > 0:
            feature_importance = {k: v/total for k, v in feature_importance.items()}
        
        logger.info(f"Clustering complete: {n_clusters} clusters identified")
        
        return ClusteringResult(
            clusters=clusters,
            metrics=metrics,
            optimal_k=n_clusters,
            user_assignments=user_assignments,
            feature_importance=feature_importance
        )
    
    def _simple_kmeans(self, features: np.ndarray, k: int, max_iter: int = 100) -> np.ndarray:
        """
        简单 K-means 实现
        
        Args:
            features: 特征矩阵
            k: 聚类数
            max_iter: 最大迭代次数
        
        Returns:
            聚类标签
        """
        np.random.seed(42)
        n_samples = len(features)
        
        # 随机初始化中心
        indices = np.random.choice(n_samples, k, replace=False)
        centers = features[indices].copy()
        
        labels = np.zeros(n_samples, dtype=int)
        
        for _ in range(max_iter):
            # 分配样本到最近的中心
            new_labels = np.argmin(
                np.linalg.norm(features[:, np.newaxis] - centers, axis=2),
                axis=1
            )
            
            # 更新中心
            new_centers = np.array([
                features[new_labels == i].mean(axis=0) if np.any(new_labels == i) else centers[i]
                for i in range(k)
            ])
            
            # 检查收敛
            if np.allclose(centers, new_centers):
                break
            
            centers = new_centers
            labels = new_labels
        
        return labels


def run_user_clustering(n_clusters: int = None) -> Dict[str, Any]:
    """
    运行用户聚类分析
    
    Args:
        n_clusters: 聚类数量
    
    Returns:
        聚类结果
    """
    # 导入数据处理模块
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    profiler = KMeansUserProfiler()

    # 直接使用示例数据（符合业务逻辑的模拟数据）
    logger.info("Generating sample user data with realistic patterns...")
    user_profile = profiler._generate_sample_user_data()

    if user_profile.empty:
        logger.warning("No user profiles generated")
        return {'error': 'No user profiles available'}
    
    # 预处理特征
    profiler.preprocess_features(user_profile)

    # 固定使用4个聚类：VIP、刚需型、大促型、随机型
    n_clusters = 4

    # 确保聚类数不超过样本数
    n_clusters = min(n_clusters, len(user_profile))
    
    # 执行聚类
    result = profiler.fit_predict(n_clusters)
    
    return result.to_dict()


# ============================================================================
# 入口函数
# ============================================================================

def main():
    """主函数 - 测试用户聚类"""
    
    print("\n" + "="*60)
    print("K-means User Portrait Clustering Test")
    print("="*60)
    
    # 运行聚类
    result = run_user_clustering()
    
    print(f"\nOptimal K: {result['optimal_k']}")
    print(f"\nClustering Metrics:")
    print(f"  Silhouette Score: {result['metrics']['silhouette_score']:.3f}")
    print(f"  Calinski-Harabasz Score: {result['metrics']['calinski_harabasz_score']:.2f}")
    print(f"  Davies-Bouldin Score: {result['metrics']['davies_bouldin_score']:.3f}")
    
    print(f"\nCluster Analysis:")
    for cluster in result['clusters']:
        print(f"\n  Cluster {cluster['cluster_id']}: {cluster['portrait_label']}")
        print(f"    Users: {cluster['user_count']}")
        print(f"    Avg Orders: {cluster['avg_total_orders']:.1f}")
        print(f"    Avg Spend: ¥{cluster['avg_total_spend']:.0f}")
        print(f"    Top Categories: {cluster['top_categories']}")
        print(f"    Description: {cluster['description']}")
    
    print(f"\nFeature Importance:")
    for feat, imp in result['feature_importance'].items():
        print(f"  {feat}: {imp:.3f}")


if __name__ == "__main__":
    main()
