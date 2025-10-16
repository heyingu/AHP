# ahp_robustness/src/pruning/clustering_pruner.py

from src.pruning.base_pruner import BasePruner # <--- 修改点
from src.models.model_loader import load_pruning_model
from sklearn.cluster import KMeans
import numpy as np

class ClusteringPruner(BasePruner):
    """
    使用聚类方法进行剪枝，保证候选的多样性。
    将M个候选聚为K个簇，从每个簇中选择最接近中心点的代表。
    """
    def load_model(self):
        # 聚类同样需要句子嵌入模型
        return load_pruning_model(model_name="sentence-transformers/all-mpnet-base-v2")

    def prune(self, original_text, candidates):
        if not candidates or len(candidates) < self.k:
            return candidates # 如果候选数量不足K，直接返回全部
        
        # 1. 将所有候选编码为向量
        candidate_embeddings = self.model.encode(candidates)
        
        # 2. 使用K-Means算法进行聚类
        kmeans = KMeans(n_clusters=self.k, random_state=42, n_init=10)
        kmeans.fit(candidate_embeddings)
        
        pruned_candidates = []
        # 3. 为每个簇找到最接近中心点的候选
        for i in range(self.k):
            # 获取当前簇的所有点的索引
            cluster_indices = np.where(kmeans.labels_ == i)[0]
            if len(cluster_indices) == 0:
                continue

            # 获取簇中心
            cluster_center = kmeans.cluster_centers_[i]
            
            # 获取簇内所有点的向量
            cluster_embeddings = candidate_embeddings[cluster_indices]
            
            # 计算每个点到中心点的距离
            distances = np.linalg.norm(cluster_embeddings - cluster_center, axis=1)
            
            # 找到距离最近的那个点的索引（在簇内的相对索引）
            closest_in_cluster_idx = np.argmin(distances)
            
            # 转换回在原始candidates列表中的绝对索引
            original_idx = cluster_indices[closest_in_cluster_idx]
            
            pruned_candidates.append(candidates[original_idx])
            
        return pruned_candidates