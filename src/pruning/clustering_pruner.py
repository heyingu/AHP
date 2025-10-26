# src/pruning/clustering_pruner.py
import logging
import numpy as np
from sentence_transformers import SentenceTransformer # 依赖 sentence-transformers
from sklearn.cluster import KMeans # 依赖 scikit-learn
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Union
from .base_pruner import BasePruner
import torch

class ClusteringPruner(BasePruner):
    """使用 K-Means 聚类对候选句子的嵌入进行聚类，并从每个簇中选择代表性样本。"""

    # __init__ 不再需要主模型 model 和 tokenizer，因为它使用独立的 SentenceTransformer
    def __init__(self, n_clusters: int = 5, model_name: str = 'all-MiniLM-L6-v2', device: Union[str, torch.device] = 'cuda'):
        """
        初始化聚类剪枝器。

        Args:
            n_clusters (int): K-Means 算法的目标簇数量。默认值 5。
                              这个值在这里扮演了 'threshold' 的角色。
            model_name (str): 用于计算句子嵌入的 SentenceTransformer 模型名称或路径。
            device (Union[str, torch.device]): 计算设备 ('cuda' 或 'cpu')。
        """
        super().__init__(n_clusters) # n_clusters 存放在 self.threshold 中
        self.n_clusters = n_clusters # 也可以单独存一份
        self.device = torch.device(device)
        # 加载 SentenceTransformer 模型
        try:
            self.embedding_model = SentenceTransformer(model_name, device=self.device)
            logging.info(f"聚类剪枝器已初始化，模型: {model_name}, 目标簇数: {self.n_clusters}")
        except Exception as e:
            logging.error(f"加载 SentenceTransformer 模型 '{model_name}' 失败: {e}")
            raise RuntimeError(f"无法加载嵌入模型 '{model_name}'")

    def prune(self, original_text: str, candidates: List[str], **kwargs) -> List[str]:
        """
        对候选句子进行聚类，并从每个簇中选择最接近簇中心的样本。

        Args:
            original_text (str): 原始文本（当前未使用，但保留接口一致性）。
            candidates (List[str]): 待剪枝的候选句子列表。
            **kwargs: 其他可能的上下文信息。

        Returns:
            List[str]: 从每个簇中选出的代表性候选句子列表。
                     返回列表的长度最多为 n_clusters。
        """
        pruned_candidates = []
        if not candidates:
            return []

        # 如果候选数量少于或等于目标簇数，无需聚类，直接返回所有候选
        if len(candidates) <= self.n_clusters:
            logging.debug(f"候选数量 ({len(candidates)}) 不多于目标簇数 ({self.n_clusters})，跳过聚类。")
            return candidates

        # --- 计算所有候选的嵌入 ---
        try:
            candidate_embeddings = self.embedding_model.encode(candidates, convert_to_numpy=True, device=self.device, batch_size=32) # KMeans 需要 Numpy 数组
            logging.debug(f"已计算 {len(candidates)} 个候选的嵌入，形状: {candidate_embeddings.shape}")
        except Exception as e:
            logging.error(f"计算候选嵌入时出错: {e}", exc_info=True)
            return candidates # Fallback: 返回所有候选

        # --- 执行 K-Means 聚类 ---
        try:
            # 确保 n_clusters 不大于样本数
            actual_n_clusters = min(self.n_clusters, len(candidates))
            if actual_n_clusters != self.n_clusters:
                 logging.warning(f"目标簇数 ({self.n_clusters}) 大于候选数 ({len(candidates)})，将使用 {actual_n_clusters} 个簇。")

            kmeans = KMeans(n_clusters=actual_n_clusters, random_state=42, n_init=10) # n_init='auto' in newer sklearn
            kmeans.fit(candidate_embeddings)
            cluster_centers = kmeans.cluster_centers_ # 获取每个簇的中心点
            labels = kmeans.labels_ # 获取每个候选所属的簇标签
            logging.debug(f"已将候选聚类到 {actual_n_clusters} 个簇中。")
        except Exception as e:
            logging.error(f"执行 K-Means 聚类时出错: {e}", exc_info=True)
            return candidates # Fallback: 返回所有候选

        # --- 从每个簇中选择最接近中心的样本 ---
        for i in range(actual_n_clusters):
            # 获取属于当前簇 i 的所有候选的索引
            indices_in_cluster = np.where(labels == i)[0]

            if len(indices_in_cluster) == 0:
                # 理论上 KMeans 不会产生空簇，但以防万一
                logging.warning(f"聚类发现空簇 {i}，跳过。")
                continue

            # 获取当前簇的嵌入和中心点
            cluster_embeddings = candidate_embeddings[indices_in_cluster]
            center = cluster_centers[i]

            # 计算簇内每个点到簇中心的距离（可以使用余弦相似度或欧氏距离）
            # 这里使用余弦相似度，选择相似度最高的（最接近中心的）
            similarities = cosine_similarity(cluster_embeddings, center.reshape(1, -1)).flatten()

            # 找到相似度最高的那个候选在簇内的索引
            closest_index_in_cluster = np.argmax(similarities)
            # 映射回原始候选列表中的索引
            original_index = indices_in_cluster[closest_index_in_cluster]

            # 将最接近中心的候选添加到结果列表
            pruned_candidates.append(candidates[original_index])
            logging.debug(f"簇 {i}: 选择索引 {original_index} (相似度={similarities[closest_index_in_cluster]:.3f}) -> '{candidates[original_index][:50]}...'")

        logging.debug(f"聚类剪枝: 原始 {len(candidates)} 个候选, 剩余 {len(pruned_candidates)} 个。")
        return pruned_candidates