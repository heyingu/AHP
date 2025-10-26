# src/pruning/semantic_pruner.py
import logging
import numpy as np
from sentence_transformers import SentenceTransformer, util # 依赖 sentence-transformers 库
from typing import List, Union
from .base_pruner import BasePruner
import torch

class SemanticPruner(BasePruner):
    """使用句子嵌入间的语义相似度来剪枝候选句子。"""

    # 修改 __init__，不再需要 model 和 tokenizer，因为它使用独立的 SentenceTransformer
    def __init__(self, threshold: float = 0.8, model_name: str = 'all-MiniLM-L6-v2', device: Union[str, torch.device] = 'cuda'):
        """
        初始化语义相似度剪枝器。

        Args:
            threshold (float): 语义相似度阈值 (例如余弦相似度)。高于此阈值的候选将被保留。默认值 0.8。
            model_name (str): 用于计算句子嵌入的 SentenceTransformer 模型名称或路径。
            device (Union[str, torch.device]): 计算设备 ('cuda' 或 'cpu')。
        """
        super().__init__(threshold)
        self.device = torch.device(device)
        # 加载 SentenceTransformer 模型
        try:
            self.embedding_model = SentenceTransformer(model_name, device=self.device)
            logging.info(f"语义相似度剪枝器已初始化，模型: {model_name}, 阈值 (保留 Sim >=): {self.threshold}")
        except Exception as e:
            logging.error(f"加载 SentenceTransformer 模型 '{model_name}' 失败: {e}")
            raise RuntimeError(f"无法加载嵌入模型 '{model_name}'")

    def calculate_similarity(self, text1: str, text2: str) -> float:
        """
        计算两个文本之间的余弦相似度。

        Args:
            text1 (str): 第一个文本。
            text2 (str): 第二个文本。

        Returns:
            float: 两个文本嵌入之间的余弦相似度。
        """
        try:
            # 计算两个文本的嵌入向量
            embeddings = self.embedding_model.encode([text1, text2], convert_to_tensor=True, device=self.device)
            # 计算余弦相似度
            cosine_score = util.pytorch_cos_sim(embeddings[0], embeddings[1])
            return cosine_score.item()
        except Exception as e:
            logging.error(f"计算文本间相似度时出错: {e}", exc_info=True)
            return 0.0 # 出错时返回最低相似度

    def prune(self, original_text: str, candidates: List[str], **kwargs) -> List[str]:
        """
        根据与原始文本的语义相似度阈值筛选候选句子列表。

        Args:
            original_text (str): 用于比较的原始文本。
            candidates (List[str]): 待剪枝的候选句子列表。
            **kwargs: 其他可能的上下文信息 (例如 masked_text)。

        Returns:
            List[str]: 与原始文本语义相似度高于阈值的候选句子列表。
        """
        pruned_candidates = []
        scores = []
        if not candidates:
            return []

        # 计算原始文本的嵌入（只需一次）
        try:
            original_embedding = self.embedding_model.encode(original_text, convert_to_tensor=True, device=self.device)
        except Exception as e:
            logging.error(f"计算原始文本嵌入时出错: {e}", exc_info=True)
            return [] # 无法计算原始嵌入，无法进行比较

        # 计算所有候选的嵌入（批量计算效率更高）
        try:
            candidate_embeddings = self.embedding_model.encode(candidates, convert_to_tensor=True, device=self.device, batch_size=32) # 使用合适的批次大小
        except Exception as e:
            logging.error(f"计算候选文本嵌入时出错: {e}", exc_info=True)
            return [] # 无法计算候选嵌入，无法进行比较

        # 计算原始文本与所有候选的余弦相似度
        cosine_scores = util.pytorch_cos_sim(original_embedding, candidate_embeddings)[0].cpu().numpy() # 获取第一行（原始文本与所有候选的相似度）并转为numpy

        # 根据阈值筛选
        for i, score in enumerate(cosine_scores):
            scores.append(score)
            if score >= self.threshold:
                pruned_candidates.append(candidates[i])
            else:
                logging.debug(f"剪枝候选 (Sim={score:.3f} < {self.threshold}): {candidates[i][:50]}...")

        logging.debug(f"语义相似度剪枝: 原始 {len(candidates)} 个候选, 剩余 {len(pruned_candidates)} 个。最高 Sim: {max(scores):.3f} (如果有候选)")

        # 可选：如果没有候选通过，是否返回最佳的一个？
        # if not pruned_candidates and candidates:
        #     best_idx = np.argmax(scores)
        #     logging.warning(f"语义剪枝后无候选剩余，将返回相似度最高的候选 (Sim={scores[best_idx]:.3f})。")
        #     return [candidates[best_idx]]

        return pruned_candidates