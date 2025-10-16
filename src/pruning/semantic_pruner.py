# ahp_robustness/src/pruning/semantic_pruner.py

from src.pruning.base_pruner import BasePruner # <--- 修改点
from src.models.model_loader import load_pruning_model
from sentence_transformers.util import cos_sim
import torch

class SemanticPruner(BasePruner):
    """
    使用语义相似度进行剪枝。
    选择与原始文本余弦相似度最高的K个候选。
    """
    def load_model(self):
        # 使用model_loader加载预训练的句子嵌入模型
        return load_pruning_model(model_name="sentence-transformers/all-mpnet-base-v2")

    def prune(self, original_text, candidates):
        if not candidates:
            return []
        
        # 1. 将原始文本和所有候选编码为向量
        original_embedding = self.model.encode(original_text, convert_to_tensor=True)
        candidate_embeddings = self.model.encode(candidates, convert_to_tensor=True)
        
        # 2. 计算余弦相似度
        cosine_scores = cos_sim(original_embedding, candidate_embeddings)
        
        # 3. 找出分数最高的K个候选
        # 我们使用torch.topk来高效地找到最高分的索引
        top_k_scores, top_k_indices = torch.topk(cosine_scores[0], min(self.k, len(candidates)))
        
        # 4. 根据索引选出最优候选
        pruned_candidates = [candidates[i] for i in top_k_indices]
        
        return pruned_candidates