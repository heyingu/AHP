# src/pruning/nli_pruner.py
import logging
import numpy as np
from sentence_transformers import CrossEncoder # 使用 CrossEncoder 计算 NLI 分数
from typing import List, Union
from .base_pruner import BasePruner
import torch

class NLIPruner(BasePruner):
    """使用自然语言推断 (NLI) 模型判断候选与原始文本的关系来剪枝。"""

    # __init__ 不再需要主模型 model 和 tokenizer，因为它使用独立的 CrossEncoder
    def __init__(self, threshold: float = 0.5, model_name: str = 'cross-encoder/nli-roberta-base', device: Union[str, torch.device] = 'cuda'):
        """
        初始化 NLI 剪枝器。

        Args:
            threshold (float): NLI "蕴含 (entailment)" 分数的阈值。高于此阈值的候选将被保留。默认值 0.5。
            model_name (str): 用于计算 NLI 分数的 CrossEncoder 模型名称或路径。
            device (Union[str, torch.device]): 计算设备 ('cuda' 或 'cpu')。
        """
        super().__init__(threshold)
        self.device = torch.device(device)
        # 加载 NLI CrossEncoder 模型
        try:
            # NLI 模型通常输出三个分数：[contradiction, entailment, neutral]
            self.nli_model = CrossEncoder(model_name, device=self.device)
            logging.info(f"NLI 剪枝器已初始化，模型: {model_name}, 阈值 (保留 Entailment >=): {self.threshold}")
        except Exception as e:
            logging.error(f"加载 NLI CrossEncoder 模型 '{model_name}' 失败: {e}")
            raise RuntimeError(f"无法加载 NLI 模型 '{model_name}'")

    def calculate_nli_scores(self, text_pairs: List[List[str]]) -> np.ndarray:
        """
        批量计算文本对的 NLI 分数。

        Args:
            text_pairs (List[List[str]]): 文本对列表，每个元素是 [premise, hypothesis]。

        Returns:
            np.ndarray: NLI 分数数组，形状为 [num_pairs, num_labels] (通常是 3)。
        """
        try:
            # 使用 CrossEncoder 的 predict 方法计算分数
            scores = self.nli_model.predict(text_pairs, apply_softmax=True, batch_size=32) # 使用合适的批次大小
            return scores
        except Exception as e:
            logging.error(f"计算 NLI 分数时出错: {e}", exc_info=True)
            # 返回一个默认值，例如全零或均匀分布
            return np.ones((len(text_pairs), 3)) / 3.0

    def prune(self, original_text: str, candidates: List[str], **kwargs) -> List[str]:
        """
        根据候选句子与原始文本之间的 NLI "蕴含" 分数进行剪枝。
        我们假设原始文本是前提 (premise)，候选句子是假设 (hypothesis)。

        Args:
            original_text (str): 作为前提 (premise) 的原始文本。
            candidates (List[str]): 作为假设 (hypothesis) 的待剪枝候选句子列表。
            **kwargs: 其他可能的上下文信息。

        Returns:
            List[str]: NLI 模型判断为 "蕴含" (且分数高于阈值) 的候选句子列表。
        """
        pruned_candidates = []
        entailment_scores = []
        if not candidates:
            return []

        # 构建 NLI 模型需要的输入对：[original_text, candidate]
        nli_input_pairs = [[original_text, cand] for cand in candidates]

        # 批量计算所有候选与原始文本的 NLI 分数
        nli_scores = self.calculate_nli_scores(nli_input_pairs) # 返回 [num_candidates, 3]

        # NLI 模型输出通常是 [contradiction, entailment, neutral] 的顺序
        # 我们关心的是 entailment 分数 (索引 1)
        # 注意：不同模型的输出顺序可能不同，请根据您使用的模型确认
        entailment_idx = 1 # 假设蕴含分数在索引 1

        # 根据蕴含分数和阈值进行筛选
        for i, score_triple in enumerate(nli_scores):
            entailment_score = score_triple[entailment_idx]
            entailment_scores.append(entailment_score)
            if entailment_score >= self.threshold:
                pruned_candidates.append(candidates[i])
            else:
                # 记录被剪枝的候选及其蕴含分数
                logging.debug(f"剪枝候选 (Entailment={entailment_score:.3f} < {self.threshold}): {candidates[i][:50]}...")

        logging.debug(f"NLI 剪枝: 原始 {len(candidates)} 个候选, 剩余 {len(pruned_candidates)} 个。最高 Entailment: {max(entailment_scores):.3f} (如果有候选)")

        # 可选：如果没有候选通过，是否返回得分最高的一个？
        # if not pruned_candidates and candidates:
        #     best_idx = np.argmax(entailment_scores)
        #     logging.warning(f"NLI 剪枝后无候选剩余，将返回蕴含分数最高的候选 (Entailment={entailment_scores[best_idx]:.3f})。")
        #     return [candidates[best_idx]]

        return pruned_candidates