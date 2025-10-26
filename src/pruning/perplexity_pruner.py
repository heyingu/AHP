# src/pruning/perplexity_pruner.py
import torch
import logging
import numpy as np
from transformers import PreTrainedModel, PreTrainedTokenizer
from typing import List, Union
from .base_pruner import BasePruner
import math # 引入 math

class PerplexityPruner(BasePruner):
    """使用语言模型计算的困惑度 (Perplexity) 分数来剪枝候选句子。"""

    # 修改 __init__ 以接收 model 和 tokenizer
    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, threshold: float = 100.0, device: Union[str, torch.device] = 'cuda'):
        """
        初始化困惑度剪枝器。

        Args:
            model (PreTrainedModel): 用于计算困惑度的语言模型实例。
            tokenizer (PreTrainedTokenizer): 模型对应的分词器实例。
            threshold (float): 困惑度阈值。低于此阈值的候选将被保留。默认值 100.0。
            device (Union[str, torch.device]): 计算设备 ('cuda' 或 'cpu')。
        """
        super().__init__(threshold) # 调用基类初始化
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device(device)
        logging.info(f"困惑度剪枝器已初始化，阈值 (保留 PPL <): {self.threshold}")

    @torch.no_grad() # 计算困惑度不需要梯度
    def calculate_score(self, text: str) -> float:
        """
        计算给定文本的困惑度。困惑度越低，表示文本越流畅、越符合语言模型。

        Args:
            text (str): 需要计算困惑度的文本。

        Returns:
            float: 文本的困惑度分数。如果计算失败则返回无穷大 (float('inf'))。
        """
        score = float('inf') # 默认值，表示非常差
        # 确保文本不为空
        if not text:
            logging.warning("尝试为 LLaMA 模型计算困惑度时收到空文本，返回 inf。")
            return score
        try:
            # 使用分词器对文本进行编码
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=self.model.config.max_position_embeddings).to(self.device)
            input_ids = inputs["input_ids"]

            # 对于自回归模型 (如 LLaMA, GPT)，计算损失时 labels 通常就是 input_ids
            # 模型会内部处理移位以预测下一个 token
            outputs = self.model(**inputs, labels=input_ids)
            neg_log_likelihood = outputs.loss # 获取损失 (通常是序列的平均负对数似然)

            # 计算困惑度 PPL = exp(loss)
            # 检查 loss 是否有效且序列长度大于0
            if input_ids.shape[1] > 0 and neg_log_likelihood is not None:
                # 检查 loss 是否为 NaN 或无穷大
                if not math.isnan(neg_log_likelihood.item()) and not math.isinf(neg_log_likelihood.item()):
                    score = torch.exp(neg_log_likelihood).item()
                else:
                    logging.warning(f"为文本 '{text[:50]}...' 计算的 LLaMA 损失无效 (NaN/inf)，返回 inf。")
            else:
                logging.warning(f"无法为文本 '{text[:50]}...' 计算 LLaMA 困惑度（输入长度或损失问题），返回 inf。")

        except Exception as e:
            logging.error(f"使用 LLaMA 模型计算困惑度时出错: {e}", exc_info=True) # exc_info=True 会记录详细错误堆栈

        return score

    def prune(self, original_text: str, candidates: List[str], **kwargs) -> List[str]:
        """
        根据困惑度阈值筛选候选句子列表。

        Args:
            original_text (str): 原始文本（当前未使用，但保留接口一致性）。
            candidates (List[str]): 待剪枝的候选句子列表。
            **kwargs: 其他可能的上下文信息 (例如 masked_text)。

        Returns:
            List[str]: 困惑度低于阈值的候选句子列表。
        """
        pruned_candidates = [] # 存储通过剪枝的候选
        scores = [] # 存储每个候选的分数，方便调试
        if not candidates:
            return [] # 如果没有候选，直接返回空列表

        # 遍历所有候选
        for candidate in candidates:
            score = self.calculate_score(candidate) # 计算当前候选的困惑度
            scores.append(score)
            # 如果分数低于设定的阈值，则保留该候选
            if score < self.threshold:
                pruned_candidates.append(candidate)
            else:
                # 记录被剪枝的候选及其分数（可选，用于调试）
                logging.debug(f"剪枝候选 (PPL={score:.2f} >= {self.threshold}): {candidate[:50]}...")

        logging.debug(f"困惑度剪枝: 原始 {len(candidates)} 个候选, 剩余 {len(pruned_candidates)} 个。最低 PPL: {min(scores):.2f} (如果有候选)")
        # 如果所有候选都被剪枝掉了，可以选择是否返回原始列表或空列表
        # if not pruned_candidates and candidates:
        #     logging.warning("困惑度剪枝后无候选剩余，将返回所有原始候选。")
        #     return candidates
        return pruned_candidates