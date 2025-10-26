# src/components/masking.py
import torch
import numpy as np
import random
import logging
from transformers import PreTrainedTokenizer, PreTrainedModel
from typing import List, Tuple, Union

class AdversarialMasker:
    """根据对抗性策略（例如词语重要性）选择并遮蔽输入文本中的词语。"""
    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, device: Union[str, torch.device] = 'cuda', **kwargs):
        """
        初始化对抗性遮蔽器。

        Args:
            model: 用于计算重要性的模型。
            tokenizer: 模型对应的分词器。
            device: 计算设备 ('cuda' or 'cpu')。
            **kwargs: 其他特定于您实现的参数 (例如重要性计算方法)。
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device(device)
        self.importance_method = kwargs.get('importance_method', 'gradient') # 示例参数
        logging.info(f"对抗性遮蔽器已初始化 (重要性计算方法: {self.importance_method})。")
        # ... 其他初始化 ...

    def _calculate_word_importance(self, text: str) -> np.ndarray:
        """
        【需要您实现】计算文本中每个词的重要性得分。
        这通常涉及：
        1. 对原始文本进行预测，获取原始 logits 或 loss。
        2. 逐个遮蔽词语或计算梯度，观察 logits 或 loss 的变化。
        3. 根据变化量计算得分。

        Args:
            text (str): 输入文本。

        Returns:
            np.ndarray: 每个词对应的重要性得分数组。
        """
        logging.warning("AdversarialMasker._calculate_word_importance 尚未完全实现！")
        # --- 占位符：返回随机得分 ---
        words = text.split()
        return np.random.rand(len(words))
        # --- 结束占位符 ---

    def mask_input(self, text: str, mask_rate: float) -> Tuple[str, List[int]]:
        """
        根据计算出的重要性得分选择并遮蔽输入文本中的词语。

        Args:
            text (str): 原始输入文本。
            mask_rate (float): 需要遮蔽的词语比例。

        Returns:
            Tuple[str, List[int]]: 返回遮蔽后的文本和被遮蔽词语的原始索引列表。
        """
        words = text.split()
        n_words = len(words)
        if n_words == 0:
            return "", []

        # 计算需要遮蔽的数量
        n_mask = max(1, int(round(n_words * mask_rate)))
        n_mask = min(n_mask, n_words)

        # 1. 计算重要性得分 【调用您实现的函数】
        importance_scores = self._calculate_word_importance(text)

        # 确保得分数组长度与词数一致
        if len(importance_scores) != n_words:
            logging.error(f"重要性得分数量 ({len(importance_scores)}) 与词数 ({n_words}) 不匹配！将使用随机遮蔽。")
            # Fallback 到随机选择
            mask_indices = sorted(random.sample(range(n_words), n_mask))
        else:
            # 2. 根据得分选择要遮蔽的词语索引 (选择得分最高的 n_mask 个)
            # 使用 argsort 获取按得分降序排列的索引，然后取前 n_mask 个
            mask_indices = np.argsort(importance_scores)[::-1][:n_mask].tolist()
            mask_indices.sort() # 按原始顺序排序

        # 3. 执行遮蔽
        masked_words = list(words)
        # TODO: 从配置或初始化参数中获取 mask_token
        mask_token = "<MASK>" # 应该使用 self.args.mask_token 或类似方式获取
        for idx in mask_indices:
            masked_words[idx] = mask_token

        logging.debug(f"对抗性遮蔽索引: {mask_indices}")
        return " ".join(masked_words), mask_indices


class RandomMasker:
    """应用随机遮蔽到输入文本。"""
    def __init__(self, tokenizer: PreTrainedTokenizer, mask_token: str = "<MASK>", mask_rate: float = 0.15):
        """
        初始化随机遮蔽器。

        Args:
            tokenizer (PreTrainedTokenizer): 分词器实例 (虽然这里简单按空格分词，但保留以备将来使用)。
            mask_token (str): 用于替换被选中词语的遮蔽标记。
            mask_rate (float): 需要随机遮蔽的词语比例。
        """
        self.tokenizer = tokenizer
        self.mask_token = mask_token
        self.mask_rate = mask_rate
        logging.info(f"随机遮蔽器已初始化，遮蔽率: {mask_rate}, 遮蔽标记: '{mask_token}'")

    def mask_input(self, text: str) -> Tuple[str, List[int]]:
        """
        对单个输入文本进行随机遮蔽。

        Args:
            text (str): 原始输入文本。

        Returns:
            Tuple[str, List[int]]: 返回遮蔽后的文本和被遮蔽词语的索引列表。
        """
        words = text.split() # 使用简单的空格分词
        n_words = len(words)
        if n_words == 0:
            return "", [] # 处理空文本

        # 计算需要遮蔽的词数，至少为1，且不超过总词数
        n_mask = max(1, int(round(n_words * self.mask_rate)))
        n_mask = min(n_mask, n_words) # 确保遮蔽数不超限

        # 随机选择要遮蔽的词的索引
        mask_indices = sorted(random.sample(range(n_words), n_mask))

        masked_words = list(words) # 创建词列表副本
        # 将选定索引位置的词替换为遮蔽标记
        for idx in mask_indices:
            masked_words[idx] = self.mask_token

        return " ".join(masked_words), mask_indices

    def mask_input_multiple(self, text: str, num_masks: int) -> List[str]:
        """
        为单个输入文本生成多个随机遮蔽的版本 (用于 SelfDenoise 集成)。

        Args:
            text (str): 原始输入文本。
            num_masks (int): 需要生成的遮蔽版本数量。

        Returns:
            List[str]: 包含多个遮蔽后文本的列表。
        """
        masked_texts = []
        words = text.split()
        n_words = len(words)
        if n_words == 0:
            return [""] * num_masks # 处理空文本

        # 计算需要遮蔽的词数
        n_mask = max(1, int(round(n_words * self.mask_rate)))
        n_mask = min(n_mask, n_words)

        # 防止无法采样的情况 (虽然前面的 min 操作已处理，再检查一次)
        if n_words < n_mask:
             logging.warning(f"无法从 {n_words} 个词中选择 {n_mask} 个索引。将调整遮蔽数量。")
             n_mask = n_words

        # 生成指定数量的遮蔽版本
        for _ in range(num_masks):
            # 每次都重新随机选择索引，这是随机平滑的关键
            mask_indices = random.sample(range(n_words), n_mask)
            masked_words = list(words)
            for idx in mask_indices:
                masked_words[idx] = self.mask_token
            masked_texts.append(" ".join(masked_words))

        return masked_texts