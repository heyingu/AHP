# src/pruning/base_pruner.py
import logging
from abc import ABC, abstractmethod
from typing import List, Any

class BasePruner(ABC):
    """
    所有剪枝器方法的抽象基类。
    """
    def __init__(self, threshold: Any):
        """
        初始化基类。

        Args:
            threshold (Any): 用于剪枝的阈值或参数，具体含义由子类定义。
        """
        self.threshold = threshold
        logging.info(f"初始化 {self.__class__.__name__}，阈值/参数: {self.threshold}")

    @abstractmethod
    def prune(self, original_text: str, candidates: List[str], **kwargs) -> List[str]:
        """
        根据特定规则剪枝候选句子列表。

        Args:
            original_text (str): 原始输入文本，可能用于比较或计算。
            candidates (List[str]): 待剪枝的候选句子列表。
            **kwargs: 其他可能需要的上下文信息 (例如 masked_text)。

        Returns:
            List[str]: 通过剪枝规则筛选后保留的候选句子列表。
        """
        raise NotImplementedError("子类必须实现 prune 方法")

    def __call__(self, original_text: str, candidates: List[str], **kwargs) -> List[str]:
        """使剪枝器对象可以像函数一样被调用。"""
        return self.prune(original_text, candidates, **kwargs)