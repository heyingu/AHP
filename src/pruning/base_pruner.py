# # ahp_robustness/src/pruning/base_pruner.py


from abc import ABC, abstractmethod

class BasePruner(ABC):
    """
    所有剪枝器实现的抽象基类。
    """
    # --- 关键修改点 ---
    # 将参数名从 'k' 修改为 'k_val' 以匹配 pipeline 中的调用
    def __init__(self, k_val):
        """
        初始化剪枝器。
        Args:
            k_val (int): 剪枝后要保留的最优候选数量。
        """
        self.k = k_val # 将传入的 k_val 赋值给实例变量 self.k
        self.model = self.load_model()

    @abstractmethod
    def load_model(self):
        """
        加载此剪枝方法所需的模型。
        """
        pass

    @abstractmethod
    def prune(self, original_text, candidates):
        """
        执行剪枝操作。

        Args:
            original_text (str): 原始未遮蔽的输入文本。
            candidates (list[str]): 由LLM生成的M个候选句子。

        Returns:
            list[str]: 经过筛选后的K个最优候选句子。
        """
        pass
# from abc import ABC, abstractmethod

# class BasePruner(ABC):
#     """
#     所有剪枝器实现的抽象基类。
#     """
#     def __init__(self, k):
#         """
#         初始化剪枝器。
#         Args:
#             k (int): 剪枝后要保留的最优候选数量。
#         """
#         self.k = k
#         self.model = self.load_model()

#     @abstractmethod
#     def load_model(self):
#         """
#         加载此剪枝方法所需的模型。
#         """
#         pass

#     @abstractmethod
#     def prune(self, original_text, candidates):
#         """
#         执行剪枝操作。

#         Args:
#             original_text (str): 原始未遮蔽的输入文本。
#             candidates (list[str]): 由LLM生成的M个候选句子。

#         Returns:
#             list[str]: 经过筛选后的K个最优候选句子。
#         """
#         pass