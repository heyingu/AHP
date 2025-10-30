# src/components/masking.py
import torch
import numpy as np
import random
import logging
from transformers import PreTrainedTokenizer, PreTrainedModel
from typing import List, Tuple, Union, TYPE_CHECKING
from tqdm.auto import tqdm # 导入 tqdm 以显示进度条

# 使用 TYPE_CHECKING 块进行类型提示，以避免在运行时发生循环导入
if TYPE_CHECKING:
    from ..models.model_loader import AlpacaModel

# --- RandomMasker 类 (保持不变) ---
# (请确保您文件中的 RandomMasker 类代码在这里)
class RandomMasker:
    """应用随机遮蔽到输入文本。"""
    def __init__(self, tokenizer: PreTrainedTokenizer, mask_token: str = "<MASK>", mask_rate: float = 0.15):
        self.tokenizer = tokenizer
        self.mask_token = mask_token
        self.mask_rate = mask_rate
        logging.info(f"随机遮蔽器已初始化，遮蔽率: {mask_rate}, 遮蔽标记: '{mask_token}'")

    def mask_input(self, text: str) -> Tuple[str, List[int]]:
        words = text.split() 
        n_words = len(words)
        if n_words == 0:
            return "", [] 
        n_mask = max(1, int(round(n_words * self.mask_rate)))
        n_mask = min(n_mask, n_words) 
        mask_indices = sorted(random.sample(range(n_words), n_mask))
        masked_words = list(words) 
        for idx in mask_indices:
            masked_words[idx] = self.mask_token
        return " ".join(masked_words), mask_indices

    def mask_input_multiple(self, text: str, num_masks: int) -> List[str]:
        masked_texts = []
        words = text.split()
        n_words = len(words)
        if n_words == 0:
            return [""] * num_masks
        n_mask = max(1, int(round(n_words * self.mask_rate)))
        n_mask = min(n_mask, n_words)
        if n_words < n_mask:
             logging.warning(f"无法从 {n_words} 个词中选择 {n_mask} 个索引。将调整遮蔽数量。")
             n_mask = n_words
        for _ in range(num_masks):
            mask_indices = random.sample(range(n_words), n_mask)
            masked_words = list(words)
            for idx in mask_indices:
                masked_words[idx] = self.mask_token
            masked_texts.append(" ".join(masked_words))
        return masked_texts
# --- RandomMasker 类结束 ---


# --- 修改 AdversarialMasker 类 ---
class AdversarialMasker:
    """根据对抗性策略（例如词语重要性）选择并遮蔽输入文本中的词语。"""

    # 修改 __init__ 方法签名以接收 model_wrapper (AlpacaModel 实例)
    def __init__(self, model_wrapper: 'AlpacaModel', **kwargs):
        """
        初始化对抗性遮蔽器。
        Args:
            model_wrapper (AlpacaModel): AlpacaModel 实例，提供 _format_prompt 和 _get_logit_probs_batch 方法。
            **kwargs: 其他特定于您实现的参数。
        """
        # 直接使用传入的 model_wrapper
        self.model_wrapper = model_wrapper
        self.device = model_wrapper.device
        # 从 model_wrapper (它持有 args) 中获取 mask_token
        self.mask_token = model_wrapper.args.mask_token 
        logging.info("对抗性遮蔽器已初始化 (使用模型包装器)。")

    @torch.no_grad() # 确保不计算梯度
    def _calculate_word_importance(self, text: str) -> np.ndarray:
        """
        [已完善] 计算文本中每个词对分类结果的重要性得分。
        重要性定义为：当该词被遮蔽后，原始预测类别的概率下降了多少。
        """
        logging.debug(f"正在计算 '{text[:50]}...' 的词语重要性...")
        words = text.split() # 按空格简单分词
        n_words = len(words)
        if n_words == 0:
            return np.array([])
        
        # 1. 获取原始文本的预测概率和类别
        # 使用 model_wrapper 的辅助方法
        original_prompt = self.model_wrapper._format_prompt(self.model_wrapper.classification_instruction, text)
        original_probs_tensor = self.model_wrapper._get_logit_probs_batch([original_prompt])
        
        if original_probs_tensor.shape[0] == 0:
            logging.warning("无法获取原始文本的概率。返回随机得分。")
            return np.random.rand(n_words)
            
        original_probs = original_probs_tensor[0].numpy()
        original_pred_class = np.argmax(original_probs) # 原始预测的类别索引
        original_pred_prob = original_probs[original_pred_class] # 原始预测类别的概率
        logging.debug(f"原始预测类别: {original_pred_class}, 概率: {original_pred_prob:.4f}")

        # 2. 为每个词创建遮蔽后的文本版本
        masked_texts = []
        for i in range(n_words):
            masked_words = list(words)
            masked_words[i] = self.mask_token # 使用从 wrapper 获取的 mask_token
            masked_texts.append(" ".join(masked_words))
        
        # 3. 批量获取所有遮蔽版本的预测概率
        masked_prompts = [self.model_wrapper._format_prompt(self.model_wrapper.classification_instruction, mt) for mt in masked_texts]
        
        all_masked_probs = []
        # 使用 model_wrapper 中配置的批次大小
        batch_size = self.model_wrapper.args.model_batch_size
        
        # 使用 tqdm 显示内部进度
        for i in tqdm(range(0, len(masked_prompts), batch_size), desc="计算重要性", leave=False, ncols=100):
            batch = masked_prompts[i:i+batch_size]
            probs_tensor = self.model_wrapper._get_logit_probs_batch(batch)
            all_masked_probs.append(probs_tensor)
            
        if not all_masked_probs:
            logging.warning("无法获取遮蔽文本的概率。返回随机得分。")
            return np.random.rand(n_words)
            
        all_masked_probs_np = torch.cat(all_masked_probs, dim=0).numpy() # 形状 [n_words, num_labels]

        # 4. 计算重要性得分
        # 获取每个遮蔽版本对 *原始预测类别* 的预测概率
        probs_for_original_class = all_masked_probs_np[:, original_pred_class]
        
        # 重要性 = 原始概率 - 遮蔽后概率 (下降得越多，说明越重要)
        importance_scores = original_pred_prob - probs_for_original_class
        
        logging.debug(f"重要性得分: {importance_scores}")
        return importance_scores

    def mask_input(self, text: str, mask_rate: float) -> Tuple[str, List[int]]:
        """
        根据计算出的重要性得分选择并遮蔽输入文本中的词语。
        (此方法逻辑保持不变，现在它依赖 _calculate_word_importance 的真实实现)
        """
        words = text.split()
        n_words = len(words)
        if n_words == 0:
            return "", []

        # 计算需要遮蔽的数量
        n_mask = max(1, int(round(n_words * mask_rate)))
        n_mask = min(n_mask, n_words)

        # 1. 计算重要性得分 [调用已完善的方法]
        importance_scores = self._calculate_word_importance(text)

        if len(importance_scores) != n_words:
            logging.error(f"重要性得分数量 ({len(importance_scores)}) 与词数 ({n_words}) 不匹配！将使用随机遮蔽。")
            mask_indices = sorted(random.sample(range(n_words), n_mask))
        else:
            # 2. 根据得分选择要遮蔽的词语索引 (选择得分最高的 n_mask 个)
            # 使用 argsort 获取按得分降序排列的索引，然后取前 n_mask 个
            mask_indices = np.argsort(importance_scores)[::-1][:n_mask].tolist()
            mask_indices.sort() # 按原始顺序排序

        # 3. 执行遮蔽
        masked_words = list(words)
        for idx in mask_indices:
            masked_words[idx] = self.mask_token # 使用 self.mask_token

        logging.debug(f"对抗性遮蔽索引: {mask_indices}")
        return " ".join(masked_words), mask_indices