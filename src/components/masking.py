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

    # __init__ 方法保持不变
    def __init__(self, model_wrapper: 'AlpacaModel', **kwargs):
        """
        初始化对抗性遮蔽器。
        Args:
            model_wrapper (AlpacaModel): AlpacaModel 实例，提供 _format_prompt 和 _get_logit_probs_batch 方法。
            **kwargs: 其他特定于您实现的参数。
        """
        self.model_wrapper = model_wrapper
        self.device = model_wrapper.device
        self.mask_token = model_wrapper.args.mask_token 
        logging.info("对抗性遮蔽器已初始化 (使用模型包装器)。")

    # --- 关键修改：重写 _calculate_word_importance ---
    # *** 注意：移除了 @torch.no_grad() 装饰器 ***
    def _calculate_word_importance(self, text: str) -> np.ndarray:
        """
        [新实现] 使用梯度显著图 (Gradient Saliency) 计算词语重要性。
        重要性定义为：模型对原始预测类别的置信度相对于该词输入嵌入的梯度范数。
        """
        logging.debug(f"正在计算 '{text[:50]}...' 的梯度显著图...")
        words = text.split() # 按空格简单分词 (与 mask_input 保持一致)
        n_words = len(words)
        if n_words == 0:
            return np.array([])

        # 从 model_wrapper 获取所需组件
        model = self.model_wrapper.model
        tokenizer = self.model_wrapper.tokenizer
        device = self.model_wrapper.device

        # --- 1. 定位 text 在完整 prompt 中的字符边界 ---
        original_prompt = self.model_wrapper._format_prompt(self.model_wrapper.classification_instruction, text)
        text_start_char = original_prompt.find(text)

        if text_start_char == -1:
            logging.error(f"无法在 prompt 中定位文本: '{text[:50]}...'。将使用随机遮蔽。")
            return np.random.rand(n_words)

        # --- 2. 定位每个 word 在完整 prompt 中的字符边界 ---
        word_boundaries = [] # 存储 (start_char, end_char)
        current_char_idx = text_start_char
        for word in words:
            word_start = original_prompt.find(word, current_char_idx)
            
            if word_start == -1:
                temp_idx = current_char_idx
                while temp_idx < len(original_prompt) and original_prompt[temp_idx].isspace():
                    temp_idx += 1
                if original_prompt.startswith(word, temp_idx):
                    word_start = temp_idx
                else:
                    logging.warning(f"无法在 prompt 中对齐 word: '{word}'。该词得分将为 0。")
                    word_boundaries.append((-1, -1)) # 标记为无效
                    current_char_idx += len(word) + 1 
                    continue

            word_end = word_start + len(word)
            word_boundaries.append((word_start, word_end))
            current_char_idx = word_end 

        # --- 3. 启用梯度，执行一次前向和反向传播 ---
        with torch.enable_grad(): # 确保梯度被计算
            model.eval() # 保持在评估模式 (关闭 dropout)

            # --- 3a. 获取输入嵌入 (Input Embeddings) ---
            inputs = tokenizer(
                original_prompt, 
                return_tensors="pt", 
                padding=True, 
                truncation=True,
                max_length=self.model_wrapper.args.max_seq_length,
                return_offsets_mapping=True # 获取 token 字符偏移量
            ).to(device)
            
            input_ids = inputs["input_ids"]
            token_offsets = inputs["offset_mapping"][0].cpu().numpy() # [seq_len, 2]

            embedding_layer = model.get_input_embeddings()
            input_embeddings = embedding_layer(input_ids) # [1, seq_len, embed_dim]
            
            # ---!!! ---
            # ---!!! ---
            # ** 关键修复：使用 .retain_grad() 代替 .requires_grad = True **
            # 
            # 告诉 PyTorch 在反向传播时“保留”这个非叶子张量的梯度
            input_embeddings.retain_grad()
            # ---!!! ---
            # ---!!! ---
            
            # --- 3b. 前向传播 (使用 embeddings) ---
            model.zero_grad() # 清除旧梯度
            outputs = model(
                inputs_embeds=input_embeddings, 
                attention_mask=inputs["attention_mask"]
            )

            # --- 3c. 获取目标 Logit (用于反向传播) ---
            last_token_logits = outputs.logits[:, -1, :] # [1, vocab_size]
            label_logits = last_token_logits[:, self.model_wrapper.label_tokens] # [1, num_labels]
            
            original_pred_class_idx = torch.argmax(label_logits, dim=-1)
            target_score = label_logits[0, original_pred_class_idx]

            # --- 3d. 反向传播 ---
            target_score.backward()

        # --- 4. 提取梯度并计算 Token 显著性 ---
        if input_embeddings.grad is None:
            logging.warning("未能计算梯度。返回随机得分。")
            return np.random.rand(n_words)
            
        token_saliency = torch.norm(input_embeddings.grad, dim=-1).squeeze(0)
        token_saliency_np = token_saliency.cpu().detach().numpy()

        # --- 5. 将 Token 显著性聚合 (Aggregate) 回 Word ---
        word_scores = np.zeros(n_words)
        
        for token_idx, (tok_start, tok_end) in enumerate(token_offsets):
            if tok_start == 0 and tok_end == 0:
                continue 
            
            token_center = (tok_start + tok_end) / 2

            for word_idx, (word_start, word_end) in enumerate(word_boundaries):
                if word_start == -1: 
                    continue
                
                if token_center >= word_start and token_center < word_end:
                    word_scores[word_idx] = max(word_scores[word_idx], token_saliency_np[token_idx])
                    break 
        
        logging.debug(f"梯度重要性得分: {word_scores}")
        return word_scores


    def mask_input(self, text: str, mask_rate: float) -> Tuple[str, List[int]]:
        """
        [此方法保持不变]
        根据计算出的重要性得分选择并遮蔽输入文本中的词语。
        """
        words = text.split()
        n_words = len(words)
        if n_words == 0:
            return "", []

        n_mask = max(1, int(round(n_words * mask_rate)))
        n_mask = min(n_mask, n_words)

        # 1. 计算重要性得分 [调用新的梯度方法]
        importance_scores = self._calculate_word_importance(text)

        if len(importance_scores) != n_words:
            logging.error(f"重要性得分数量 ({len(importance_scores)}) 与词数 ({n_words}) 不匹配！将使用随机遮蔽。")
            mask_indices = sorted(random.sample(range(n_words), n_mask))
        else:
            # 2. 根据得分选择要遮蔽的词语索引 (选择得分最高的 n_mask 个)
            mask_indices = np.argsort(importance_scores)[::-1][:n_mask].tolist()
            mask_indices.sort() # 按原始顺序排序

        # 3. 执行遮蔽
        masked_words = list(words)
        for idx in mask_indices:
            masked_words[idx] = self.mask_token # 使用 self.mask_token

        logging.debug(f"对抗性遮蔽索引: {mask_indices}")
        return " ".join(masked_words), mask_indices