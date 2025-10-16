# ahp_robustness/src/defenses.py

import torch
from abc import ABC, abstractmethod
import re

from src.components.masking import random_masking, adversarial_masking
from src.components.candidate_generation import generate_candidates
from src.components.result_aggregation import majority_voting
from src.pruning.nli_pruner import NliPruner

# --- 基础预测器 ---
class BasePredictor:
    def __init__(self, model, tokenizer, task):
        self.model = model
        self.tokenizer = tokenizer
        self.task = task
        # --- 关键改动：预先编码好答案的Token ---
        # 我们需要考虑带前导空格的情况，因为模型经常这样分词
        self.positive_id = self.tokenizer.encode("positive", add_special_tokens=False)[-1]
        self.negative_id = self.tokenizer.encode("negative", add_special_tokens=False)[-1]
        self.positive_id_with_space = self.tokenizer.encode(" positive", add_special_tokens=False)[-1]
        self.negative_id_with_space = self.tokenizer.encode(" negative", add_special_tokens=False)[-1]

    def _build_prompt(self, sentence):
        """(终极修正) 使用最简单、最直接的Zero-shot Prompt。"""
        # 我们不再提供任何示例，直接要求模型做判断。
        return f'Review: "{sentence}"\nSentiment:'

    def _parse_output_by_logits(self, logits):
        """(终极修正) 不再解析文本，直接比较'positive'和'negative'的概率。"""
        # 提取'positive'和'negative'这两个词（以及带空格版本）的概率
        prob_positive = logits[0, self.positive_id]
        prob_negative = logits[0, self.negative_id]
        prob_positive_space = logits[0, self.positive_id_with_space]
        prob_negative_space = logits[0, self.negative_id_with_space]

        # 取两种形式中概率更高的那个作为最终概率
        final_prob_positive = torch.max(prob_positive, prob_positive_space)
        final_prob_negative = torch.max(prob_negative, prob_negative_space)

        if final_prob_positive > final_prob_negative:
            return "positive"
        else:
            return "negative"

    def predict(self, sentence):
        prompt = self._build_prompt(sentence)
        inputs = self.tokenizer(prompt, return_tensors="pt", return_attention_mask=False).to(self.model.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            # a. 获取即将生成的下一个词的概率分布
            next_token_logits = outputs.logits[:, -1, :]
        
        # b. 直接根据概率进行解析，不再生成文本
        return self._parse_output_by_logits(next_token_logits)

# --- 防御策略 (代码无改动) ---
class BaseDefense(ABC):
    def __init__(self, predictor):
        self.predictor = predictor
        self.model = predictor.model
        self.tokenizer = predictor.tokenizer
    @abstractmethod
    def __call__(self, sentence):
        pass
class NoDefense(BaseDefense):
    def __call__(self, sentence):
        return self.predictor.predict(sentence)
class AhpDefense(BaseDefense):
    def __init__(self, predictor, k_val=3, m_val=10):
        super().__init__(predictor)
        self.k_val = k_val
        self.m_val = m_val
        self.pruner = NliPruner(self.k_val)
    def __call__(self, sentence):
        masked_text = adversarial_masking(sentence, self.model, self.tokenizer)
        candidates = generate_candidates(masked_text, self.model, self.tokenizer, self.m_val)
        pruned_candidates = self.pruner.prune(sentence, candidates)
        final_predictions = [self.predictor.predict(c) for c in pruned_candidates]
        final_result = majority_voting(final_predictions)
        if final_result is None:
            return "negative" if self.predictor.task == 'sst2' else "World"
        return final_result
class SelfDenoiseDefense(BaseDefense):
    def __init__(self, predictor, num_samples=10):
        super().__init__(predictor)
        self.num_samples = num_samples
    def __call__(self, sentence):
        masked_texts = [random_masking(sentence, self.tokenizer) for _ in range(self.num_samples)]
        denoised_candidates = []
        for mt in masked_texts:
            candidates = generate_candidates(mt, self.model, self.tokenizer, num_candidates=1)
            denoised_candidates.extend(candidates)
        final_predictions = [self.predictor.predict(c) for c in denoised_candidates]
        final_result = majority_voting(final_predictions)
        if final_result is None:
            return "negative" if self.predictor.task == 'sst2' else "World"
        return final_result