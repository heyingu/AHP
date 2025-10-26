# ahp_robustness/src/defenses.py

import torch
from abc import ABC, abstractmethod
import re

# --- 核心依赖导入 ---
from src.components.masking import random_masking, adversarial_masking
from src.components.candidate_generation import CandidateGenerator 
from src.components.result_aggregation import majority_voting
from src.pruning.nli_pruner import NliPruner

# --- 文本清洗函数 ---
def clean_prediction_text(text, task):
    if not isinstance(text, str): return ""
    cleaned_text = text.lower().strip()
    if task == 'sst2':
        if "positive" in cleaned_text: return "positive"
        if "negative" in cleaned_text: return "negative"
    elif task == 'ag_news':
        if "world" in cleaned_text: return "World"
        if "sports" in cleaned_text: return "Sports"
        if "business" in cleaned_text: return "Business"
        if "technology" in cleaned_text or "sci/tech" in cleaned_text: return "Sci/Tech"
    return cleaned_text

# --- 基础预测器 (Prompt 已修改) ---
class BasePredictor:
    def __init__(self, model, tokenizer, task):
        self.model = model
        self.tokenizer = tokenizer
        self.task = task
        self.positive_id = self.tokenizer.encode("positive", add_special_tokens=False)[-1]
        self.negative_id = self.tokenizer.encode("negative", add_special_tokens=False)[-1]
        self.positive_id_with_space = self.tokenizer.encode(" positive", add_special_tokens=False)[-1]
        self.negative_id_with_space = self.tokenizer.encode(" negative", add_special_tokens=False)[-1]

    def _get_instruction_template(self):
        # ==================== 核心修改点 ====================
        # 使用与Alpaca模型更匹配的指令格式，这与论文中的方法论一致。
        if self.task == 'sst2':
            return """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
Analyze the sentiment of the following movie review. Respond with a single word: 'positive' or 'negative'.

### Input:
{review_text}

### Response:
"""
        else:
            raise ValueError(f"未知的任务类型: {self.task}")
        # =====================================================

    def _parse_output_by_logits(self, logits):
        final_prob_positive = torch.max(logits[0, self.positive_id], logits[0, self.positive_id_with_space])
        final_prob_negative = torch.max(logits[0, self.negative_id], logits[0, self.negative_id_with_space])
        return "positive" if final_prob_positive > final_prob_negative else "negative"

    def predict(self, sentence):
        prompt = self._get_instruction_template().format(review_text=sentence)
        inputs = self.tokenizer(prompt, return_tensors="pt", return_attention_mask=False).to(self.model.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            next_token_logits = outputs.logits[:, -1, :]
        return self._parse_output_by_logits(next_token_logits)

# --- 防御策略抽象基类 ---
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

# --- AHP防御 ---
class AhpDefense(BaseDefense):
    def __init__(self, predictor, k_val=3, m_val=10):
        super().__init__(predictor)
        self.k_val = k_val
        self.m_val = m_val
        
        device = self.predictor.model.device
        self.pruner = NliPruner(k_val=self.k_val, device=device)
        self.candidate_generator = CandidateGenerator(self.model, self.tokenizer, num_candidates=self.m_val)

    def __call__(self, sentence):
        masked_text = random_masking(sentence, self.tokenizer)
        # masked_text = adversarial_masking(sentence, self.model, self.tokenizer)
        candidates = self.candidate_generator.generate(masked_text)
        pruned_candidates = self.pruner.prune(sentence, candidates)
        
        if not pruned_candidates:
            return self.predictor.predict(sentence)
            
        final_predictions = [self.predictor.predict(c) for c in pruned_candidates]
        final_result = majority_voting(final_predictions)
        
        return final_result if final_result is not None else "negative"

# --- Self-Denoise 防御 ---
class SelfDenoiseDefense(BaseDefense):
    def __init__(self, predictor, num_samples=10):
        super().__init__(predictor)
        self.num_samples = num_samples
        self.denoiser = CandidateGenerator(self.model, self.tokenizer, num_candidates=1)

    def __call__(self, sentence):
        masked_texts = [random_masking(sentence, self.tokenizer) for _ in range(self.num_samples)]
        
        denoised_candidates = []
        for mt in masked_texts:
            new_candidates = self.denoiser.generate(mt)
            denoised_candidates.extend(new_candidates)
        
        if not denoised_candidates:
            return self.predictor.predict(sentence)

        final_predictions = [self.predictor.predict(c) for c in denoised_candidates]
        final_result = majority_voting(final_predictions)

        return final_result if final_result is not None else "negative"