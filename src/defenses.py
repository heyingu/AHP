# ahp_robustness/src/defenses.py

import torch

def clean_prediction_text(text, task):
    """
    根据不同的任务，清洗LLM的输出。
    (这个函数与pipeline.py中的版本保持同步)
    """
    if not isinstance(text, str):
        return ""
    
    cleaned_text = text.lower().strip()
    
    if task == 'sst2':
        if "positive" in cleaned_text:
            return "positive"
        if "negative" in cleaned_text:
            return "negative"
            
    elif task == 'ag_news':
        if "world" in cleaned_text:
            return "World"
        if "sports" in cleaned_text:
            return "Sports"
        if "business" in cleaned_text:
            return "Business"
        if "technology" in cleaned_text or "sci/tech" in cleaned_text:
            return "Sci/Tech"
            
    return cleaned_text

class baseline_defense:
    """
    一个简单的包装类，用于“无防御”的基线模型。
    它的接口与AhpPipeline类相同，都有一个 predict_single 方法。
    """
    def __init__(self, main_model, main_tokenizer, task='sst2'):
        self.main_model = main_model
        self.main_tokenizer = main_tokenizer
        self.task = task
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    def _get_instruction_template(self):
        """
        根据任务类型返回对应的指令模板。
        (这个函数与pipeline.py中的版本保持同步)
        """
        if self.task == 'sst2':
            return """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
Analyze the sentiment of the following movie review. Respond with a single word: 'positive' or 'negative'.

Review: "{review_text}"

### Response:
"""
        elif self.task == 'ag_news':
            return """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
Classify the following news article into one of four categories: "World", "Sports", "Business", or "Sci/Tech". Respond with a single word.

Article: "{review_text}"

### Response:
"""
        else:
            raise ValueError(f"未知的任务类型: {self.task}")

    def predict_single(self, sentence):
        """
        使用原始模型直接进行一次预测。
        """
        instruction_template = self._get_instruction_template()
        prompt = instruction_template.format(review_text=sentence)
        inputs = self.main_tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.main_model.generate(**inputs, max_new_tokens=10, pad_token_id=self.main_tokenizer.eos_token_id)

        prediction_part = self.main_tokenizer.decode(
            outputs[0][inputs.input_ids.shape[-1]:], 
            skip_special_tokens=True
        )

        cleaned_prediction = clean_prediction_text(prediction_part, self.task)
        
        return cleaned_prediction

# import torch
# from abc import ABC, abstractmethod
# import re

# from src.components.masking import random_masking, adversarial_masking
# from src.components.candidate_generation import generate_candidates
# from src.components.result_aggregation import majority_voting
# from src.pruning.nli_pruner import NliPruner

# # --- 基础预测器 ---
# class BasePredictor:
#     def __init__(self, model, tokenizer, task):
#         self.model = model
#         self.tokenizer = tokenizer
#         self.task = task
#         # --- 关键改动：预先编码好答案的Token ---
#         # 我们需要考虑带前导空格的情况，因为模型经常这样分词
#         self.positive_id = self.tokenizer.encode("positive", add_special_tokens=False)[-1]
#         self.negative_id = self.tokenizer.encode("negative", add_special_tokens=False)[-1]
#         self.positive_id_with_space = self.tokenizer.encode(" positive", add_special_tokens=False)[-1]
#         self.negative_id_with_space = self.tokenizer.encode(" negative", add_special_tokens=False)[-1]

#     def _build_prompt(self, sentence):
#         """(终极修正) 使用最简单、最直接的Zero-shot Prompt。"""
#         # 我们不再提供任何示例，直接要求模型做判断。
#         return f'Review: "{sentence}"\nSentiment:'

#     def _parse_output_by_logits(self, logits):
#         """(终极修正) 不再解析文本，直接比较'positive'和'negative'的概率。"""
#         # 提取'positive'和'negative'这两个词（以及带空格版本）的概率
#         prob_positive = logits[0, self.positive_id]
#         prob_negative = logits[0, self.negative_id]
#         prob_positive_space = logits[0, self.positive_id_with_space]
#         prob_negative_space = logits[0, self.negative_id_with_space]

#         # 取两种形式中概率更高的那个作为最终概率
#         final_prob_positive = torch.max(prob_positive, prob_positive_space)
#         final_prob_negative = torch.max(prob_negative, prob_negative_space)

#         if final_prob_positive > final_prob_negative:
#             return "positive"
#         else:
#             return "negative"

#     def predict(self, sentence):
#         prompt = self._build_prompt(sentence)
#         inputs = self.tokenizer(prompt, return_tensors="pt", return_attention_mask=False).to(self.model.device)
#         with torch.no_grad():
#             outputs = self.model(**inputs)
#             # a. 获取即将生成的下一个词的概率分布
#             next_token_logits = outputs.logits[:, -1, :]
        
#         # b. 直接根据概率进行解析，不再生成文本
#         return self._parse_output_by_logits(next_token_logits)

# # --- 防御策略 (代码无改动) ---
# class BaseDefense(ABC):
#     def __init__(self, predictor):
#         self.predictor = predictor
#         self.model = predictor.model
#         self.tokenizer = predictor.tokenizer
#     @abstractmethod
#     def __call__(self, sentence):
#         pass
# class NoDefense(BaseDefense):
#     def __call__(self, sentence):
#         return self.predictor.predict(sentence)
# class AhpDefense(BaseDefense):
#     def __init__(self, predictor, k_val=3, m_val=10):
#         super().__init__(predictor)
#         self.k_val = k_val
#         self.m_val = m_val
#         self.pruner = NliPruner(self.k_val)
#     def __call__(self, sentence):
#         masked_text = adversarial_masking(sentence, self.model, self.tokenizer)
#         candidates = generate_candidates(masked_text, self.model, self.tokenizer, self.m_val)
#         pruned_candidates = self.pruner.prune(sentence, candidates)
#         final_predictions = [self.predictor.predict(c) for c in pruned_candidates]
#         final_result = majority_voting(final_predictions)
#         if final_result is None:
#             return "negative" if self.predictor.task == 'sst2' else "World"
#         return final_result
# class SelfDenoiseDefense(BaseDefense):
#     def __init__(self, predictor, num_samples=10):
#         super().__init__(predictor)
#         self.num_samples = num_samples
#     def __call__(self, sentence):
#         masked_texts = [random_masking(sentence, self.tokenizer) for _ in range(self.num_samples)]
#         denoised_candidates = []
#         for mt in masked_texts:
#             candidates = generate_candidates(mt, self.model, self.tokenizer, num_candidates=1)
#             denoised_candidates.extend(candidates)
#         final_predictions = [self.predictor.predict(c) for c in denoised_candidates]
#         final_result = majority_voting(final_predictions)
#         if final_result is None:
#             return "negative" if self.predictor.task == 'sst2' else "World"
#         return final_result