# # ahp_robustness/src/components/candidate_generation.py
import torch

class CandidateGenerator:
    def __init__(self, model, tokenizer, num_candidates=10, max_new_tokens=64):
        """
        初始化候选生成器。
        
        Args:
            model: 主 LLM 模型。
            tokenizer: 对应的分词器。
            num_candidates (int): 要生成的候选数量 (M值)。
            max_new_tokens (int): 每个候选生成的最大 token 数。
        """
        self.model = model
        self.tokenizer = tokenizer
        self.num_candidates = num_candidates
        self.max_new_tokens = max_new_tokens

    def _build_denoising_prompt(self, masked_text):
        """
        构建一个用于去噪（填词）的指令Prompt。
        """
        # 使用Vicuna官方推荐的对话模板
        # 保持这个模板，因为它与您使用的circulus/alpaca-7b这类模型兼容性好
#         prompt = f"""A chat between a user and an artificial intelligence assistant.
# USER: I have a sentence with masked words. Your task is to fill in these masks to make the sentence coherent and natural. Only provide the completed sentence, without any explanations.

# Sentence: "{masked_text}"
# ASSISTANT:"""
        prompt = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
I have a sentence where some parts might be unclear or masked. Your task is to rewrite the sentence to make it coherent and natural. Provide only the completed sentence, without any other text or explanations.

Sentence: "{masked_text}"

### Response:
"""
        return prompt

    def generate(self, masked_text):
        """
        从被遮蔽的文本生成多个候选句子。
        """
        prompt = self._build_denoising_prompt(masked_text)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                num_return_sequences=self.num_candidates,
                num_beams=self.num_candidates,
                do_sample=True,
                top_p=0.9,
                temperature=0.7,
                early_stopping=True,
                pad_token_id=self.tokenizer.eos_token_id
            )

        full_outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        candidates = []
        for text in full_outputs:
            # assistant_response = text.split("ASSISTANT:")[-1].strip()
            assistant_response = text.split("### Response:")[-1].strip()
            # 增加一个检查，确保不返回空字符串
            if assistant_response:
                candidates.append(assistant_response)
            
        return candidates
# import torch

# def _build_denoising_prompt(masked_text):
#     """
#     (新函数) 构建一个用于去噪（填词）的指令Prompt。
#     """
#     # 使用Vicuna官方推荐的对话模板
#     prompt = f"""A chat between a user and an artificial intelligence assistant.
# USER: I have a sentence with masked words, marked as '[MASK]'. Your task is to fill in these masks to make the sentence coherent and natural. Only provide the completed sentence, without any explanations.

# Sentence: "{masked_text}"
# ASSISTANT:"""
#     return prompt

# def generate_candidates(masked_text, model, tokenizer, num_candidates=10, max_new_tokens=64):
#     """
#     (修正后) 从被遮蔽的文本生成多个候选句子。
#     """
#     prompt = _build_denoising_prompt(masked_text)
#     inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
#     with torch.no_grad():
#         outputs = model.generate(
#             **inputs,
#             max_new_tokens=max_new_tokens,
#             num_return_sequences=num_candidates,
#             num_beams=num_candidates,
#             do_sample=True,
#             top_p=0.9,
#             temperature=0.7,
#             early_stopping=True,
#             pad_token_id=tokenizer.eos_token_id # 防止过早停止
#         )

#     # 解码并清理输出，只保留ASSISTANT:之后的部分
#     full_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    
#     candidates = []
#     for text in full_outputs:
#         # 找到ASSISTANT:之后的内容
#         assistant_response = text.split("ASSISTANT:")[-1].strip()
#         candidates.append(assistant_response)
        
#     return candidates