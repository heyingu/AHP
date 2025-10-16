# ahp_robustness/src/pruning/perplexity_pruner.py

from .base_pruner import BasePruner
from src.models.model_loader import load_main_llm # 这里我们复用主模型加载器来加载一个中小型模型
import torch
from tqdm.auto import tqdm

class PerplexityPruner(BasePruner):
    """
    使用困惑度 (PPL) 进行剪枝。
    选择语言模型认为最流畅（PPL最低）的K个候选。
    """
    def load_model(self):
        # 为保证效率，使用一个中小型模型如GPT-2来计算PPL
        # 注意：这里也可以选择加载一个独立的、更小的模型
        model, tokenizer = load_main_llm(model_name="gpt2", use_4bit=False)
        return model, tokenizer

    def prune(self, original_text, candidates):
        if not candidates:
            return []

        model, tokenizer = self.model
        ppls = []
        
        # print("正在计算候选句子的困惑度...")
        for sentence in tqdm(candidates, desc="Calculating PPL"):
            try:
                inputs = tokenizer(sentence, return_tensors="pt").to(model.device)
                with torch.no_grad():
                    outputs = model(**inputs, labels=inputs["input_ids"])
                
                # 计算困惑度: PPL = exp(cross_entropy_loss)
                ppl = torch.exp(outputs.loss).item()
                ppls.append(ppl)
            except Exception:
                # 如果句子过长或有问题，给一个很高的PPL值
                ppls.append(float('inf'))

        # 将候选和其PPL值配对
        candidate_ppls = list(zip(candidates, ppls))
        
        # 按PPL值升序排序
        sorted_candidates = sorted(candidate_ppls, key=lambda x: x[1])
        
        # 选择PPL最低的K个候选
        pruned_candidates = [candidate for candidate, ppl in sorted_candidates[:self.k]]
        
        return pruned_candidates