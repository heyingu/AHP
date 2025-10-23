# ahp_robustness/src/pruning/nli_pruner.py

from src.pruning.base_pruner import BasePruner
from src.models.model_loader import load_nli_model
import torch
from tqdm.auto import tqdm

class NliPruner(BasePruner):
    def __init__(self, k_val, device):
        # 1. 首先，调用父类的构造函数
        super().__init__(k_val)
        
        # 2. 然后，设置自己独有的属性
        self.device = device
        
        # 3. 最后，在所有必需的属性都设置好之后，再加载模型
        self.nli_model, self.nli_tokenizer = self.load_model()

    def load_model(self):
        model, tokenizer = load_nli_model()
        # self.device 在此时已经存在了
        model.to(self.device)
        model.eval()
        return model, tokenizer

    def prune(self, original_sentence, candidates):
        if not candidates:
            return []

        scores = []
        with torch.no_grad():
            for candidate in tqdm(candidates, desc="Calculating NLI Scores", leave=False):
                inputs = self.nli_tokenizer(original_sentence, candidate, return_tensors='pt', truncation=True, padding=True).to(self.device)
                outputs = self.nli_model(**inputs)
                
                probs = torch.softmax(outputs.logits, dim=-1)
                
                entailment_prob = probs[0, 2].item() 
                scores.append(entailment_prob)

        scored_candidates = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        
        pruned_candidates = [candidate for candidate, score in scored_candidates[:self.k]]
        
        return pruned_candidates