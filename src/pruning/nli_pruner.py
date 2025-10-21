# ahp_robustness/src/pruning/nli_pruner.py

from src.pruning.base_pruner import BasePruner # <--- 修改点
from src.models.model_loader import load_nli_model
import torch
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

class NliPruner(BasePruner):
    """
    使用自然语言推理 (NLI) 进行剪枝。
    选择与原始文本构成"蕴含"关系且置信度最高的K个候选。
    """
    def __init__(self, k_val, device):
        self.device = device
        super().__init__(k_val)
        
    def load_model(self):
        # 加载预训练的NLI模型
        model, tokenizer = load_nli_model(model_name="roberta-large-mnli")
        model.to("cuda" if torch.cuda.is_available() else "cpu")
        model.eval()
        return model, tokenizer

    def prune(self, original_text, candidates):
        if not candidates:
            return []
        
        model, tokenizer = self.model
        scores = []

        # print("正在计算候选句子的NLI蕴含分数...")
        for hypothesis in tqdm(candidates, desc="Calculating NLI Scores"):
            try:
                # NLI模型输入格式：[CLS] premise [SEP] hypothesis [SEP]
                tokenized_input = tokenizer(original_text, hypothesis, return_tensors='pt', truncation=True, padding=True).to(model.device)
                
                with torch.no_grad():
                    outputs = model(**tokenized_input)
                
                # 获取预测的logits，并转换为概率
                probs = torch.softmax(outputs.logits, dim=-1)
                
                # NLI模型的标签通常是: 0 -> contradiction, 1 -> neutral, 2 -> entailment
                # 我们需要蕴含（entailment）的概率
                entailment_prob = probs[0][2].item()
                scores.append(entailment_prob)
            except Exception:
                scores.append(0.0) # 如果出错，给一个最低分

        # 将候选和其分数配对
        candidate_scores = list(zip(candidates, scores))
        
        # 按蕴含分数降序排序
        sorted_candidates = sorted(candidate_scores, key=lambda x: x[1], reverse=True)
        
        # 选择分数最高的K个候选
        pruned_candidates = [candidate for candidate, score in sorted_candidates[:self.k]]
        
        return pruned_candidates