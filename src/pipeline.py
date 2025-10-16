# ahp_robustness/src/pipeline.py

from src.components.masking import adversarial_masking
from src.components.candidate_generation import generate_candidates
from src.components.result_aggregation import majority_voting
import torch
import re

# 导入所有剪枝器
from src.pruning.semantic_pruner import SemanticPruner
from src.pruning.perplexity_pruner import PerplexityPruner
from src.pruning.nli_pruner import NliPruner
from src.pruning.clustering_pruner import ClusteringPruner

class AhpPipeline:
    def __init__(self, main_model, main_tokenizer, pruner_name, k_val, m_val=10, task='sst2'):
        """
        初始化AHP框架流水线。
        task (str): 'sst2' 或 'ag_news'，决定使用哪个分类prompt。
        """
        self.model = main_model
        self.tokenizer = main_tokenizer
        self.m_val = m_val
        self.k_val = k_val
        self.task = task
        self.pruner = self._get_pruner(pruner_name, self.k_val)
        
    def _get_pruner(self, name, k):
        """根据名称初始化并返回一个剪枝器实例。"""
        if name == 'semantic':
            return SemanticPruner(k)
        elif name == 'perplexity':
            return PerplexityPruner(k)
        elif name == 'nli':
            return NliPruner(k)
        elif name == 'clustering':
            return ClusteringPruner(k)
        else:
            raise ValueError(f"未知的剪枝器名称: {name}")

    def predict_single(self, text):
        """对单个文本样本执行完整的AHP预测流程。"""
        masked_text = adversarial_masking(text, self.model, self.tokenizer)
        candidates = generate_candidates(masked_text, self.model, self.tokenizer, self.m_val)
        pruned_candidates = self.pruner.prune(text, candidates)
        final_predictions = self._get_final_predictions(pruned_candidates)
        final_result = majority_voting(final_predictions)
        
        # 如果聚合失败，提供一个基于任务的默认值
        if final_result is None:
            return "negative" if self.task == 'sst2' else "World"
            
        return final_result
        
    def _build_classification_prompt(self, sentence):
        """根据任务类型构建分类prompt。"""
        if self.task == 'sst2':
            return f"""A chat between a user and an artificial intelligence assistant.
USER: You are an expert in sentiment analysis. Classify the sentiment of the following movie review as "positive" or "negative".

Review: "An astonishing cinematic achievement."
Sentiment: positive

Review: "A complete waste of time and money."
Sentiment: negative

Review: "{sentence}"
Sentiment:
ASSISTANT:"""
        elif self.task == 'ag_news':
            return f"""A chat between a user and an artificial intelligence assistant.
USER: You are an expert in news classification. Classify the following news article into one of four categories: "World", "Sports", "Business", or "Technology".

Article: "The tech giant launched its new flagship phone with a revolutionary camera system."
Category: Technology

Article: "The stock market hit a record high after the central bank's announcement."
Category: Business

Article: "{sentence}"
Category:
ASSISTANT:"""
        else:
            raise ValueError(f"未知的任务类型: {self.task}")

    def _parse_prediction(self, generated_text):
        """根据任务类型解析模型的输出。"""
        text = generated_text.lower().strip()
        if self.task == 'sst2':
            if text.startswith("positive"): return "positive"
            if text.startswith("negative"): return "negative"
        elif self.task == 'ag_news':
            # 提高解析的鲁棒性
            if "world" in text: return "World"
            if "sports" in text: return "Sports"
            if "business" in text: return "Business"
            if "technology" in text: return "Technology"
        
        return "unknown" 

    def _get_final_predictions(self, candidates):
        """使用主LLM对K个候选进行最终预测。"""
        predictions = []
        for candidate_text in candidates:
            prompt = self._build_classification_prompt(candidate_text)
            inputs = self.tokenizer(prompt, return_tensors="pt", return_attention_mask=False).to(self.model.device)
            
            with torch.no_grad():
                # 增加生成token数量以容纳更长的类别名称
                output = self.model.generate(**inputs, max_new_tokens=10, pad_token_id=self.tokenizer.eos_token_id)

            prediction_text = self.tokenizer.decode(output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            parsed_pred = self._parse_prediction(prediction_text)
            predictions.append(parsed_pred)
            
        return predictions