# ahp_robustness/src/pipeline.py

import torch
from src.components.candidate_generation import CandidateGenerator
from src.components.result_aggregation import majority_voting # 确认使用 majority_voting

# 导入所有剪枝器
from src.pruning.semantic_pruner import SemanticPruner
from src.pruning.perplexity_pruner import PerplexityPruner
from src.pruning.nli_pruner import NliPruner
from src.pruning.clustering_pruner import ClusteringPruner

# 这是一个开关，您可以设置为False来关闭调试信息的打印
DEBUG_MODE = False

# --- 新增的文本清洗函数 ---
def clean_prediction_text(text, task):
    """
    清洗LLM的输出，提取核心的情感标签。
    例如： " The final answer is: positive." -> "positive"
    """
    if not isinstance(text, str):
        return "" # 如果不是字符串，返回空
    
    # 转换为小写，去除首尾空格
    cleaned_text = text.lower().strip()
    
    if task == 'sst2':
        if "positive" in cleaned_text:
            return "positive"
        if "negative" in cleaned_text:
            return "negative"
            
    elif task == 'ag_news':
        # 检查每个类别，返回标准的大写格式以匹配数据集
        if "world" in cleaned_text:
            return "World"
        if "sports" in cleaned_text:
            return "Sports"
        if "business" in cleaned_text:
            return "Business"
        if "technology" in cleaned_text or "sci/tech" in cleaned_text:
            return "Sci/Tech"
            
    return cleaned_text

class AhpPipeline:
    def __init__(self, main_model, main_tokenizer, pruner_name, k_val, m_val, task='sst2'):
        self.main_model = main_model
        self.main_tokenizer = main_tokenizer
        self.k_val = k_val
        self.m_val = m_val
        self.task = task # 保存任务类型
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.candidate_generator = CandidateGenerator(
            self.main_model, 
            self.main_tokenizer, 
            num_candidates=self.m_val
        )

        if pruner_name == 'semantic':
            self.pruner = SemanticPruner(k_val=self.k_val)
        elif pruner_name == 'perplexity':
            self.pruner = PerplexityPruner(k_val=self.k_val, device=self.device)
        elif pruner_name == 'nli':
            self.pruner = NliPruner(k_val=self.k_val, device=self.device)
        elif pruner_name == 'clustering':
            self.pruner = ClusteringPruner(k_val=self.k_val)
        else:
            raise ValueError(f"未知的剪枝器名称: {pruner_name}")


    def _get_instruction_template(self):
        """
        (新函数) 根据任务类型返回对应的指令模板。
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
    def _final_prediction(self, candidates):
        predictions = []
        instruction_template = self._get_instruction_template()
        
        for i, candidate in enumerate(candidates):
            prompt = instruction_template.format(review_text=candidate)
            inputs = self.main_tokenizer(prompt, return_tensors="pt").to(self.device)
            
            outputs = self.main_model.generate(**inputs, max_new_tokens=10, pad_token_id=self.main_tokenizer.eos_token_id)
            
            prediction_part = self.main_tokenizer.decode(
                outputs[0][inputs.input_ids.shape[-1]:], 
                skip_special_tokens=True
            )
            
            # (关键修改) 将任务类型传入清洗函数
            cleaned_prediction = clean_prediction_text(prediction_part, self.task)

            if DEBUG_MODE and i == 0:
                print("\n--- [DEBUG] Final Prediction Step ---")
                print(f"Task: {self.task}")
                print(f"Input Prompt:\n{prompt}")
                print(f"Raw Prediction Text: '{prediction_part}'")
                print(f"Cleaned Prediction: '{cleaned_prediction}'")
                print("-------------------------------------\n")

            predictions.append(cleaned_prediction)
            
        return predictions

    def predict_single(self, sentence):
        masked_sentence = sentence 
        candidates = self.candidate_generator.generate(masked_sentence)
        if not candidates:
            final_outputs = self._final_prediction([sentence])
            return majority_voting(final_outputs)

        pruned_candidates = self.pruner.prune(sentence, candidates)
        final_outputs = self._final_prediction(pruned_candidates)
        final_prediction = majority_voting(final_outputs)

        return final_prediction
# class AhpPipeline:
#     def __init__(self, main_model, main_tokenizer, pruner_name, k_val, m_val=10, task='sst2'):
#         """
#         初始化AHP框架流水线。
#         task (str): 'sst2' 或 'ag_news'，决定使用哪个分类prompt。
#         """
#         self.model = main_model
#         self.tokenizer = main_tokenizer
#         self.m_val = m_val
#         self.k_val = k_val
#         self.task = task
#         self.pruner = self._get_pruner(pruner_name, self.k_val)
        
#     def _get_pruner(self, name, k):
#         """根据名称初始化并返回一个剪枝器实例。"""
#         if name == 'semantic':
#             return SemanticPruner(k)
#         elif name == 'perplexity':
#             return PerplexityPruner(k)
#         elif name == 'nli':
#             return NliPruner(k)
#         elif name == 'clustering':
#             return ClusteringPruner(k)
#         else:
#             raise ValueError(f"未知的剪枝器名称: {name}")

#     def predict_single(self, text):
#         """对单个文本样本执行完整的AHP预测流程。"""
#         masked_text = adversarial_masking(text, self.model, self.tokenizer)
#         candidates = generate_candidates(masked_text, self.model, self.tokenizer, self.m_val)
#         pruned_candidates = self.pruner.prune(text, candidates)
#         final_predictions = self._get_final_predictions(pruned_candidates)
#         final_result = majority_voting(final_predictions)
        
#         # 如果聚合失败，提供一个基于任务的默认值
#         if final_result is None:
#             return "negative" if self.task == 'sst2' else "World"
            
#         return final_result
        
#     def _build_classification_prompt(self, sentence):
#         """根据任务类型构建分类prompt。"""
#         if self.task == 'sst2':
#             return f"""A chat between a user and an artificial intelligence assistant.
# USER: You are an expert in sentiment analysis. Classify the sentiment of the following movie review as "positive" or "negative".

# Review: "An astonishing cinematic achievement."
# Sentiment: positive

# Review: "A complete waste of time and money."
# Sentiment: negative

# Review: "{sentence}"
# Sentiment:
# ASSISTANT:"""
#         elif self.task == 'ag_news':
#             return f"""A chat between a user and an artificial intelligence assistant.
# USER: You are an expert in news classification. Classify the following news article into one of four categories: "World", "Sports", "Business", or "Technology".

# Article: "The tech giant launched its new flagship phone with a revolutionary camera system."
# Category: Technology

# Article: "The stock market hit a record high after the central bank's announcement."
# Category: Business

# Article: "{sentence}"
# Category:
# ASSISTANT:"""
#         else:
#             raise ValueError(f"未知的任务类型: {self.task}")

#     def _parse_prediction(self, generated_text):
#         """根据任务类型解析模型的输出。"""
#         text = generated_text.lower().strip()
#         if self.task == 'sst2':
#             if text.startswith("positive"): return "positive"
#             if text.startswith("negative"): return "negative"
#         elif self.task == 'ag_news':
#             # 提高解析的鲁棒性
#             if "world" in text: return "World"
#             if "sports" in text: return "Sports"
#             if "business" in text: return "Business"
#             if "technology" in text: return "Technology"
        
#         return "unknown" 

#     def _get_final_predictions(self, candidates):
#         """使用主LLM对K个候选进行最终预测。"""
#         predictions = []
#         for candidate_text in candidates:
#             prompt = self._build_classification_prompt(candidate_text)
#             inputs = self.tokenizer(prompt, return_tensors="pt", return_attention_mask=False).to(self.model.device)
            
#             with torch.no_grad():
#                 # 增加生成token数量以容纳更长的类别名称
#                 output = self.model.generate(**inputs, max_new_tokens=10, pad_token_id=self.tokenizer.eos_token_id)

#             prediction_text = self.tokenizer.decode(output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
#             parsed_pred = self._parse_prediction(prediction_text)
#             predictions.append(parsed_pred)
            
#         return predictions