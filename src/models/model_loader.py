# src/models/model_loader.py
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, RobertaForMaskedLM, RobertaTokenizer
import numpy as np
# from tqdm.auto import tqdm  <-- 禁用这里的 tqdm，防止刷屏
import logging
from typing import List, Dict, Union, Optional
import copy
import random
import gc

try:
    from ..args_config import AHPSettings 
    from ..components.masking import AdversarialMasker, RandomMasker
    from ..components.candidate_generation import CandidateGenerator
    from ..pruning.base_pruner import BasePruner
    from ..pruning import PerplexityPruner, SemanticPruner, NLIPruner, ClusteringPruner
    from ..components.result_aggregation import aggregate_results
except ImportError as e:
     logging.error(f"无法导入 AHP 组件，请检查 model_loader.py 中的导入路径: {e}")
     class BasePruner: pass
     class AdversarialMasker: pass
     class CandidateGenerator: pass
     def aggregate_results(*args, **kwargs): return np.array([0.5, 0.5])


# --- 定义 Prompt 模板 ---
ALPACA_TEMPLATE = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
"""

ALPACA_TEMPLATE_NO_INPUT = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{}

### Response:
"""

# --- 定义数据集特定指令和标签映射 ---
DATASET_INSTRUCTIONS = {
    "sst2": {
        "classification": "Given an English sentence input, determine its sentiment as positive or negative. Respond with positive or negative only.",
        "denoise_explicit": """Fill in the masked word {} with a suitable word. The output sentence must be natural, coherent, and the same length as the input. Respond with the completed sentence directly.

### Input:
a {} , funny and {} transporting re-imagining {} {} and the beast and 1930s {} films

### Response:
a stirring , funny and finally transporting re-imagining of beauty and the beast and 1930s horror films

### Input:
{}""",
        "label_map": {"negative": 0, "positive": 1}, 
        "label_tokens": [29940, 9135] 
    },
    "agnews": {
        "classification": """Classify the news article based on its title and description into one of the four categories: World, Sports, Business, or Science/Technology. Respond with the category name only.

### Input:
{}""",
       "denoise_explicit": """Please replace each masked position in the input sentence with a suitable word to make it natural and coherent. Each mask must be replaced by only one word. Return the completed sentence directly.

### Input:
{}""",
        "label_map": {"World": 0, "Sports": 1, "Business": 2, "Technology": 3}, 
        "label_tokens": [14058, 29903, 16890, 7141] 
    }
}

class AlpacaModel:
    def __init__(self, args: AHPSettings):
        self.args = args
        self.device = torch.device(args.device)
        self.tokenizer: Optional[transformers.PreTrainedTokenizer] = None 
        self.model: Optional[transformers.PreTrainedModel] = None 
        self.roberta_tokenizer: Optional[RobertaTokenizer] = None 
        self.roberta_model: Optional[RobertaForMaskedLM] = None 
        self._load_model()

        self.adversarial_masker: Optional[AdversarialMasker] = None 
        self.random_masker: Optional[RandomMasker] = None 
        self.candidate_generator: Optional[CandidateGenerator] = None
        self.pruner: Optional[BasePruner] = None 

        self.set_dataset_mode(args.dataset_name)
        self._initialize_maskers()

    def _initialize_maskers(self):
        if (self.args.defense_method == 'ahp' or self.args.defense_method == 'topk') and self.adversarial_masker is None:
             try:
                 self.adversarial_masker = AdversarialMasker(self) 
                 logging.info("已初始化 AHP 所需的对抗性遮蔽器。")
             except Exception as e:
                 logging.error(f"初始化 AdversarialMasker 时出错: {e}", exc_info=True)
                 raise e
        if (self.args.defense_method == 'selfdenoise' or 
            (self.args.defense_method == 'ahp' and self.args.ahp_masking_strategy == 'random')) and \
           self.random_masker is None:
            
            self.random_masker = RandomMasker(self.tokenizer, mask_token=self.args.mask_token, mask_rate=self.args.mask_rate)
            logging.info("已初始化[随机]遮蔽器 (用于 SelfDenoise 或 AHP-Random)。")

    def _load_model(self):
        logging.info(f"正在从 {self.args.model_path} 加载模型...")
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.args.model_path,
            cache_dir=self.args.cache_dir
        )
        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            self.args.model_path,
            cache_dir=self.args.cache_dir,
            torch_dtype=torch.float16 # 确保开启半精度
        )

        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            logging.warning("分词器没有 pad_token，将使用 eos_token 作为 pad_token。")
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if self.args.mask_token not in self.tokenizer.get_vocab():
             logging.warning(f"遮蔽标记 '{self.args.mask_token}' 不在分词器词汇表中。正在添加为特殊标记...")
             num_added_toks = self.tokenizer.add_special_tokens({'additional_special_tokens': [self.args.mask_token]})
             if num_added_toks > 0:
                 logging.info(f"已添加 {num_added_toks} 个新标记到分词器。")
                 self.model.resize_token_embeddings(len(self.tokenizer))
                 logging.info("已调整模型嵌入层大小。")

        self.model.to(self.device)
        self.model.eval()
        logging.info("模型加载并配置完成。")

    def _load_roberta_denoiser(self):
        if self.roberta_model is None:
            logging.info("正在加载 RoBERTa 去噪器 (roberta-large)...")
            roberta_path = "/root/autodl-tmp/cache/huggingface/hub/models--roberta-large"
            self.roberta_tokenizer = transformers.RobertaTokenizer.from_pretrained(roberta_path, cache_dir=self.args.cache_dir)
            self.roberta_model = transformers.RobertaForMaskedLM.from_pretrained(roberta_path, cache_dir=self.args.cache_dir)

            if self.args.mask_token != self.roberta_tokenizer.mask_token and \
               self.args.mask_token not in self.roberta_tokenizer.get_vocab():
                 logging.warning(f"正在将遮蔽标记 '{self.args.mask_token}' 添加到 RoBERTa 分词器...")
                 num_added = self.roberta_tokenizer.add_special_tokens({'additional_special_tokens': [self.args.mask_token]})
                 if num_added > 0:
                     self.roberta_model.resize_token_embeddings(len(self.roberta_tokenizer))
                     logging.info("已调整 RoBERTa 模型嵌入层大小。")

            self.roberta_model.to(self.device)
            self.roberta_model.eval()
            logging.info("RoBERTa 去噪器加载完成。")


    def set_dataset_mode(self, dataset_name: str):
        if dataset_name not in DATASET_INSTRUCTIONS:
            raise ValueError(f"未找到数据集 '{dataset_name}' 的指令配置。")
        self.dataset_name = dataset_name
        self.instructions = DATASET_INSTRUCTIONS[dataset_name]
        self.classification_instruction = self.instructions["classification"]
        self.denoise_instruction_template = self.instructions["denoise_explicit"]
        self.label_map = self.instructions["label_map"]
        self.num_labels = len(self.label_map)
        self.label_tokens = self.instructions["label_tokens"]
        logging.info(f"模型已设置为处理数据集: {dataset_name}")

    def _get_pruner(self) -> Optional[BasePruner]:
        method = self.args.ahp_pruning_method
        threshold = self.args.ahp_pruning_threshold 

        if self.pruner is not None and self.pruner.__class__.__name__.lower().startswith(method):
             return self.pruner 

        logging.info(f"正在初始化剪枝器: {method}，参数/阈值: {threshold}")
        try:
            if method == 'perplexity':
                self.pruner = PerplexityPruner(self.model, self.tokenizer, threshold=threshold, device=self.device)
            elif method == 'semantic':
                self.pruner = SemanticPruner(threshold=threshold, device=self.device)
            elif method == 'nli':
                self.pruner = NLIPruner(threshold=threshold, device=self.device)
            elif method == 'clustering':
                 self.pruner = ClusteringPruner(n_clusters=int(threshold), device=self.device)
            elif method == 'none':
                self.pruner = None
            else:
                raise ValueError(f"未知的剪枝方法: {method}")
            return self.pruner
        except Exception as e: 
             logging.error(f"初始化剪枝器 '{method}' 时出错: {e}", exc_info=True)
             raise RuntimeError(f"无法初始化剪枝器 '{method}'")


    def _format_prompt(self, instruction: str, input_text: str) -> str:
        if input_text:
            if self.dataset_name == 'agnews' and "### Input:" in instruction and "{}" in instruction:
                 full_instruction = instruction.format(input_text)
                 return ALPACA_TEMPLATE_NO_INPUT.format(full_instruction)
            else:
                 return ALPACA_TEMPLATE.format(instruction, input_text)
        else:
            return ALPACA_TEMPLATE_NO_INPUT.format(instruction)

    @torch.no_grad()
    def _generate_batch(self, prompts: List[str], max_new_tokens=80) -> List[str]:
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True,
                                max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        generate_ids = self.model.generate(
            **inputs, 
            max_new_tokens=max_new_tokens,
            # --- [修复] 关闭采样，回归确定性生成 ---
            do_sample=False,  # <--- 改为 False
            # temperature=0.7, # 采样参数不再需要
            # top_p=0.9,
            num_return_sequences=1,
        )

        input_len = inputs["input_ids"].shape[1]
        output_ids = generate_ids[:, input_len:]
        outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return outputs

    @torch.no_grad()
    def _get_logit_probs_batch(self, prompts: List[str]) -> torch.Tensor:
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True,
                                max_length=self.args.max_seq_length)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        if 'token_type_ids' in inputs:
             inputs.pop('token_type_ids')

        outputs = self.model(**inputs)
        last_token_logits = outputs.logits[:, -1, :]
        label_logits = last_token_logits[:, self.label_tokens]
        label_probs = torch.softmax(label_logits, dim=-1)
        return label_probs.cpu()


    def _apply_ahp_defense(self, texts: List[str]) -> List[np.ndarray]:
        """[已修改] 应用完整的 AHP 防御流程。禁用内部 tqdm 以防止刷屏。"""
        # logging.info("正在应用 AHP 防御...")
        final_aggregated_probs = []

        if self.adversarial_masker is None:
             self._initialize_maskers()
             if self.adversarial_masker is None:
                 raise RuntimeError("对抗性遮蔽器未能初始化，无法执行 AHP 防御。")
        
        if self.candidate_generator is None:
            try:
                self.candidate_generator = CandidateGenerator(self)
                logging.info("已初始化候选生成器。")
            except Exception as e:
                 logging.error(f"初始化 CandidateGenerator 时出错: {e}", exc_info=True)
                 raise RuntimeError("无法初始化 CandidateGenerator")

        current_pruner = self._get_pruner()
        
        # [核心修复] 移除了 tqdm
        for text_idx, text in enumerate(texts):
            try:
                masked_texts_list = []
                target_num_candidates = self.args.ahp_num_candidates

                if self.args.ahp_masking_strategy == 'stochastic':
                    masked_texts_list = self.adversarial_masker.mask_input_stochastic(
                        text, 
                        mask_rate=self.args.mask_rate, 
                        num_samples=target_num_candidates, 
                        temperature=self.args.ahp_temperature
                    )
                
                elif self.args.ahp_masking_strategy == 'random':
                    if self.random_masker is None:
                        self._initialize_maskers()
                    masked_texts_list = self.random_masker.mask_input_multiple(
                        text, num_masks=target_num_candidates
                    )

                else: # default: 'adversarial'
                    masked_text, _ = self.adversarial_masker.mask_input(text, self.args.mask_rate)
                    masked_texts_list = [masked_text]

                candidates = []
                
                # === [性能优化补丁] ===
                if self.args.ahp_masking_strategy in ['stochastic', 'random']:
                    # 优化路径：批量 1-to-1 生成，避免 K*K 的计算冗余
                    # 之前的逻辑: 对每个 masked_text 生成 K 个候选但只用第一个
                    # 现在的逻辑: 对所有 masked_texts 一次性 RoBERTa，直接返回等长的候选
                    logging.debug(
                        f"[AHP Optimization] Using generate_one_per_mask for {len(masked_texts_list)} masks"
                    )
                    candidates = self.candidate_generator.generate_one_per_mask(masked_texts_list)
                    
                    if not candidates:
                        logging.warning(f"AHP Stochastic 模式下生成了空候选列表，回退到原始文本")
                        candidates = masked_texts_list  # 回退方案
                else:
                    if masked_texts_list:
                        candidates = self.candidate_generator.generate_candidates(masked_texts_list[0])

                if not candidates:
                    fallback_text = masked_texts_list[0] if masked_texts_list else text
                    candidate_prompts = [
                        self._format_prompt(self.classification_instruction, fallback_text)
                    ]
                    probs_tensor = self._get_logit_probs_batch(candidate_prompts)
                    all_candidate_probs = probs_tensor.numpy()
                else:
                    if current_pruner:
                        pruned_candidates = current_pruner.prune(
                            original_text=text,
                            candidates=candidates,
                            masked_text=masked_texts_list[0] 
                        )
                        if not pruned_candidates:
                            pruned_candidates = candidates
                    else:
                        pruned_candidates = candidates
    
                    candidate_prompts = [
                        self._format_prompt(self.classification_instruction, cand)
                        for cand in pruned_candidates
                    ]
    
                    candidate_probs_list = []
                    # [核心修复] 移除了内部 tqdm
                    for i in range(0, len(candidate_prompts), self.args.model_batch_size):
                        batch_prompts = candidate_prompts[i:i + self.args.model_batch_size]
                        probs_tensor = self._get_logit_probs_batch(batch_prompts)
                        candidate_probs_list.append(probs_tensor)
    
                    if candidate_probs_list:
                        all_candidate_probs = torch.cat(candidate_probs_list, dim=0).numpy()
                    else:
                        all_candidate_probs = np.array([np.ones(self.num_labels) / self.num_labels])
    
                aggregated_prob = aggregate_results(
                    all_candidate_probs,
                    strategy=self.args.ahp_aggregation_strategy
                )
                final_aggregated_probs.append(aggregated_prob)
    
            except Exception as e:
                logging.error(f"AHP 处理错误: {e}", exc_info=True)
                final_aggregated_probs.append(np.ones(self.num_labels) / self.num_labels)
    
            if (text_idx + 1) % 50 == 0:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
    
        # logging.info("AHP 防御应用完成")
        return final_aggregated_probs


    def _denoise_texts(self, masked_texts: List[str], denoiser_type: str ,do_sample: bool = False) -> List[str]:
        denoised_texts = []
        if denoiser_type == 'alpaca':
            # (Alpaca 逻辑部分保持不变，它会调用上面修复过的 _generate_batch)
            template = self.denoise_instruction_template
            last_placeholder_idx = template.rfind('{}')
            if last_placeholder_idx == -1:
                logging.error("去噪指令模板中未找到用于填充输入文本的 '{}' 占位符！")
                return ["Error: Invalid denoise template"] * len(masked_texts)

            temp_marker = "__TEMP_INPUT_PLACEHOLDER__"
            template_with_marker = template[:last_placeholder_idx] + temp_marker + template[last_placeholder_idx+2:]
            instruction_base = template_with_marker.replace('{}', self.args.mask_token)
            final_instruction_template = instruction_base.replace(temp_marker, '{}')

            prompts = [final_instruction_template.format(mt) for mt in masked_texts]
            
            # 移除了 tqdm 以避免嵌套刷屏
            for i in range(0, len(prompts), self.args.model_batch_size):
                 batch_prompts = prompts[i:i + self.args.model_batch_size]
                 responses = self._generate_batch(batch_prompts)
                 denoised_texts.extend(responses)

        elif denoiser_type == 'roberta':
            self._load_roberta_denoiser()
            roberta_mask_token_id = self.roberta_tokenizer.mask_token_id
            roberta_input_texts = [t.replace(self.args.mask_token, self.roberta_tokenizer.mask_token) for t in masked_texts]
            
            outputs = []
            
            # === [性能优化补丁] RoBERTa 独立批大小 ===
            # RoBERTa-Large 只有 300M 参数，远小于 Alpaca-7B。
            # 不应该用 Alpaca 的 batch_size（可能是 4），而应该用更大的值。
            # 根据显存调整：
            #   - 24GB VRAM: 32-48
            #   - 40GB VRAM: 64-96
            #   - 80GB VRAM: 128-256
            roberta_batch_size = 32  # ← 根据你的显存改这个值
            
            logging.debug(
                f"[RoBERTa Optimization] Processing {len(roberta_input_texts)} texts with batch_size={roberta_batch_size}"
            )
            
            for i in range(0, len(roberta_input_texts), roberta_batch_size):
                batch_texts = roberta_input_texts[i:i + roberta_batch_size]
                
                inputs = self.roberta_tokenizer(
                    batch_texts, 
                    return_tensors="pt", 
                    padding=True, 
                    truncation=True, 
                    max_length=self.args.max_seq_length
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    logits = self.roberta_model(**inputs).logits
                
                mask_token_indices = (inputs['input_ids'] == roberta_mask_token_id)
                predicted_token_ids = inputs['input_ids'].clone()
                
                if torch.any(mask_token_indices):
                    # 使用 argmax（确定性），而非采样
                    masked_logits = logits[mask_token_indices]
                    if do_sample:
                        # [修改] 使用随机采样 (Multinomial)
                        # 温度设为 1.0，可以根据需要调整
                        probs = torch.softmax(masked_logits, dim=-1)
                        best_token_ids = torch.multinomial(probs, num_samples=1).squeeze(-1)
                    else:
                        # [修改] 使用确定性 Argmax
                        best_token_ids = torch.argmax(masked_logits, dim=-1)
                    # best_token_ids = torch.argmax(masked_logits, dim=-1)
                    predicted_token_ids[mask_token_indices] = best_token_ids
                
                batch_outputs = self.roberta_tokenizer.batch_decode(
                    predicted_token_ids, 
                    skip_special_tokens=True
                )
                outputs.extend(batch_outputs)
            
            denoised_texts = outputs

            # self._load_roberta_denoiser() 

            # roberta_mask_token_id = self.roberta_tokenizer.mask_token_id
            # roberta_input_texts = [t.replace(self.args.mask_token, self.roberta_tokenizer.mask_token) for t in masked_texts]

            # outputs = [] 
            # for i in range(0, len(roberta_input_texts), self.args.model_batch_size):
            #     batch_texts = roberta_input_texts[i:i+self.args.model_batch_size]

            #     inputs = self.roberta_tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=self.args.max_seq_length)
            #     inputs = {k: v.to(self.device) for k, v in inputs.items()} 

            #     with torch.no_grad(): 
            #         logits = self.roberta_model(**inputs).logits 

            #     mask_token_indices = (inputs['input_ids'] == roberta_mask_token_id)
            #     predicted_token_ids = inputs['input_ids'].clone()

            #     if torch.any(mask_token_indices):
            #         # --- [修复] 回归确定性预测 (Argmax) ---
            #         # 我们需要模型最有把握的预测，而不是随机抽样
            #         masked_logits = logits[mask_token_indices]
            #         best_token_ids = torch.argmax(masked_logits, dim=-1) # 取概率最大的词
                    
            #         predicted_token_ids[mask_token_indices] = best_token_ids
            #         # --- 修复结束 ---

            #     batch_outputs = self.roberta_tokenizer.batch_decode(predicted_token_ids, skip_special_tokens=True)
            #     outputs.extend(batch_outputs)
            # denoised_texts = outputs
            
        else:
             raise ValueError(f"未知的去噪器类型: {denoiser_type}")

        cleaned_texts = [t.strip() for t in denoised_texts]
        return cleaned_texts


    def _apply_selfdenoise_defense(self, texts: List[str]) -> List[np.ndarray]:
        # logging.info("正在应用 SelfDenoise 防御...")
        aggregated_probs_list = [] 

        if self.random_masker is None:
             self._initialize_maskers()
             if self.random_masker is None:
                  raise RuntimeError("随机遮蔽器未能初始化，无法执行 SelfDenoise 防御。")

        # [核心修复] 移除了 tqdm
        for text_idx, text in enumerate(texts):
            try:
                masked_texts = self.random_masker.mask_input_multiple(text, self.args.selfdenoise_ensemble_size)
                denoised_candidates = self._denoise_texts(masked_texts, self.args.selfdenoise_denoiser)

                candidate_prompts = [self._format_prompt(self.classification_instruction, cand) for cand in denoised_candidates]
                candidate_probs_list = [] 
                
                # [核心修复] 移除了内部 tqdm
                for i in range(0, len(candidate_prompts), self.args.model_batch_size):
                    batch_prompts = candidate_prompts[i : i + self.args.model_batch_size]
                    probs = self._get_logit_probs_batch(batch_prompts) 
                    candidate_probs_list.append(probs)

                if not candidate_probs_list:
                    all_candidate_probs = np.array([np.ones(self.num_labels) / self.num_labels])
                else:
                    all_candidate_probs = torch.cat(candidate_probs_list, dim=0).numpy()

                predictions = np.argmax(all_candidate_probs, axis=1)
                votes = np.bincount(predictions, minlength=self.num_labels)
                majority_class = np.argmax(votes)

                final_prob = np.zeros(self.num_labels)
                final_prob[majority_class] = 1.0
                aggregated_probs_list.append(final_prob)

            except Exception as e:
                logging.error(f"SelfDenoise 防御出错: {e}", exc_info=True)
                uniform_prob = np.ones(self.num_labels) / self.num_labels
                aggregated_probs_list.append(uniform_prob)

            if (text_idx + 1) % 50 == 0:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

        # logging.info("SelfDenoise 防御应用完成。")
        return aggregated_probs_list


    def _apply_topk_defense(self, texts: List[str]) -> List[np.ndarray]:
            """
            TopK 防御实现：
            1. 梯度排序：识别并遮蔽 Top K% 最可能受攻击的词 (Deterministic)。
            2. 随机填充：使用 RoBERTa 对该遮蔽文本进行多次采样填充 (Ensemble)。
            3. 投票：聚合预测结果。
            """
            aggregated_probs_list = []
            
            # 确保初始化了梯度计算所需的 AdversarialMasker
            if self.adversarial_masker is None:
                 self._initialize_maskers()
                 if self.adversarial_masker is None:
                      raise RuntimeError("TopK 需要 AdversarialMasker 来计算梯度，但初始化失败。")
            
            ensemble_size = self.args.topk_ensemble_size
            
            for text_idx, text in enumerate(texts):
                try:
                    # 1. 梯度排序与遮蔽 (产生 1 个遮蔽文本)
                    # mask_input 内部已经实现了: 计算梯度 -> 排序 -> 遮蔽 Top N
                    masked_text, _ = self.adversarial_masker.mask_input(text, self.args.mask_rate)
                    
                    # 2. 构造 Ensemble 输入
                    # 将同一个 masked_text 复制 N 份
                    masked_texts_batch = [masked_text] * ensemble_size
                    
                    # 3. 随机填充 (RoBERTa Sampling)
                    # 关键点：开启 do_sample=True，使得即使输入相同，输出也不同
                    denoised_candidates = self._denoise_texts(
                        masked_texts_batch, 
                        denoiser_type='roberta', # 强制使用 RoBERTa，因为它支持 MaskedLM 采样
                        do_sample=True 
                    )
                    
                    # 4. 预测与投票
                    candidate_prompts = [self._format_prompt(self.classification_instruction, cand) for cand in denoised_candidates]
                    candidate_probs_list = [] 
                    
                    for i in range(0, len(candidate_prompts), self.args.model_batch_size):
                        batch_prompts = candidate_prompts[i : i + self.args.model_batch_size]
                        probs = self._get_logit_probs_batch(batch_prompts) 
                        candidate_probs_list.append(probs)
    
                    if not candidate_probs_list:
                        all_candidate_probs = np.array([np.ones(self.num_labels) / self.num_labels])
                    else:
                        all_candidate_probs = torch.cat(candidate_probs_list, dim=0).numpy()
    
                    # 多数投票
                    predictions = np.argmax(all_candidate_probs, axis=1)
                    votes = np.bincount(predictions, minlength=self.num_labels)
                    majority_class = np.argmax(votes)
    
                    final_prob = np.zeros(self.num_labels)
                    final_prob[majority_class] = 1.0
                    aggregated_probs_list.append(final_prob)
    
                except Exception as e:
                    logging.error(f"TopK 防御出错: {e}", exc_info=True)
                    aggregated_probs_list.append(np.ones(self.num_labels) / self.num_labels)
    
                if (text_idx + 1) % 50 == 0:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
    
            return aggregated_probs_list
    

    def predict_batch(self, texts: List[str]) -> List[np.ndarray]:
        if self.args.defense_method == 'none':
            # logging.debug("正在执行无防御预测...")
            prompts = [self._format_prompt(self.classification_instruction, text) for text in texts]
            all_probs = [] 
            
            # [核心修复] 移除了 tqdm
            for i in range(0, len(prompts), self.args.model_batch_size):
                 batch_prompts = prompts[i:i + self.args.model_batch_size]
                 probs = self._get_logit_probs_batch(batch_prompts)
                 all_probs.append(probs)

            if not all_probs:
                logging.warning("无防御预测返回了空的概率列表。")
                return [np.ones(self.num_labels) / self.num_labels] * len(texts)

            final_probs_np = torch.cat(all_probs, dim=0).numpy()
            return [p for p in final_probs_np]

        elif self.args.defense_method == 'ahp':
            return self._apply_ahp_defense(texts)
        elif self.args.defense_method == 'selfdenoise':
            return self._apply_selfdenoise_defense(texts)
        elif self.args.defense_method == 'topk':
            return self._apply_topk_defense(texts)
        else:
            raise ValueError(f"未知的防御方法: {self.args.defense_method}")

    def __call__(self, text_input_list: List[str]) -> np.ndarray:
        if not isinstance(text_input_list, list):
            text_input_list = [text_input_list]

        if not text_input_list:
            logging.warning("TextAttack Wrapper 收到了空输入列表。")
            return np.zeros((0, self.num_labels))

        prob_list = self.predict_batch(text_input_list)

        if not prob_list or not isinstance(prob_list, list):
            logging.error(f"predict_batch 未能返回有效列表。")
            return np.array([np.ones(self.num_labels) / self.num_labels] * len(text_input_list))

        valid_probs = []
        expected_shape = (self.num_labels,)
        for i, p in enumerate(prob_list):
             if isinstance(p, np.ndarray) and p.shape == expected_shape:
                 valid_probs.append(p)
             else:
                 valid_probs.append(np.ones(self.num_labels) / self.num_labels)

        if not valid_probs:
             return np.zeros((len(text_input_list), self.num_labels))

        try:
            return np.stack(valid_probs)
        except ValueError as e:
             logging.error(f"在 __call__ 中堆叠概率时出错: {e}")
             return np.array([np.ones(self.num_labels) / self.num_labels] * len(text_input_list))