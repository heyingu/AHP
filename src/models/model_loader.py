# src/models/model_loader.py
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, RobertaForMaskedLM, RobertaTokenizer
import numpy as np
from tqdm.auto import tqdm # 引入 tqdm 以显示进度条
import logging
from typing import List, Dict, Union, Optional
import copy # 用于深拷贝列表等
import random
import gc # 引入垃圾回收器，用于清理 GPU 显存

try:
    from ..args_config import AHPSettings 
    from ..components.masking import AdversarialMasker, RandomMasker
    from ..components.candidate_generation import CandidateGenerator # <--- 确保导入
    from ..pruning.base_pruner import BasePruner
    from ..pruning import PerplexityPruner, SemanticPruner, NLIPruner, ClusteringPruner
    from ..components.result_aggregation import aggregate_results
except ImportError as e:
     logging.error(f"无法导入 AHP 组件，请检查 model_loader.py 中的导入路径: {e}")
     # ... (占位符定义) ...
     class BasePruner: pass
     class AdversarialMasker: pass
     class CandidateGenerator: pass
     def aggregate_results(*args, **kwargs): return np.array([0.5, 0.5])


# --- 定义 Prompt 模板 (来自 SelfDenoise/alpaca.py 和您的 Notebook) ---
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
# 这些指令用于分类任务和 SelfDenoise 的去噪任务
DATASET_INSTRUCTIONS = {
    "sst2": {
        # SST-2 的指令 (保持不变, 它工作正常)
       # "classification": """Given an English sentence input, determine its sentiment. Respond with "positive" or "negative" only.
"classification": """"Given an English sentence input, determine its sentiment as positive or negative. Respond with positive or negative only.",
### Input:
A stirring , funny and finally transporting re-imagining of beauty and the beast and 1930s horror films

### Response:
positive

### Input:
{}""",
        
        
        "denoise_explicit": """Fill in the masked word {} with a suitable word. The output sentence must be natural, coherent, and the same length as the input. Respond with the completed sentence directly.

### Input:
a {} , funny and {} transporting re-imagining {} {} and the beast and 1930s {} films

### Response:
a stirring , funny and finally transporting re-imagining of beauty and the beast and 1930s horror films

### Input:
{}""",
        
        "label_map": {"negative": 0, "positive": 1}, 
        "label_tokens": [29940, 9135] # ' Negative', ' Positive'
    },
    "agnews": {
        # AG News 分类指令 (保持不变)
#         "classification": """Classify the news article based on its title and description into one of the four categories: World, Sports, Business, or Science/Technology. Respond with the category name only.

# ### Input:
# Title: Wall St. Bears Claw Back Into the Black (Reuters)
# Description: Reuters - Short-sellers, Wall Street's dwindling band of ultra-cynics, are seeing green again.

# ### Response:
# Business

# ### Input:
# {}""",
        "classification": """Classify the news article based on its title and description into one of the four categories: World, Sports, Business, or Science/Technology. Respond with the category name only.

### Input:
{}""",
        
        # --- 关键修改在这里 ---
        # 移除了第一个句子中所有 '{}' 占位符，
        # 这样 _denoise_texts 就不会错误地替换它们。
       "denoise_explicit": """Please replace each masked position in the input sentence with a suitable word to make it natural and coherent. Each mask must be replaced by only one word. Return the completed sentence directly.

### Input:
{}""",
        # --- 修改结束 ---

        "label_map": {"World": 0, "Sports": 1, "Business": 2, "Technology": 3}, 
        "label_tokens": [14058, 29903, 16890, 7141] # 'World', 'Sports', 'Business', 'Technology'
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
        self._load_model() # 加载主模型

        # --- AHP 和 SelfDenoise 组件初始化 ---
        self.adversarial_masker: Optional[AdversarialMasker] = None 
        self.random_masker: Optional[RandomMasker] = None 
        self.candidate_generator: Optional[CandidateGenerator] = None # <--- 添加属性
        self.pruner: Optional[BasePruner] = None 

        self.set_dataset_mode(args.dataset_name)
        self._initialize_maskers() # 调用初始化遮蔽器的方法

    def _initialize_maskers(self):
        """根据配置的防御方法，初始化所需的遮蔽器。"""
        if self.args.defense_method == 'ahp' and self.adversarial_masker is None:
             try:
                 # --- 修改这里 ---
                 # 传递 self (AlpacaModel 实例)
                 self.adversarial_masker = AdversarialMasker(self) 
                 # --- 修改结束 ---
                 logging.info("已初始化 AHP 所需的对抗性遮蔽器。")
             except Exception as e:
                 logging.error(f"初始化 AdversarialMasker 时出错: {e}", exc_info=True)
                 raise e
        if (self.args.defense_method == 'selfdenoise' or 
            (self.args.defense_method == 'ahp' and self.args.ahp_masking_strategy == 'random')) and \
           self.random_masker is None:
            
            self.random_masker = RandomMasker(self.tokenizer, mask_token=self.args.mask_token, mask_rate=self.args.mask_rate)
            logging.info("已初始化[随机]遮蔽器 (用于 SelfDenoise 或 AHP-Random)。")
            
        # if self.args.defense_method == 'selfdenoise' and self.random_masker is None:
        #     self.random_masker = RandomMasker(self.tokenizer, mask_token=self.args.mask_token, mask_rate=self.args.mask_rate)
        #     logging.info("已初始化 SelfDenoise 所需的随机遮蔽器。")

    def _load_model(self):
        """加载 Alpaca 模型和分词器到指定设备。"""
        logging.info(f"正在从 {self.args.model_path} 加载模型...")
        # 加载分词器
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.args.model_path,
            cache_dir=self.args.cache_dir # 指定缓存目录
        )
        # 加载模型
        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            self.args.model_path,
            cache_dir=self.args.cache_dir,
            # torch_dtype=torch.float16 # 如果显存不足，可以考虑使用 float16 加载
        )

        # --- 配置分词器 ---
        self.tokenizer.padding_side = "left" # LLM 通常需要左填充
        # 检查并设置填充标记 (pad_token)，如果不存在则使用句子结束标记 (eos_token)
        if self.tokenizer.pad_token is None:
            logging.warning("分词器没有 pad_token，将使用 eos_token 作为 pad_token。")
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # --- 处理遮蔽标记 (mask_token) ---
        # 检查用户指定的遮蔽标记是否在词汇表中
        # 如果不在，需要将其添加为特殊标记，以防止分词器将其拆分
        if self.args.mask_token not in self.tokenizer.get_vocab():
             logging.warning(f"遮蔽标记 '{self.args.mask_token}' 不在分词器词汇表中。正在添加为特殊标记...")
             # 使用 add_special_tokens 添加，这样模型会将其视为一个整体
             num_added_toks = self.tokenizer.add_special_tokens({'additional_special_tokens': [self.args.mask_token]})
             if num_added_toks > 0:
                 logging.info(f"已添加 {num_added_toks} 个新标记到分词器。")
                 # 如果添加了新标记，需要调整模型嵌入层的大小以匹配新的词汇表大小
                 self.model.resize_token_embeddings(len(self.tokenizer))
                 logging.info("已调整模型嵌入层大小。")

        # 将模型移动到指定设备 (GPU 或 CPU)
        self.model.to(self.device)
        # 将模型设置为评估模式 (关闭 dropout 等)
        self.model.eval()
        logging.info("模型加载并配置完成。")

    def _load_roberta_denoiser(self):
        """按需加载 RoBERTa 模型作为 SelfDenoise 的去噪器。"""
        # 只有在 RoBERTa 模型未加载时才执行加载操作
        # --- 检查这里的缩进 ---
        if self.roberta_model is None: # <--- 这一行应该和下面的 logging.info 对齐
            logging.info("正在加载 RoBERTa 去噪器 (roberta-large)...")
            roberta_path = "/root/autodl-tmp/cache/huggingface/hub/models--roberta-large" # 使用 Hugging Face Hub 上的标准 RoBERTa 模型
            # 加载 RoBERTa 的分词器
            # --- 确保下面所有行都相对于 if 语句正确缩进 ---
            self.roberta_tokenizer = transformers.RobertaTokenizer.from_pretrained(roberta_path, cache_dir=self.args.cache_dir)
            # 加载 RoBERTa 的 Masked Language Model (用于填充遮蔽词)
            self.roberta_model = transformers.RobertaForMaskedLM.from_pretrained(roberta_path, cache_dir=self.args.cache_dir)

            # --- 检查 RoBERTa 分词器是否包含我们的遮蔽标记 ---
            if self.args.mask_token != self.roberta_tokenizer.mask_token and \
               self.args.mask_token not in self.roberta_tokenizer.get_vocab():
                 logging.warning(f"正在将遮蔽标记 '{self.args.mask_token}' 添加到 RoBERTa 分词器...")
                 num_added = self.roberta_tokenizer.add_special_tokens({'additional_special_tokens': [self.args.mask_token]})
                 if num_added > 0:
                     self.roberta_model.resize_token_embeddings(len(self.roberta_tokenizer)) # 调整嵌入层大小
                     logging.info("已调整 RoBERTa 模型嵌入层大小。")

            # 将 RoBERTa 模型移动到设备并设为评估模式
            self.roberta_model.to(self.device)
            self.roberta_model.eval()
            logging.info("RoBERTa 去噪器加载完成。")


    def set_dataset_mode(self, dataset_name: str):
        """根据数据集名称，设置模型所需的指令、标签映射等。"""
        if dataset_name not in DATASET_INSTRUCTIONS:
            raise ValueError(f"未找到数据集 '{dataset_name}' 的指令配置。")
        self.dataset_name = dataset_name
        # 获取该数据集的所有配置信息
        self.instructions = DATASET_INSTRUCTIONS[dataset_name]
        # 设置分类任务的指令
        self.classification_instruction = self.instructions["classification"]
        # 设置去噪任务的指令模板
        self.denoise_instruction_template = self.instructions["denoise_explicit"]
        # 设置标签名称到整数 ID 的映射
        self.label_map = self.instructions["label_map"]
        # 获取标签数量
        self.num_labels = len(self.label_map)
        # 获取与标签对应的 token ID 列表 (用于直接从 logits 计算概率)
        self.label_tokens = self.instructions["label_tokens"]
        logging.info(f"模型已设置为处理数据集: {dataset_name}")

    def _get_pruner(self) -> Optional[BasePruner]:
        """[已修改] 根据配置参数，按需初始化并返回 AHP 剪枝器实例。"""
        method = self.args.ahp_pruning_method
        threshold = self.args.ahp_pruning_threshold 

        if self.pruner is not None and self.pruner.__class__.__name__.lower().startswith(method):
             return self.pruner 

        logging.info(f"正在初始化剪枝器: {method}，参数/阈值: {threshold}")
        try:
            # --- 确保剪枝器在初始化时接收到所需的模型/分词器 ---
            if method == 'perplexity':
                # PerplexityPruner 需要主模型
                self.pruner = PerplexityPruner(self.model, self.tokenizer, threshold=threshold, device=self.device)
            elif method == 'semantic':
                # SemanticPruner 加载自己的模型，不需要主模型
                self.pruner = SemanticPruner(threshold=threshold, device=self.device)
            elif method == 'nli':
                 # NLIPruner 加载自己的模型，不需要主模型
                self.pruner = NLIPruner(threshold=threshold, device=self.device)
            elif method == 'clustering':
                 # ClusteringPruner 加载自己的模型，不需要主模型
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
        """使用 Alpaca 的模板格式化指令和输入。"""
        if input_text:
            # 特殊处理 AG News：输入文本（标题+描述）需要填入指令模板的占位符中
            if self.dataset_name == 'agnews' and "### Input:" in instruction and "{}" in instruction:
                 # 将 input_text 填入 instruction 中的 {}
                 full_instruction = instruction.format(input_text)
                 # 使用无输入的模板，因为输入已经合并到指令里了
                 return ALPACA_TEMPLATE_NO_INPUT.format(full_instruction)
            else:
                 # 对于其他情况（如 SST-2），使用标准带输入的模板
                 return ALPACA_TEMPLATE.format(instruction, input_text)
        else:
            # 如果没有输入文本（例如去噪任务的 Prompt），使用无输入的模板
            return ALPACA_TEMPLATE_NO_INPUT.format(instruction)

    @torch.no_grad() # 禁用梯度计算，节省显存和计算资源
    def _generate_batch(self, prompts: List[str], max_new_tokens=80) -> List[str]:
        """内部辅助函数，用于为一批 Prompt 生成文本回复 (主要用于去噪)。"""
        # 使用分词器处理一批 Prompt，进行填充和截断
        # 允许更长的输入长度，因为包含 few-shot 示例的 Prompt 可能较长
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True,
                                max_length=512)
        # 将输入张量移动到指定设备
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # 调用模型的 generate 方法生成文本
        generate_ids = self.model.generate(
            **inputs, 
            max_new_tokens=max_new_tokens, 
            
            # --- 修改开始 ---
            do_sample=True,             # <--- 启用采样
            temperature=0.7,            # <--- 增加一点随机性 (可调)
            top_p=0.9,                  # <--- 使用 nucleus sampling
            num_return_sequences=1,     # <--- 每次调用只返回一个序列
            # --- 移除无效参数 ---
            # (日志警告 'pad_token_id' 无效，我们移除它，
            #  模型会默认使用 eos_token_id 作为停止符)
            # pad_token_id=self.tokenizer.eos_token_id 
            # --- 修改结束 ---
        )

        # 解码生成的部分
        input_len = inputs["input_ids"].shape[1] # 获取输入部分的长度
        output_ids = generate_ids[:, input_len:] # 只取生成的新 token ID
        # 使用分词器将 token ID 解码回文本字符串
        # skip_special_tokens=True 会移除像 <eos> 这样的特殊标记
        outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return outputs

    @torch.no_grad() # 禁用梯度计算
    def _get_logit_probs_batch(self, prompts: List[str]) -> torch.Tensor:
        """
        获取模型对一批 Prompt 的预测概率分布 (仅针对预定义的标签词)。
        用于分类任务。
        """
        # 使用分词器处理一批 Prompt
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True,
                                max_length=self.args.max_seq_length) # 使用配置的最大序列长度
        # 将输入移动到设备
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        # 某些模型 (如 RoBERTa) 不使用 token_type_ids，如果存在则移除
        if 'token_type_ids' in inputs:
             inputs.pop('token_type_ids')

        # 获取模型输出 (包含所有词的 logits)
        outputs = self.model(**inputs)
        # 获取最后一个 token 位置的 logits (通常认为这里包含了分类信息)
        last_token_logits = outputs.logits[:, -1, :]
        # 从所有词的 logits 中，只选择我们关心的标签对应的 token ID 的 logits
        label_logits = last_token_logits[:, self.label_tokens]
        # 对标签 logits 应用 softmax 函数，得到概率分布
        label_probs = torch.softmax(label_logits, dim=-1)
        # 将结果从 GPU 移回 CPU 并返回
        return label_probs.cpu()


    def _apply_ahp_defense(self, texts: List[str]) -> List[np.ndarray]:
        """[已修改] 应用完整的 AHP (对抗性层次处理) 防御流程。"""
        logging.info("正在应用 AHP 防御...")
        final_aggregated_probs = []

        # --- 确保 AHP 组件已初始化 ---
        if self.adversarial_masker is None:
             self._initialize_maskers()
             if self.adversarial_masker is None:
                 raise RuntimeError("对抗性遮蔽器未能初始化，无法执行 AHP 防御。")
        
        # 确保候选生成器已初始化
        if self.candidate_generator is None:
            try:
                # --- 修改这里 ---
                # 实例化时传入 self (AlpacaModel 实例)
                self.candidate_generator = CandidateGenerator(self)
                # --- 修改结束 ---
                logging.info("已初始化候选生成器。")
            except Exception as e:
                 logging.error(f"初始化 CandidateGenerator 时出错: {e}", exc_info=True)
                 raise RuntimeError("无法初始化 CandidateGenerator")

        current_pruner = self._get_pruner() # 获取剪枝器实例

        # --- 逐个处理文本 (使用 enumerate 获取索引) ---
        for text_idx, text in enumerate(tqdm(texts, desc="AHP 防御流程", leave=False, ncols=100)):
            try:
                # 1. 对抗性遮蔽
                # (现在会调用已完善的 _calculate_word_importance)
                # masked_text, masked_indices = self.adversarial_masker.mask_input(text, self.args.mask_rate)
                # logging.debug(f"AHP Masked Text: {masked_text}")
                if self.args.ahp_masking_strategy == 'random':
                    # 1. 随机遮蔽
                    if self.random_masker is None: # 兜底检查
                         raise RuntimeError("AHP-Random 模式需要 RandomMasker，但它未被初始化。")
                    # 注意：RandomMasker.mask_input 使用在 __init__ 中设置的 mask_rate
                    masked_text, masked_indices = self.random_masker.mask_input(text)
                    logging.debug(f"AHP [Random] Masked Text: {masked_text}")

                else: # 默认 'adversarial'
                    # 1. 对抗性遮蔽
                    if self.adversarial_masker is None: # 兜底检查
                         raise RuntimeError("AHP-Adversarial 模式需要 AdversarialMasker，但它未被初始化。")
                    # 注意：AdversarialMasker.mask_input 需要传入 mask_rate
                    masked_text, masked_indices = self.adversarial_masker.mask_input(text, self.args.mask_rate)
                    logging.debug(f"AHP [Adversarial] Masked Text: {masked_text}")

                # 2. 候选生成
                # (现在会调用已完善的 generate_candidates)
                candidates = self.candidate_generator.generate_candidates(masked_text)
                logging.debug(f"Generated {len(candidates)} candidates.")

                # (处理无候选的情况)
                if not candidates:
                    logging.warning(f"No candidates generated for: {text[:50]}... Skipping pruning/prediction.")
                    candidate_prompts = [self._format_prompt(self.classification_instruction, masked_text)]
                    probs_tensor = self._get_logit_probs_batch(candidate_prompts)
                    all_candidate_probs = probs_tensor.numpy()
                else:
                    # 3. 剪枝
                    if current_pruner:
                        pruned_candidates = current_pruner.prune(original_text=text, candidates=candidates, masked_text=masked_text)
                        logging.debug(f"Pruned to {len(pruned_candidates)} candidates using {self.args.ahp_pruning_method}.")
                        if not pruned_candidates:
                             logging.warning("Pruning removed all candidates. Predicting on original candidates.")
                             pruned_candidates = candidates 
                    else:
                        pruned_candidates = candidates
                        logging.debug("No pruning applied.")

                    # 4. 预测 (分批)
                    candidate_prompts = [self._format_prompt(self.classification_instruction, cand) for cand in pruned_candidates]
                    candidate_probs_list = []
                    for i in range(0, len(candidate_prompts), self.args.model_batch_size):
                        batch_prompts = candidate_prompts[i:i + self.args.model_batch_size]
                        probs_tensor = self._get_logit_probs_batch(batch_prompts)
                        candidate_probs_list.append(probs_tensor)

                    if not candidate_probs_list:
                         logging.error(f"Prediction on candidates failed for: {text[:50]}...")
                         all_candidate_probs = np.array([np.ones(self.num_labels) / self.num_labels])
                    else:
                         all_candidate_probs = torch.cat(candidate_probs_list, dim=0).numpy()

                # 5. 结果聚合 (确保 aggregate_results 已正确导入)
                aggregated_prob = aggregate_results(all_candidate_probs, strategy=self.args.ahp_aggregation_strategy)
                final_aggregated_probs.append(aggregated_prob)
                logging.debug(f"Aggregated Prob: {aggregated_prob}")

            except Exception as e:
                logging.error(f"处理文本 '{text[:50]}...' 时 AHP 防御出错: {e}", exc_info=True)
                uniform_prob = np.ones(self.num_labels) / self.num_labels
                final_aggregated_probs.append(uniform_prob)

            # --- 显存清理 (使用 text_idx) ---
            if text_idx % 10 == 0 and text_idx > 0: 
                if torch.cuda.is_available():
                    logging.debug(f"AHP: 清理显存 (样本 {text_idx})")
                    torch.cuda.empty_cache()
                    gc.collect()

        logging.info("AHP 防御应用完成。")
        return final_aggregated_probs


    def _denoise_texts(self, masked_texts: List[str], denoiser_type: str) -> List[str]:
        """
        [已修改] 使用 Alpaca 或 RoBERTa 对一批被遮蔽的文本进行去噪（恢复）。
        """
        denoised_texts = []
        if denoiser_type == 'alpaca':
            # (Alpaca 部分保持不变 - 即使它不能用，我们先保留它)
            logging.debug(f"使用 Alpaca 去噪 {len(masked_texts)} 个文本...")
            template = self.denoise_instruction_template
            last_placeholder_idx = template.rfind('{}')
            if last_placeholder_idx == -1:
                logging.error("去噪指令模板中未找到用于填充输入文本的 '{}' 占位符！")
                return ["Error: Invalid denoise template"] * len(masked_texts)

            temp_marker = "__TEMP_INPUT_PLACEHOLDER__"
            template_with_marker = template[:last_placeholder_idx] + temp_marker + template[last_placeholder_idx+2:]
            instruction_base = template_with_marker.replace('{}', self.args.mask_token)
            final_instruction_template = instruction_base.replace(temp_marker, '{}')

            prompts = [self._format_prompt(final_instruction_template.format(mt), "") for mt in masked_texts]
            for i in tqdm(range(0, len(prompts), self.args.model_batch_size), desc="去噪 (Alpaca)", leave=False, ncols=100):
                 batch_prompts = prompts[i:i + self.args.model_batch_size]
                 gen_max_tokens = int(self.args.max_seq_length * 1.5) 
                 responses = self._generate_batch(batch_prompts, max_new_tokens=gen_max_tokens)
                 denoised_texts.extend(responses)

        elif denoiser_type == 'roberta':
            # --- 关键修改：使 RoBERTa 能够采样 ---
            logging.debug(f"使用 RoBERTa 去噪 {len(masked_texts)} 个文本 (带采样)...")
            self._load_roberta_denoiser() 

            roberta_mask_token_id = self.roberta_tokenizer.mask_token_id
            
            # (确保使用 <unk> 或其他 mask_token 都能正确替换)
            roberta_input_texts = [t.replace(self.args.mask_token, self.roberta_tokenizer.mask_token) for t in masked_texts]

            outputs = [] 
            for i in tqdm(range(0, len(roberta_input_texts), self.args.model_batch_size), desc="去噪 (RoBERTa)", leave=False, ncols=100):
                batch_texts = roberta_input_texts[i:i+self.args.model_batch_size]

                inputs = self.roberta_tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=self.args.max_seq_length)
                inputs = {k: v.to(self.device) for k, v in inputs.items()} 

                with torch.no_grad(): 
                    logits = self.roberta_model(**inputs).logits 

                mask_token_indices = (inputs['input_ids'] == roberta_mask_token_id)
                predicted_token_ids = inputs['input_ids'].clone()

                if torch.any(mask_token_indices):
                    # --- 修改开始 ---
                    # 不要使用 argmax (确定性)，我们要采样
                    # 1. 获取遮蔽位置的 logits
                    masked_logits = logits[mask_token_indices] # [num_masks_in_batch, vocab_size]
                    
                    # 2. 对 logits 应用 softmax 得到概率
                    masked_probs = torch.softmax(masked_logits, dim=-1)
                    
                    # 3. 从概率分布中采样 1 个 token
                    # (torch.multinomial 需要 1D 或 2D tensor, masked_probs 是 2D, 所以 OK)
                    # num_samples=1
                    sampled_token_ids = torch.multinomial(masked_probs, num_samples=1).squeeze(-1) # [num_masks_in_batch]
                    
                    # 4. 用采样到的 token ID 替换 '<mask>' ID
                    predicted_token_ids[mask_token_indices] = sampled_token_ids
                    # --- 修改结束 ---

                batch_outputs = self.roberta_tokenizer.batch_decode(predicted_token_ids, skip_special_tokens=True)
                outputs.extend(batch_outputs)
            denoised_texts = outputs
            # --- 关键修改结束 ---
            
        else:
             raise ValueError(f"未知的去噪器类型: {denoiser_type}")

        cleaned_texts = [t.strip() for t in denoised_texts]
        logging.debug(f"去噪后示例文本 (前 50 字符): {cleaned_texts[0][:50] if cleaned_texts else '无'}")
        return cleaned_texts


    def _apply_selfdenoise_defense(self, texts: List[str]) -> List[np.ndarray]:
        """
        应用 SelfDenoise 防御流程 (随机遮蔽 -> 去噪 -> 集成预测)。

        Args:
            texts (List[str]): 需要进行防御和预测的原始文本列表。

        Returns:
            List[np.ndarray]: 每个输入文本对应的最终集成预测概率分布列表 (通常是 one-hot 形式)。
        """
        logging.info("正在应用 SelfDenoise 防御...")
        aggregated_probs_list = [] # 存储最终结果

        # --- 确保随机遮蔽器已初始化 ---
        if self.random_masker is None:
             self._initialize_maskers()
             if self.random_masker is None:
                  raise RuntimeError("随机遮蔽器未能初始化，无法执行 SelfDenoise 防御。")

        # --- 逐个处理输入文本 ---
        for text_idx, text in enumerate(tqdm(texts, desc="SelfDenoise 防御流程", leave=False, ncols=100)):
            try:
                # --- 1. 生成多个随机遮蔽版本 ---
                # 调用随机遮蔽器的 mask_input_multiple 方法
                masked_texts = self.random_masker.mask_input_multiple(text, self.args.selfdenoise_ensemble_size)
                logging.debug(f"为 SelfDenoise 生成了 {len(masked_texts)} 个遮蔽版本。")

                # --- 2. 对所有遮蔽版本进行去噪 ---
                # 调用内部的 _denoise_texts 方法
                denoised_candidates = self._denoise_texts(masked_texts, self.args.selfdenoise_denoiser)

                # --- 3. 对去噪后的候选进行预测 (分批处理) ---
                # 为每个去噪后的候选构建分类 Prompt
                candidate_prompts = [self._format_prompt(self.classification_instruction, cand) for cand in denoised_candidates]
                candidate_probs_list = [] # 存储每个批次的预测概率
                # 分批预测
                for i in range(0, len(candidate_prompts), self.args.model_batch_size):
                    batch_prompts = candidate_prompts[i : i + self.args.model_batch_size]
                    probs = self._get_logit_probs_batch(batch_prompts) # 获取概率 Tensor
                    candidate_probs_list.append(probs)

                # 处理未能获取任何预测概率的情况
                if not candidate_probs_list:
                    logging.warning(f"未能为 SelfDenoise 候选生成预测概率: {text[:50]}...")
                    # Fallback: 使用均匀分布作为所有候选的概率
                    all_candidate_probs = np.array([np.ones(self.num_labels) / self.num_labels])
                else:
                    # 合并所有批次的概率 Tensor 为一个 Numpy 数组 [ensemble_size, num_labels]
                    all_candidate_probs = torch.cat(candidate_probs_list, dim=0).numpy()

                # --- 4. 结果聚合 (多数投票) ---
                # 对于随机平滑 (SelfDenoise 的基础)，标准做法是对预测的类别进行多数投票
                predictions = np.argmax(all_candidate_probs, axis=1) # 获取每个候选的预测类别 ID
                votes = np.bincount(predictions, minlength=self.num_labels) # 统计每个类别的票数
                majority_class = np.argmax(votes) # 找到得票最多的类别

                # 将多数投票结果转换为 one-hot 概率分布 (即，最终预测为 1，其他为 0)
                final_prob = np.zeros(self.num_labels)
                final_prob[majority_class] = 1.0
                aggregated_probs_list.append(final_prob) # 添加到最终结果列表
                logging.debug(f"SelfDenoise 投票结果: {votes}, 多数类别: {majority_class}")

            except Exception as e:
                # 捕获处理单个文本时的错误
                logging.error(f"处理文本 '{text[:50]}...' 时 SelfDenoise 防御出错: {e}", exc_info=True)
                # Fallback: 返回均匀分布
                uniform_prob = np.ones(self.num_labels) / self.num_labels
                aggregated_probs_list.append(uniform_prob)

            # --- 可选：显存清理 ---
            if text_idx % 10 == 0 and text_idx > 0:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    gc.collect()

        logging.info("SelfDenoise 防御应用完成。")
        return aggregated_probs_list


    def predict_batch(self, texts: List[str]) -> List[np.ndarray]:
        """
        主要的批量预测函数，根据配置的防御方法路由到相应的处理流程。

        Args:
            texts (List[str]): 需要预测的文本列表。

        Returns:
            List[np.ndarray]: 每个输入文本对应的预测概率分布 (numpy 数组) 列表。
        """
        # 根据 self.args.defense_method 的值决定执行哪个流程
        if self.args.defense_method == 'none':
            # --- 无防御：直接预测 ---
            logging.debug("正在执行无防御预测...")
            # 构建分类 Prompt
            prompts = [self._format_prompt(self.classification_instruction, text) for text in texts]
            all_probs = [] # 存储所有批次的概率
            # 分批处理
            # 使用 tqdm 显示预测进度
            for i in tqdm(range(0, len(prompts), self.args.model_batch_size), desc="无防御预测", leave=False, ncols=100):
                 batch_prompts = prompts[i:i + self.args.model_batch_size]
                 probs = self._get_logit_probs_batch(batch_prompts) # 获取概率 Tensor
                 all_probs.append(probs)

            # 处理未获得任何概率的情况 (例如输入列表为空)
            if not all_probs:
                logging.warning("无防御预测返回了空的概率列表。")
                # 返回一个包含与输入文本数量相同的均匀分布概率的列表
                return [np.ones(self.num_labels) / self.num_labels] * len(texts)

            # 合并所有批次的概率 Tensor 并转换为 Numpy 数组
            final_probs_np = torch.cat(all_probs, dim=0).numpy()
            # 将大的 Numpy 数组转换回包含每个样本概率分布的列表
            return [p for p in final_probs_np]

        elif self.args.defense_method == 'ahp':
            # --- 应用 AHP 防御 ---
            return self._apply_ahp_defense(texts)
        elif self.args.defense_method == 'selfdenoise':
            # --- 应用 SelfDenoise 防御 ---
            return self._apply_selfdenoise_defense(texts)
        else:
            # 配置了未知的防御方法
            raise ValueError(f"未知的防御方法: {self.args.defense_method}")

    def __call__(self, text_input_list: List[str]) -> np.ndarray:
        """
        提供 TextAttack 的 ModelWrapper 所期望的接口。
        接收一个文本字符串列表，返回一个包含这些文本预测概率的 Numpy 数组。

        Args:
            text_input_list (List[str]): 输入的文本列表。

        Returns:
            np.ndarray: 形状为 [batch_size, num_labels] 的概率数组。
        """
        # 确保输入是列表格式
        if not isinstance(text_input_list, list):
            text_input_list = [text_input_list]

        # 处理空输入列表的情况
        if not text_input_list:
            logging.warning("TextAttack Wrapper 收到了空输入列表。")
            # 返回一个空的或者形状正确的零数组
            return np.zeros((0, self.num_labels))

        # 调用核心的 predict_batch 方法获取概率列表
        prob_list = self.predict_batch(text_input_list)

        # 检查 predict_batch 是否返回了预期的列表
        if not prob_list or not isinstance(prob_list, list):
            logging.error(f"predict_batch 未能在 __call__ 中返回有效的概率列表。返回类型: {type(prob_list)}。将返回默认值。")
            # Fallback: 返回一个批量的均匀分布概率
            return np.array([np.ones(self.num_labels) / self.num_labels] * len(text_input_list))

        # --- 健壮性检查：确保列表中的每个元素都是形状正确的 Numpy 数组 ---
        valid_probs = []
        expected_shape = (self.num_labels,)
        for i, p in enumerate(prob_list):
             # 检查是否为 Numpy 数组且形状是否匹配
             if isinstance(p, np.ndarray) and p.shape == expected_shape:
                 valid_probs.append(p)
             else:
                 # 如果形状不匹配或类型错误，记录错误并用均匀分布替换
                 logging.error(f"在 __call__ 中发现无效的概率数组（索引 {i}）。"
                               f"期望形状 {expected_shape}，实际类型 {type(p)}，形状 {getattr(p, 'shape', 'N/A')}。"
                               f"将替换为均匀分布。")
                 valid_probs.append(np.ones(self.num_labels) / self.num_labels)

        # 检查 valid_probs 是否为空 (如果所有输入都处理失败)
        if not valid_probs:
             logging.error("在 __call__ 中，所有样本的概率处理都失败了。")
             return np.zeros((len(text_input_list), self.num_labels)) # 返回零数组或均匀分布

        # 将包含有效概率数组的列表堆叠成一个大的 Numpy 数组 [batch_size, num_labels]
        try:
            return np.stack(valid_probs)
        except ValueError as e:
             # 如果堆叠失败 (例如，尽管进行了检查，形状仍然不一致)
             logging.error(f"在 __call__ 中堆叠概率时出错: {e}。有效概率数量: {len(valid_probs)}")
             # Fallback: 再次返回批量的均匀分布概率
             return np.array([np.ones(self.num_labels) / self.num_labels] * len(text_input_list))