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

# --- 导入 AHP 组件 (请根据您的项目结构调整路径) ---
try:
    # 假设 args_config 在 src 目录下
    from ..args_config import AHPSettings
    # 假设 masking, candidate_generation, result_aggregation 在 src/components/ 目录下
    from ..components.masking import AdversarialMasker, RandomMasker
    from ..components.candidate_generation import CandidateGenerator
    from ..components.result_aggregation import aggregate_results # 导入您的聚合函数
    # 假设 base_pruner 和具体的剪枝器都在 src/pruning/ 目录下
    from ..pruning.base_pruner import BasePruner
    from ..pruning import PerplexityPruner, SemanticPruner, NLIPruner, ClusteringPruner
except ImportError as e:
     # 如果导入失败，记录错误并提示用户检查路径
     logging.error(f"无法导入 AHP 组件，请检查 model_loader.py 中的导入路径: {e}")
     # 定义临时的占位符类/函数，使得代码在缺少组件时也能运行（但功能不完整）
     class BasePruner: pass
     class AdversarialMasker: pass
     class CandidateGenerator: pass
     def aggregate_results(*args, **kwargs): return np.array([0.5, 0.5]) # 返回一个默认的均匀分布概率
     logging.warning("由于导入错误，AHP 组件将使用占位符。AHP 防御将无法正常工作。")


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
        "classification": "请判断输入英文句子的情感是积极的还是消极的。(Given an English sentence input, determine its sentiment as positive or negative.)", # 分类指令
        # 去噪指令，包含示例 (few-shot prompting)
        "denoise_explicit": """请将输入句子中每个遮蔽词 {} 替换为合适的词语。输出的句子应自然、连贯，且长度与输入句子相同。请直接给出答案。

### Input:
a {} , funny and {} transporting re-imagining {} {} and the beast and 1930s {} films

### Response:
a stirring , funny and finally transporting re-imagining of beauty and the beast and 1930s horror films

### Input:
{}""",
        "label_map": {"negative": 0, "positive": 1}, # 标签名称到 ID 的映射
        # Alpaca-7B 对于 'Negative' 和 'Positive' 的 token ID (需要根据实际模型验证)
        "label_tokens": [29940, 9135] # 例如：' Negative' ' Positive' 的 ID
    },
    "agnews": {
        # AG News 的分类指令，包含示例 (few-shot prompting)
        "classification": """请根据新闻文章的标题和描述，将其分类到以下四个类别之一：世界 (World)、体育 (Sports)、商业 (Business) 或 科技 (Science/Technology)。请直接返回类别名称作为答案。

### Input:
Title: Wall St. Bears Claw Back Into the Black (Reuters)
Description: Reuters - Short-sellers, Wall Street's dwindling band of ultra-cynics, are seeing green again.

### Response:
Business

### Input:
{}""", # 真实输入的占位符
        # AG News 的去噪指令
       "denoise_explicit": """请将输入句子中每个被遮蔽的位置 \"{}\" 替换为合适的词语，使其自然且连贯。每个 \"{}\" 只能替换为一个词。返回的句子应与给定句子的长度相同。请直接给出答案。

### Input:
{}""",
        "label_map": {"World": 0, "Sports": 1, "Business": 2, "Technology": 3}, # 标签名称到 ID 的映射
        # Alpaca-7B 对于 'World', 'Sports', 'Business', 'Technology' 的 token ID (需要根据实际模型验证)
        "label_tokens": [14058, 29903, 16890, 7141] # 示例 ID
    }
}


class AlpacaModel:
    """封装 Alpaca 模型加载、预测以及集成 AHP 和 SelfDenoise 防御逻辑的类。"""
    def __init__(self, args: AHPSettings):
        """
        初始化 AlpacaModel。

        Args:
            args (AHPSettings): 包含所有配置参数的对象。
        """
        self.args = args
        self.device = torch.device(args.device) # 设置计算设备 (cuda 或 cpu)
        self.tokenizer: Optional[transformers.PreTrainedTokenizer] = None # 分词器
        self.model: Optional[transformers.PreTrainedModel] = None # Alpaca 模型
        self.roberta_tokenizer: Optional[RobertaTokenizer] = None # RoBERTa 分词器 (用于 SelfDenoise 去噪)
        self.roberta_model: Optional[RobertaForMaskedLM] = None # RoBERTa 模型 (用于 SelfDenoise 去噪)
        self._load_model() # 加载主模型

        # --- AHP 和 SelfDenoise 组件初始化 (惰性加载或在使用时加载) ---
        self.adversarial_masker: Optional[AdversarialMasker] = None # 对抗性遮蔽器
        self.random_masker: Optional[RandomMasker] = None # 随机遮蔽器
        self.candidate_generator: Optional[CandidateGenerator] = None # 候选生成器
        self.pruner: Optional[BasePruner] = None # 剪枝器 (具体类型在 AHP 流程中确定)

        # --- 根据数据集名称设置模型的特定指令和标签信息 ---
        self.set_dataset_mode(args.dataset_name)

        # --- 在初始化时就准备好可能需要的遮蔽器 ---
        self._initialize_maskers()

    def _initialize_maskers(self):
        """根据配置的防御方法，初始化所需的遮蔽器。"""
        # 如果防御方法是 AHP，且对抗性遮蔽器尚未初始化
        if self.args.defense_method == 'ahp' and self.adversarial_masker is None:
             try:
                 # 实例化对抗性遮蔽器 (您需要确保 AdversarialMasker 类已正确实现)
                 # 它通常需要模型、分词器和设备信息来计算词语重要性
                 self.adversarial_masker = AdversarialMasker(self.model, self.tokenizer, device=self.device)
                 logging.info("已初始化 AHP 所需的对抗性遮蔽器。")
             except NameError: # 如果 AdversarialMasker 类未定义或导入失败
                 logging.error("AdversarialMasker 类未找到或未实现。")
                 raise RuntimeError("AHP 防御需要 AdversarialMasker，但它不可用。")
             except Exception as e: # 捕获其他可能的初始化错误
                 logging.error(f"初始化 AdversarialMasker 时出错: {e}")
                 raise e

        # 如果防御方法是 SelfDenoise，且随机遮蔽器尚未初始化
        if self.args.defense_method == 'selfdenoise' and self.random_masker is None:
            # 实例化随机遮蔽器
            self.random_masker = RandomMasker(self.tokenizer, mask_token=self.args.mask_token, mask_rate=self.args.mask_rate)
            logging.info("已初始化 SelfDenoise 所需的随机遮蔽器。")

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
            roberta_path = "roberta-large" # 使用 Hugging Face Hub 上的标准 RoBERTa 模型
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
        """
        根据配置参数，按需初始化并返回 AHP 剪枝器实例。
        如果已初始化且类型匹配，则返回缓存的实例。
        """
        method = self.args.ahp_pruning_method
        threshold = self.args.ahp_pruning_threshold # 注意：这个参数的含义取决于剪枝方法

        # 如果剪枝器已存在，并且其类型与当前配置的方法匹配，则直接返回
        # (通过类名的小写前缀来简单判断类型是否匹配)
        if self.pruner is not None and self.pruner.__class__.__name__.lower().startswith(method):
             return self.pruner

        logging.info(f"正在初始化剪枝器: {method}，参数/阈值: {threshold}")
        try:
            # 根据方法名称选择并初始化对应的剪枝器类
            # 您需要确保这些剪枝器类已正确实现并可以导入
            if method == 'perplexity':
                # 困惑度剪枝器，需要模型、分词器和阈值
                self.pruner = PerplexityPruner(self.model, self.tokenizer, threshold=threshold, device=self.device)
            elif method == 'semantic':
                # 语义相似度剪枝器，需要一个嵌入模型 (可能在类内部加载) 和阈值
                self.pruner = SemanticPruner(threshold=threshold, device=self.device)
            elif method == 'nli':
                 # NLI (自然语言推断) 剪枝器，需要一个 NLI 模型和阈值
                self.pruner = NLIPruner(threshold=threshold, device=self.device)
            elif method == 'clustering':
                 # 聚类剪枝器，需要嵌入模型，并且参数通常是簇的数量 (n_clusters)
                 # 我们将 threshold 强制转换为整数作为簇数
                 self.pruner = ClusteringPruner(n_clusters=int(threshold), device=self.device)
            elif method == 'none':
                # 不使用剪枝
                self.pruner = None
            else:
                # 配置了未知的剪枝方法
                raise ValueError(f"未知的剪枝方法: {method}")
            # 返回新创建或已缓存的剪枝器实例
            return self.pruner
        except NameError as e: # 如果对应的剪枝器类未定义或导入失败
             logging.error(f"方法 '{method}' 对应的剪枝器类未找到或未实现: {e}")
             raise RuntimeError(f"需要剪枝器 '{method}' 但其不可用。")
        except Exception as e: # 捕获其他初始化错误
             logging.error(f"初始化剪枝器 '{method}' 时出错: {e}")
             raise e


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
            **inputs, # 将处理好的输入传递给模型
            max_new_tokens=max_new_tokens, # 控制生成文本的最大长度
            do_sample=False, # 使用贪婪解码 (Greedy Decoding)，保证结果一致性，除非需要探索性生成
            pad_token_id=self.tokenizer.eos_token_id # 重要：指定 pad_token_id，防止生成意外停止
            # 可以根据需要从 self.args 添加其他生成参数，例如：
            # temperature=self.args.temperature,
            # top_p=self.args.top_p,
            # num_beams=self.args.num_beams,
            # repetition_penalty=self.args.repetition_penalty,
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
        """
        应用完整的 AHP (对抗性层次处理) 防御流程。

        Args:
            texts (List[str]): 需要进行防御和预测的原始文本列表。

        Returns:
            List[np.ndarray]: 每个输入文本对应的最终聚合概率分布列表。
        """
        logging.info("正在应用 AHP 防御...")
        final_aggregated_probs = [] # 存储每个文本最终的聚合概率

        # --- 初始化 AHP 组件 (如果尚未完成) ---
        # 确保遮蔽器已初始化
        if self.adversarial_masker is None:
             self._initialize_maskers()
             if self.adversarial_masker is None:
                 raise RuntimeError("对抗性遮蔽器未能初始化，无法执行 AHP 防御。")
        # 确保候选生成器已初始化
        if self.candidate_generator is None:
            try:
                # 假设 CandidateGenerator 需要模型和分词器
                self.candidate_generator = CandidateGenerator(self.model, self.tokenizer, num_candidates=self.args.ahp_num_candidates, device=self.device)
                logging.info("已初始化候选生成器。")
            except NameError:
                 logging.error("CandidateGenerator 类未找到或未实现。")
                 raise RuntimeError("AHP 防御需要 CandidateGenerator，但它不可用。")
            except Exception as e:
                 logging.error(f"初始化 CandidateGenerator 时出错: {e}")
                 raise e

        # 获取剪枝器实例 (如果配置了剪枝方法，这里会惰性初始化)
        current_pruner = self._get_pruner()

        # --- 逐个处理输入文本 ---
        # 使用 tqdm 显示 AHP 防御的进度
        for text in tqdm(texts, desc="AHP 防御流程", leave=False, ncols=100):
            try:
                # --- 1. 对抗性遮蔽 ---
                # 调用对抗性遮蔽器的 mask_input 方法
                # 假设它返回遮蔽后的文本 (masked_text) 和被遮蔽词的索引 (masked_indices)
                masked_text, masked_indices = self.adversarial_masker.mask_input(text, self.args.mask_rate)
                logging.debug(f"AHP 遮蔽后文本 (前 50 字符): {masked_text[:50]}...")
                logging.debug(f"AHP 遮蔽索引: {masked_indices}")

                # --- 2. 候选生成 ---
                # 基于遮蔽后的文本生成多个候选句子
                # 假设 generate_candidates 返回一个字符串列表
                candidates = self.candidate_generator.generate_candidates(masked_text)
                logging.debug(f"为 '{text[:20]}...' 生成了 {len(candidates)} 个候选。")

                # 处理未能生成候选的情况
                if not candidates:
                    logging.warning(f"未能为文本 '{text[:50]}...' 生成任何候选。将尝试直接预测遮蔽文本。")
                    # Fallback: 直接预测遮蔽后的文本
                    candidate_prompts = [self._format_prompt(self.classification_instruction, masked_text)]
                    probs_tensor = self._get_logit_probs_batch(candidate_prompts)
                    all_candidate_probs = probs_tensor.numpy() # [1, num_labels]
                else:
                    # --- 3. 剪枝 ---
                    if current_pruner:
                        # 调用剪枝器的 prune 方法，传入原始文本、候选列表和遮蔽文本作为上下文
                        pruned_candidates = current_pruner.prune(original_text=text, candidates=candidates, masked_text=masked_text)
                        logging.debug(f"使用 {self.args.ahp_pruning_method} 剪枝后剩余 {len(pruned_candidates)} 个候选。")
                        # 处理剪枝后没有候选剩余的情况
                        if not pruned_candidates:
                             logging.warning("剪枝操作移除了所有候选。将使用剪枝前的所有候选进行预测。")
                             pruned_candidates = candidates # Fallback: 使用所有未剪枝的候选
                    else:
                        # 如果没有配置剪枝方法
                        pruned_candidates = candidates
                        logging.debug("未应用剪枝。")

                    # --- 4. 对剪枝后的候选进行预测 (分批处理) ---
                    # 为每个候选构建分类 Prompt
                    candidate_prompts = [self._format_prompt(self.classification_instruction, cand) for cand in pruned_candidates]
                    candidate_probs_list = [] # 存储每个批次的预测概率
                    # 按照配置的模型批次大小进行分批预测
                    for i in range(0, len(candidate_prompts), self.args.model_batch_size):
                        batch_prompts = candidate_prompts[i : i + self.args.model_batch_size]
                        # 获取这批候选的预测概率 (Tensor)
                        probs_tensor = self._get_logit_probs_batch(batch_prompts)
                        candidate_probs_list.append(probs_tensor)

                    # 检查是否成功获取了概率
                    if not candidate_probs_list:
                         logging.error(f"对候选进行预测失败: '{text[:50]}...'")
                         # Fallback: 返回均匀分布概率
                         all_candidate_probs = np.array([np.ones(self.num_labels) / self.num_labels])
                    else:
                         # 将所有批次的概率 Tensor 合并为一个大的 Numpy 数组 [num_candidates, num_labels]
                         all_candidate_probs = torch.cat(candidate_probs_list, dim=0).numpy()

                # --- 5. 结果聚合 ---
                # 调用您实现的 aggregate_results 函数进行聚合
                # 传入所有候选的概率分布和配置的聚合策略
                aggregated_prob = aggregate_results(all_candidate_probs, strategy=self.args.ahp_aggregation_strategy)
                final_aggregated_probs.append(aggregated_prob) # 将聚合结果添加到最终列表中
                logging.debug(f"聚合后概率: {aggregated_prob}")

            except Exception as e:
                # 捕获在处理单个文本时可能发生的任何错误
                logging.error(f"处理文本 '{text[:50]}...' 时 AHP 防御出错: {e}", exc_info=True) # exc_info=True 会记录详细的错误堆栈
                # Fallback: 为出错的样本返回均匀分布概率
                uniform_prob = np.ones(self.num_labels) / self.num_labels
                final_aggregated_probs.append(uniform_prob)

            # --- 可选：显存清理 ---
            # 如果在处理大量文本时遇到显存不足 (OOM) 问题，可以尝试定期清理缓存
            if i % 10 == 0: # 例如每处理 10 个文本清理一次
                if torch.cuda.is_available():
                    torch.cuda.empty_cache() # 清理未被引用的 CUDA 缓存
                    gc.collect() # 执行 Python 的垃圾回收

        logging.info("AHP 防御应用完成。")
        return final_aggregated_probs


    def _denoise_texts(self, masked_texts: List[str], denoiser_type: str) -> List[str]:
        """
        使用 Alpaca 或 RoBERTa 对一批被遮蔽的文本进行去噪（恢复）。

        Args:
            masked_texts (List[str]): 包含遮蔽标记的文本列表。
            denoiser_type (str): 使用的去噪器类型 ('alpaca' 或 'roberta')。

        Returns:
            List[str]: 去噪（恢复）后的文本列表。
        """
        denoised_texts = []
        if denoiser_type == 'alpaca':
            # 使用 Alpaca 进行去噪
            logging.debug(f"使用 Alpaca 去噪 {len(masked_texts)} 个文本...")
            # 准备去噪指令，将遮蔽标记填入模板
            # 注意：这里的模板需要三个 {} 占位符，前两个用于指令中的示例，最后一个用于实际输入
            denoise_instruction = self.denoise_instruction_template.format(
                self.args.mask_token, self.args.mask_token, '{}'
            )
            # 构建 Prompt：将被遮蔽的文本填入去噪指令的最后一个占位符
            # 使用无输入的模板格式化，因为输入文本已整合到指令中
            prompts = [self._format_prompt(denoise_instruction.format(mt), "") for mt in masked_texts]

            # 分批生成去噪后的文本
            for i in tqdm(range(0, len(prompts), self.args.model_batch_size), desc="去噪 (Alpaca)", leave=False, ncols=100):
                 batch_prompts = prompts[i:i + self.args.model_batch_size]
                 # 去噪时允许生成更长的文本，以防原始文本较长或模型添加额外内容
                 gen_max_tokens = int(self.args.max_seq_length * 1.5) # 设定一个合理的上限
                 responses = self._generate_batch(batch_prompts, max_new_tokens=gen_max_tokens)
                 denoised_texts.extend(responses)

        elif denoiser_type == 'roberta':
            # 使用 RoBERTa 进行去噪
            logging.debug(f"使用 RoBERTa 去噪 {len(masked_texts)} 个文本...")
            self._load_roberta_denoiser() # 确保 RoBERTa 模型已加载

            # 获取 RoBERTa 分词器使用的标准遮蔽标记 ID
            roberta_mask_token_id = self.roberta_tokenizer.mask_token_id
            # 获取我们自定义的遮蔽标记在 RoBERTa 分词器中的 ID (如果已添加)
            custom_mask_token_id = self.roberta_tokenizer.convert_tokens_to_ids(self.args.mask_token)

            outputs = [] # 存储去噪后的文本
            # 分批处理
            for i in tqdm(range(0, len(masked_texts), self.args.model_batch_size), desc="去噪 (RoBERTa)", leave=False, ncols=100):
                batch_texts = masked_texts[i:i+self.args.model_batch_size]

                # --- 输入处理 ---
                # 为了让 RoBERTa 更好地理解遮蔽位置，通常需要将输入文本中的自定义 mask_token
                # 替换为 RoBERTa 自身训练时使用的 mask_token (例如 '<mask>')。
                # 当然，如果 RoBERTa 被微调过以识别我们的自定义 mask_token，则可能不需要替换。
                # 为了通用性，这里进行替换：
                roberta_input_texts = [t.replace(self.args.mask_token, self.roberta_tokenizer.mask_token) for t in batch_texts]

                # 使用 RoBERTa 分词器处理输入
                inputs = self.roberta_tokenizer(roberta_input_texts, return_tensors="pt", padding=True, truncation=True, max_length=self.args.max_seq_length)
                inputs = {k: v.to(self.device) for k, v in inputs.items()} # 移动到设备

                # --- 模型预测 ---
                with torch.no_grad(): # 去噪不需要梯度
                    logits = self.roberta_model(**inputs).logits # 获取所有位置的 logits

                # --- 填充遮蔽位置 ---
                # 找到输入中 RoBERTa 遮蔽标记 '<mask>' 的位置
                mask_token_indices = (inputs['input_ids'] == roberta_mask_token_id)
                # 复制原始输入 ID，我们只修改被遮蔽的部分
                predicted_token_ids = inputs['input_ids'].clone()

                # 如果存在遮蔽标记
                if torch.any(mask_token_indices):
                     # 在这些遮蔽位置上，用模型预测的最可能的 token ID 替换掉 '<mask>' ID
                     predicted_token_ids[mask_token_indices] = logits[mask_token_indices].argmax(axis=-1)

                # --- 解码 ---
                # 将填充后的 token ID 序列解码回文本
                batch_outputs = self.roberta_tokenizer.batch_decode(predicted_token_ids, skip_special_tokens=True)
                outputs.extend(batch_outputs)
            denoised_texts = outputs

        else:
             # 配置了未知的去噪器类型
             raise ValueError(f"未知的去噪器类型: {denoiser_type}")

        # --- 后处理 ---
        # 对去噪后的文本进行清理，例如去除首尾多余空格
        # skip_special_tokens=True 已经处理了大部分特殊标记
        cleaned_texts = [t.strip() for t in denoised_texts]
        # 打印一个去噪示例，方便调试检查
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
        for text in tqdm(texts, desc="SelfDenoise 防御流程", leave=False, ncols=100):
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
            if i % 10 == 0:
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