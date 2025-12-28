# src/components/candidate_generation.py
import torch
import logging
from transformers import PreTrainedModel, PreTrainedTokenizer
from typing import List, Union, TYPE_CHECKING
from tqdm.auto import tqdm


if TYPE_CHECKING:

    from ..models.model_loader import AlpacaModel
# 类型检查导入
if TYPE_CHECKING:
    from ..models.model_loader import AlpacaModel

class CandidateGenerator:
    """
    [已修改] 使用专用的 MaskedLM (RoBERTa) 生成候选。
    """
    def __init__(self, model_wrapper: 'AlpacaModel'):
        """
        初始化候选生成器。
        Args:
            model_wrapper (AlpacaModel): AlpacaModel 实例, 提供 _denoise_texts 方法。
        """
        
        self.model_wrapper = model_wrapper
        # --- 确保 RoBERTa 被加载 ---
        try:
            logging.info("AHP 候选生成器正在预加载 RoBERTa...")
            self.model_wrapper._load_roberta_denoiser()
            logging.info("候选生成器已初始化 (将使用 RoBERTa)。")
        except Exception as e:
            logging.error(f"CandidateGenerator 无法加载 RoBERTa: {e}", exc_info=True)
            raise e

    def generate_candidates_list(self, masked_list):
        all = []
        for t in masked_list:
            all.extend(self.generate_candidates(t))
        return all


    def generate_candidates(self, masked_text: str) -> List[str]:
        """
        [已修改] 使用 model_wrapper 中的 _denoise_texts 方法 (及 RoBERTa) 
        来生成 K 个候选。
        """
        # if dataset == 'sst2':
        #     num_candidates = self.model_wrapper.args.ahp_num_candidates 
        # else:
        #     num_candidates = 1
        
        num_candidates = self.model_wrapper.args.ahp_num_candidates 
        # print(num_candidates)
        # --- 关键修改 ---
        # 我们不再使用 Alpaca (CausalLM) 来进行 in-filling。
        # 我们调用 _denoise_texts，强制使用 'roberta'。
        
        # 1. 创建一个包含 N 个相同 masked_text 的列表
        masked_text_list = [masked_text] * num_candidates
        
        # 2. 调用 RoBERTa denoiser
        # (因为我们修改了 _denoise_texts 
        # 使其具有采样功能, 每次调用都会产生不同的结果)
        logging.debug(f"正在调用 RoBERTa denoiser 为 AHP 生成 {num_candidates} 个候选...")
        
        candidates = self.model_wrapper._denoise_texts(
            masked_text_list, 
            denoiser_type='roberta' # <--- 强制使用 RoBERTa
        )
        
        if not candidates:
             logging.warning("RoBERTa denoiser 未返回任何候选。")
             return []
             
        # (可选：去重，以防采样到相同结果)
        unique_candidates = list(dict.fromkeys(candidates))
        logging.debug(f"RoBERTa 生成了 {len(candidates)} 个候选, 其中 {len(unique_candidates)} 个是唯一的。")
        
        return unique_candidates

    def generate_one_per_mask(self, masked_texts: List[str]) -> List[str]:
        """
        [优化补丁] 针对 AHP Stochastic/Random 模式的批量 1-to-1 生成。
        
        背景：原本的设计是"对每个 masked_text 调 generate_candidates，生成 K 个但只取第一个"。
             这导致大量浪费的 RoBERTa 推理。
        
        优化：直接对整个 masked_texts 列表批量进行 RoBERTa 推理，1 对 1 填充。
        
        Args:
            masked_texts: 一批掩码文本，来自梯度采样或随机采样。
        
        Returns:
            等长的候选文本列表，每个 masked_text 对应一个填充结果。
        """
        if not masked_texts:
            logging.warning("generate_one_per_mask 收到空列表")
            return []
        
        logging.debug(
            f"[Optimized] 批量 1-to-1 去噪: {len(masked_texts)} 个 masks → RoBERTa 推理"
        )
        
        try:
            # 直接利用 _denoise_texts 的底层批处理能力
            # RoBERTa 会在内部进行批量推理，避免了之前的重复采样
            candidates = self.model_wrapper._denoise_texts(
                masked_texts,
                denoiser_type='roberta'
            )
            
            # 简单清洗：去掉前后空格
            cleaned = [c.strip() for c in candidates if isinstance(c, str)]
            
            if len(cleaned) != len(masked_texts):
                logging.warning(
                    f"RoBERTa 去噪后候选数量不一致: 输入 {len(masked_texts)}, 输出 {len(cleaned)}"
                )
            
            return cleaned
        
        except Exception as e:
            logging.error(f"generate_one_per_mask 异常: {e}", exc_info=True)
            # 降级处理：返回原始掩码文本
            return [t.strip() for t in masked_texts]
