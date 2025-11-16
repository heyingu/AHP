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