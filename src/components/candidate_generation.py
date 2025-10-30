# src/components/candidate_generation.py
import torch
import logging
from transformers import PreTrainedModel, PreTrainedTokenizer
from typing import List, Union, TYPE_CHECKING
from tqdm.auto import tqdm

# 类型检查导入
if TYPE_CHECKING:
    from ..models.model_loader import AlpacaModel

class CandidateGenerator:
    """基于遮蔽文本生成候选句子。"""
    
    # 修改 __init__ 方法签名
    def __init__(self, model_wrapper: 'AlpacaModel', **kwargs):
        """
        初始化候选生成器。
        Args:
            model_wrapper (AlpacaModel): AlpacaModel 实例，提供 _format_prompt 和 _generate_batch 方法。
            **kwargs: 其他生成参数 (num_candidates 在 model_wrapper.args 中)。
        """
        self.model_wrapper = model_wrapper
        self.device = model_wrapper.device
        # 从 model_wrapper 的 args 中获取候选数量
        self.num_candidates = model_wrapper.args.ahp_num_candidates 
        self.mask_token = model_wrapper.args.mask_token
        
        # --- 预处理去噪指令模板 ---
        # (这部分逻辑从 model_loader._denoise_texts 复制过来)
        template = self.model_wrapper.denoise_instruction_template
        last_placeholder_idx = template.rfind('{}')
        if last_placeholder_idx == -1:
            raise ValueError("AHP CandidateGenerator: 去噪指令模板中未找到用于填充输入文本的 '{}' 占位符。")

        temp_marker = "__TEMP_INPUT_PLACEHOLDER__"
        template_with_marker = template[:last_placeholder_idx] + temp_marker + template[last_placeholder_idx+2:]
        # 将模板中用于示例的 {} 替换为真实的 mask_token
        instruction_base = template_with_marker.replace('{}', self.mask_token)
        # 将临时标记替换回 {}，这个 {} 将用于填充 masked_text
        self.final_instruction_template = instruction_base.replace(temp_marker, '{}')
        
        logging.info(f"候选生成器已初始化，将为每个输入生成 {self.num_candidates} 个候选。")

    @torch.no_grad() # 生成候选时不需要计算梯度
    def generate_candidates(self, masked_text: str) -> List[str]:
        """
        [已完善] 根据输入的遮蔽文本，使用 Alpaca 模型的去噪能力生成多个候选句子。
        """
        logging.debug(f"正在为 '{masked_text[:50]}...' 生成 {self.num_candidates} 个候选。")
        
        # 1. 构建去噪 Prompt
        # 将 masked_text 填入去噪指令模板的最后一个 {}
        prompt = self.model_wrapper._format_prompt(self.final_instruction_template.format(masked_text), "")
        
        # 2. 分词
        inputs = self.model_wrapper.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True,
                                             max_length=512).to(self.device) # 允许长输入
        
        # 3. 设置生成参数 (使用 Beam Search + Sampling 增加多样性)
        # 我们希望有多样性，同时保持质量
        num_beams_to_use = max(1, self.num_candidates) * 2 # Beam search 数量通常多于返回数量
        generation_config = {
            "max_new_tokens": int(self.model_wrapper.args.max_seq_length * 1.5), # 允许生成较长的回复
            "num_return_sequences": self.num_candidates, # 返回指定数量的候选
            "num_beams": num_beams_to_use, # 使用 Beam Search
            "do_sample": True, # 启用采样以增加多样性
            "temperature": 0.7,
            "top_k": 50,
            "top_p": 0.95,
            "pad_token_id": self.model_wrapper.tokenizer.eos_token_id
        }
        
        try:
            # 4. 调用原始模型的 generate 方法
            generate_ids = self.model_wrapper.model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                **generation_config
            )
            
            # 5. 解码
            input_len = inputs["input_ids"].shape[1]
            output_ids = generate_ids[:, input_len:]
            candidates = self.model_wrapper.tokenizer.batch_decode(
                output_ids, 
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=False
            )
            
            # 6. 后处理 (简单的清理，去除首尾空格)
            cleaned_candidates = [c.strip() for c in candidates if c.strip()]
            
            # 去除重复的候选
            unique_candidates = list(dict.fromkeys(cleaned_candidates))
            
            logging.debug(f"生成了 {len(unique_candidates)} 个唯一的候选。")
            
            if not unique_candidates:
                logging.warning("未能生成有效候选，返回原始遮蔽文本。")
                return [masked_text] # Fallback
                
            return unique_candidates

        except Exception as e:
            logging.error(f"生成候选时出错: {e}", exc_info=True)
            return [masked_text] # 出错时返回原始遮蔽文本作为 fallback