# ahp_robustness/src/models/model_loader.py
# (在拥有48GB或更大显存的GPU上，应使用此简化版本)

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification, BitsAndBytesConfig
import os

def load_main_llm(model_name, use_4bit=True):
    """
    加载主 LLM 模型和分词器 (大显存简化版)。
    """
    print(f"正在加载主模型: {model_name}...")
    try:
        # 在大显存环境下，"auto" 会自动将整个模型放在GPU上，无需任何复杂配置。
        device_map = "auto"
        
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        if use_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map=device_map,
                quantization_config=quantization_config,
                trust_remote_code=True,
                low_cpu_mem_usage=True # 加速从磁盘加载
            )
            print("主模型以 4-bit 量化模式成功加载到GPU。")
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map=device_map,
                torch_dtype=torch.float16,
                trust_remote_code=True,
                low_cpu_mem_usage=True # 加速从磁盘加载
            )
            print("主模型以半精度 (float16) 模式成功加载到GPU。")
        
        return model, tokenizer
    except Exception as e:
        print(f"加载主模型失败: {e}")
        return None, None

def load_nli_model():
    """
    加载用于剪枝的 NLI 模型。
    """
    nli_model_name = "roberta-large-mnli"
    print(f"正在加载NLI模型: {nli_model_name}...")
    try:
        nli_tokenizer = AutoTokenizer.from_pretrained(nli_model_name)
        nli_model = AutoModelForSequenceClassification.from_pretrained(nli_model_name)
        # 将NLI模型也放到GPU上以获得最佳性能
        nli_model.to("cuda" if torch.cuda.is_available() else "cpu")
        print("NLI模型加载成功。")
        return nli_model, nli_tokenizer
    except Exception as e:
        print(f"加载NLI模型失败: {e}")
        return None, None

def load_pruning_model(model_name="sentence-transformers/all-mpnet-base-v2"):
    """
    加载用于快速剪枝的轻量级模型。
    这通常是一个句子嵌入模型。

    Args:
        model_name (str): sentence-transformers模型的名称。

    Returns:
        SentenceTransformer: 加载好的句子转换器模型。
    """
    print(f"正在加载剪枝模型: {model_name}...")
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = SentenceTransformer(model_name, device=device)
        print("剪枝模型加载成功。")
        return model
    except Exception as e:
        print(f"加载剪枝模型失败: {e}")
        return None
# import torch
# from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
# from sentence_transformers import SentenceTransformer

# def load_main_llm(model_name="tatsu-lab/alpaca-7b-wdiff", use_4bit=True):
#     """
#     加载主语言模型 (例如 Alpaca-7B)。

#     Args:
#         model_name (str): 要加载的Hugging Face模型名称。
#         use_4bit (bool): 是否使用4-bit量化以节省显存。

#     Returns:
#         tuple: (model, tokenizer)
#     """
#     print(f"正在加载主模型: {model_name}...")
    
#     # 根据是否使用4-bit量化来设置加载参数
#     if use_4bit:
#         model_kwargs = {
#             "load_in_4bit": True,
#             "torch_dtype": torch.float16,
#             "device_map": "auto",
#         }
#     else:
#         model_kwargs = {
#             "torch_dtype": torch.float16,
#             "device_map": "auto",
#         }

#     try:
#         model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
#         tokenizer = AutoTokenizer.from_pretrained(model_name)
#         print("主模型加载成功。")
#         return model, tokenizer
#     except Exception as e:
#         print(f"加载主模型失败: {e}")
#         print("请确保您已登录Hugging Face CLI并且有权限访问该模型。")
#         return None, None

# def load_pruning_model(model_name="sentence-transformers/all-mpnet-base-v2"):
#     """
#     加载用于快速剪枝的轻量级模型。
#     这通常是一个句子嵌入模型。

#     Args:
#         model_name (str): sentence-transformers模型的名称。

#     Returns:
#         SentenceTransformer: 加载好的句子转换器模型。
#     """
#     print(f"正在加载剪枝模型: {model_name}...")
#     try:
#         device = "cuda" if torch.cuda.is_available() else "cpu"
#         model = SentenceTransformer(model_name, device=device)
#         print("剪枝模型加载成功。")
#         return model
#     except Exception as e:
#         print(f"加载剪枝模型失败: {e}")
#         return None

# def load_nli_model(model_name="roberta-large-mnli"):
#     """
#     加载用于NLI剪枝的自然语言推理模型。

#     Args:
#         model_name (str): NLI模型的名称。

#     Returns:
#         tuple: (model, tokenizer)
#     """
#     print(f"正在加载NLI模型: {model_name}...")
#     try:
#         model = AutoModel.from_pretrained(model_name)
#         tokenizer = AutoTokenizer.from_pretrained(model_name)
#         print("NLI模型加载成功。")
#         return model, tokenizer
#     except Exception as e:
#         print(f"加载NLI模型失败: {e}")
#         return None, None