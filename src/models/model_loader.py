# ahp_robustness/src/models/model_loader.py

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
from sentence_transformers import SentenceTransformer

def load_main_llm(model_name="tatsu-lab/alpaca-7b-wdiff", use_4bit=True):
    """
    加载主语言模型 (例如 Alpaca-7B)。

    Args:
        model_name (str): 要加载的Hugging Face模型名称。
        use_4bit (bool): 是否使用4-bit量化以节省显存。

    Returns:
        tuple: (model, tokenizer)
    """
    print(f"正在加载主模型: {model_name}...")
    
    # 根据是否使用4-bit量化来设置加载参数
    if use_4bit:
        model_kwargs = {
            "load_in_4bit": True,
            "torch_dtype": torch.float16,
            "device_map": "auto",
        }
    else:
        model_kwargs = {
            "torch_dtype": torch.float16,
            "device_map": "auto",
        }

    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print("主模型加载成功。")
        return model, tokenizer
    except Exception as e:
        print(f"加载主模型失败: {e}")
        print("请确保您已登录Hugging Face CLI并且有权限访问该模型。")
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

def load_nli_model(model_name="roberta-large-mnli"):
    """
    加载用于NLI剪枝的自然语言推理模型。

    Args:
        model_name (str): NLI模型的名称。

    Returns:
        tuple: (model, tokenizer)
    """
    print(f"正在加载NLI模型: {model_name}...")
    try:
        model = AutoModel.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print("NLI模型加载成功。")
        return model, tokenizer
    except Exception as e:
        print(f"加载NLI模型失败: {e}")
        return None, None