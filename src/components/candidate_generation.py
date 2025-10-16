# ahp_robustness/src/components/candidate_generation.py

import torch

def _build_denoising_prompt(masked_text):
    """
    (新函数) 构建一个用于去噪（填词）的指令Prompt。
    """
    # 使用Vicuna官方推荐的对话模板
    prompt = f"""A chat between a user and an artificial intelligence assistant.
USER: I have a sentence with masked words, marked as '[MASK]'. Your task is to fill in these masks to make the sentence coherent and natural. Only provide the completed sentence, without any explanations.

Sentence: "{masked_text}"
ASSISTANT:"""
    return prompt

def generate_candidates(masked_text, model, tokenizer, num_candidates=10, max_new_tokens=64):
    """
    (修正后) 从被遮蔽的文本生成多个候选句子。
    """
    prompt = _build_denoising_prompt(masked_text)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_return_sequences=num_candidates,
            num_beams=num_candidates,
            do_sample=True,
            top_p=0.9,
            temperature=0.7,
            early_stopping=True,
            pad_token_id=tokenizer.eos_token_id # 防止过早停止
        )

    # 解码并清理输出，只保留ASSISTANT:之后的部分
    full_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    
    candidates = []
    for text in full_outputs:
        # 找到ASSISTANT:之后的内容
        assistant_response = text.split("ASSISTANT:")[-1].strip()
        candidates.append(assistant_response)
        
    return candidates