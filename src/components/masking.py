# ahp_robustness/src/components/masking.py

import random
import torch

def random_masking(text, tokenizer, mask_ratio=0.15):
    """
    对输入文本进行随机遮蔽。
    """
    mask_token = tokenizer.mask_token
    if mask_token is None:
        mask_token = "[MASK]"
    
    tokens = tokenizer.tokenize(text)
    if not tokens:
        return ""
        
    num_tokens = len(tokens)
    num_to_mask = int(num_tokens * mask_ratio)
    
    if num_to_mask == 0 and num_tokens > 0:
        num_to_mask = 1
        
    mask_indices = random.sample(range(num_tokens), min(num_to_mask, num_tokens))
    
    masked_tokens = tokens[:]
    for i in mask_indices:
        masked_tokens[i] = mask_token

    masked_text = tokenizer.convert_tokens_to_string(masked_tokens)
    return masked_text


def adversarial_masking(text, model, tokenizer, mask_ratio=0.15):
    """
    (修正后) 使用梯度信息进行对抗性遮蔽。
    """
    # 1. 准备输入并获取词嵌入
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(model.device)
    input_ids = inputs["input_ids"]
    
    model.zero_grad()
    word_embeddings = model.get_input_embeddings()
    input_embeds = word_embeddings(input_ids)
    
    # ===================== 修正开始 =====================
    # 移除了 .clone() 和 .requires_grad = True
    # 对于非叶子张量，我们使用 .retain_grad() 来保存它的梯度
    input_embeds.retain_grad()
    # ===================== 修正结束 =====================

    # 2. 前向传播计算损失 (使用语言模型自身的损失)
    outputs = model(inputs_embeds=input_embeds, labels=input_ids)
    loss = outputs.loss
    
    # 3. 反向传播计算梯度
    loss.backward()
    
    # 4. 从 .grad 属性获取梯度，并计算重要性分数
    # input_embeds.grad 此时应该已经被成功填充
    token_grads = input_embeds.grad.squeeze(0)
    token_importance_scores = torch.linalg.norm(token_grads, dim=-1)
    
    # 将PAD token的重要性设为0
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is not None:
        pad_indices = torch.where(input_ids.squeeze(0) == pad_token_id)
        if len(pad_indices[0]) > 0: # 确保存在pad token
            token_importance_scores[pad_indices] = -torch.inf # 设为负无穷，确保不会被选中

    # 5. 确定要遮蔽的词元
    num_tokens = (input_ids.squeeze(0) != pad_token_id).sum().item() if pad_token_id is not None else len(input_ids.squeeze(0))
    num_to_mask = int(num_tokens * mask_ratio)
    if num_to_mask == 0 and num_tokens > 0:
        num_to_mask = 1

    _, top_indices = torch.topk(token_importance_scores, k=min(num_to_mask, num_tokens))
    
    # 6. 生成遮蔽后的文本
    mask_token = tokenizer.mask_token if tokenizer.mask_token is not None else "[MASK]"
    # detach().cpu() 用于安全地将tensor转为list，避免占用GPU
    original_tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze(0).detach().cpu())
    
    masked_tokens = original_tokens[:]
    for idx in top_indices:
        masked_tokens[idx.item()] = mask_token
        
    masked_text = tokenizer.convert_tokens_to_string(masked_tokens)
    
    # 清理梯度和显存
    model.zero_grad()
    del input_embeds, outputs, loss, token_grads, token_importance_scores
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    return masked_text