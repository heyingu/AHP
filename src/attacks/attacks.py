# ahp_robustness/src/attacks/attacks.py

import textattack
# ==================== 1. 导入新的攻击配方 ====================
from textattack.attack_recipes import (
    TextBuggerLi2018, 
    DeepWordBugGao2018, 
    PWWSRen2019, 
    BERTAttackLi2020,
    TextFoolerJin2019
)
# ==========================================================
from textattack import Attacker
from textattack.models.wrappers import PyTorchModelWrapper
from textattack.attack_results import SkippedAttackResult, FailedAttackResult
# 导入我们需要的替代约束
from textattack.constraints.semantics import WordEmbeddingDistance

import pandas as pd
from tqdm.auto import tqdm
import torch

# --- 攻击模型 (保持白盒攻击模式以备将来使用) ---
class ClassificationModelForAttack(torch.nn.Module):
    def __init__(self, predictor):
        super().__init__()
        self.model = predictor.model
        self.tokenizer = predictor.tokenizer
        self.predictor = predictor

    def forward(self, input_ids):
        # 允许梯度计算
        outputs = self.model(input_ids=input_ids)
        next_token_logits = outputs.logits[:, -1, :]
        
        p_id, n_id = self.predictor.positive_id, self.predictor.negative_id
        p_space_id, n_space_id = self.predictor.positive_id_with_space, self.predictor.negative_id_with_space

        prob_positive = next_token_logits[:, p_id]
        prob_negative = next_token_logits[:, n_id]
        prob_positive_space = next_token_logits[:, p_space_id]
        prob_negative_space = next_token_logits[:, n_space_id]

        final_prob_positive = torch.max(prob_positive, prob_positive_space)
        final_prob_negative = torch.max(prob_negative, prob_negative_space)
        
        return torch.stack([final_prob_negative, final_prob_positive], dim=1)


# --- 自定义的封装器 ---
class CustomPyTorchModelWrapper(PyTorchModelWrapper):
    def __call__(self, text_input_list):
        prompts = [self.model.predictor._get_instruction_template().format(review_text=text) for text in text_input_list]
        
        model_device = next(self.model.parameters()).device
        inputs = self.tokenizer(
            prompts,
            padding="longest",
            truncation=True,
            max_length=256,
            return_tensors="pt",
        )
        input_ids = inputs["input_ids"].to(model_device)
        
        outputs = self.model(input_ids)
        return outputs.cpu().detach()

# --- 攻击器主类 (已升级) ---
class AttackerWrapper:
    def __init__(self, predictor):
        attackable_model = ClassificationModelForAttack(predictor)
        self.model_wrapper = CustomPyTorchModelWrapper(attackable_model, predictor.tokenizer)

    def attack(self, dataset_for_attack, attack_recipe_name='textbugger'):
        # ==================== 2. 添加对 TextFooler 的支持 ====================
        if attack_recipe_name == 'textbugger':
            recipe_class = TextBuggerLi2018
        elif attack_recipe_name == 'deepwordbug':
            recipe_class = DeepWordBugGao2018
        elif attack_recipe_name == 'pwws':
            recipe_class = PWWSRen2019
        elif attack_recipe_name == 'bae':
            recipe_class = BERTAttackLi2020
        elif attack_recipe_name == 'textfooler': # <-- 新增分支
            recipe_class = TextFoolerJin2019
        # ===================================================================
        else:
            raise ValueError(f"未知的攻击方法: {attack_recipe_name}")

        # 构建并修正配方的逻辑保持不变
        recipe = recipe_class.build(self.model_wrapper)

        original_constraints_count = len(recipe.constraints)
        recipe.constraints = [
            c for c in recipe.constraints 
            if not isinstance(c, textattack.constraints.semantics.sentence_encoders.universal_sentence_encoder.UniversalSentenceEncoder)
        ]
        
        if len(recipe.constraints) < original_constraints_count:
            print(">>> 成功移除存在兼容性问题的 UniversalSentenceEncoder 约束。")
            recipe.constraints.append(WordEmbeddingDistance(min_cos_sim=0.8))
            print(">>> 已添加基于 PyTorch 的 WordEmbeddingDistance 约束作为替代。")

        # 创建并执行攻击
        dataset_tuples = [(item['sentence'], item['label']) for item in dataset_for_attack]
        attack_dataset = textattack.datasets.Dataset(dataset_tuples, shuffle=False)
        attacker = Attacker(recipe, attack_dataset)
        
        print(f"正在使用 {attack_recipe_name} 生成对抗样本...")
        results_iterable = attacker.attack_dataset()

        attacked_data = []
        for result in tqdm(results_iterable, total=len(dataset_for_attack)):
            original_text = result.original_text
            ground_truth_label = result.original_result.ground_truth_output
            if isinstance(result, (SkippedAttackResult, FailedAttackResult)):
                perturbed_text = original_text
            else:
                perturbed_text = result.perturbed_text
            attacked_data.append({
                "original_text": original_text,
                "perturbed_text": perturbed_text,
                "ground_truth_id": ground_truth_label,
            })
        return pd.DataFrame(attacked_data)