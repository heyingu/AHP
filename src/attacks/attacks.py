# ahp_robustness/src/attacks/attacks.py

import textattack
from textattack.attack_recipes import TextBuggerLi2018, DeepWordBugGao2018
from textattack import Attacker
from textattack.models.wrappers import PyTorchModelWrapper
from textattack.attack_results import SkippedAttackResult, FailedAttackResult
import pandas as pd
from tqdm.auto import tqdm
import torch

# --- 攻击模型 (强制黑盒) ---
class ClassificationModelForAttack(torch.nn.Module):
    def __init__(self, predictor):
        super().__init__()
        self.model = predictor.model
        self.tokenizer = predictor.tokenizer
        self.predictor = predictor

    def forward(self, input_ids):
        # --- 关键修正：使用 torch.no_grad() 主动切断梯度 ---
        # 这会强制 TextAttack 切换到不需要梯度的黑盒攻击模式，从而避免显存爆炸
        with torch.no_grad():
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
            
            # 返回的logits不带梯度
            return torch.stack([final_prob_negative, final_prob_positive], dim=1)


# --- 自定义的封装器 (无改动，但其内部调用的模型已改变) ---
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
        
        # 这里调用的 self.model 的 forward 方法已经被 no_grad 包裹
        outputs = self.model(input_ids)
        # 因为已经没有梯度，所以不再需要 .detach()
        return outputs.cpu()

# --- 攻击器主类 (无改动) ---
class AttackerWrapper:
    def __init__(self, predictor):
        attackable_model = ClassificationModelForAttack(predictor)
        self.model_wrapper = CustomPyTorchModelWrapper(attackable_model, predictor.tokenizer)

    def attack(self, dataset_for_attack, attack_recipe_name='textbugger'):
        if attack_recipe_name == 'textbugger':
            recipe = TextBuggerLi2018.build(self.model_wrapper)
        elif attack_recipe_name == 'deepwordbug':
            recipe = DeepWordBugGao2018.build(self.model_wrapper)
        else:
            raise ValueError(f"未知的攻击方法: {attack_recipe_name}")

        dataset_tuples = [(item['sentence'], item['label']) for item in dataset_for_attack]
        attack_dataset = textattack.datasets.Dataset(dataset_tuples, shuffle=False)
        attacker = Attacker(recipe, attack_dataset)
        results_iterable = attacker.attack_dataset()

        attacked_data = []
        print(f"正在使用 {attack_recipe_name} 生成对抗样本...")
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