# ahp_robustness/src/attacks.py

import textattack
from textattack.attack_recipes import TextBuggerLi2018, DeepWordBugGao2018
from textattack import Attacker
from textattack.models.wrappers import PyTorchModelWrapper
from textattack.attack_results import SkippedAttackResult
import pandas as pd
from tqdm.auto import tqdm
import torch

# --- 自定义的分类模型 ---
class ClassificationModelForAttack(torch.nn.Module):
    def __init__(self, base_model, num_labels, tokenizer):
        super().__init__()
        self.model = base_model
        self.tokenizer = tokenizer
        self.classifier = torch.nn.Linear(base_model.config.hidden_size, num_labels)

        base_model_dtype = next(base_model.parameters()).dtype
        self.classifier.to(base_model_dtype)

        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def forward(self, input_ids):
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()
        
        self.model.eval()
        
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        last_hidden_state = outputs.hidden_states[-1]
        
        # ===================== 终极修正：使用平均池化 =====================
        # 我们不再只取最后一个token，而是计算所有有效token隐藏状态的平均值
        
        # 1. 为了正确计算平均值，需要将padding位置的hidden_state置为0
        #    需要扩展attention_mask以匹配hidden_state的维度
        expanded_mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).to(last_hidden_state.dtype)
        masked_hidden_state = last_hidden_state * expanded_mask
        
        # 2. 对所有token的向量求和，然后除以有效token的数量
        sum_hidden_state = torch.sum(masked_hidden_state, dim=1)
        sum_mask = expanded_mask.sum(1)
        # 防止除以零
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        
        # 3. 得到句子的“平均思想”向量
        mean_pooled_output = sum_hidden_state / sum_mask
        # =============================================================

        # 4. 将这个更可靠的“思想”向量送入分类头
        logits = self.classifier(mean_pooled_output)
        return logits

# --- 自定义的封装器 (无需改动) ---
class CustomPyTorchModelWrapper(PyTorchModelWrapper):
    def __call__(self, text_input_list):
        model_device = next(self.model.parameters()).device
        
        inputs = self.tokenizer(
            text_input_list,
            padding="longest",
            truncation=True,
            max_length=256,
            return_tensors="pt",
        )
        
        input_ids = inputs["input_ids"].to(model_device)
        
        with torch.no_grad():
            outputs = self.model(input_ids)

        return outputs.cpu()

# --- 攻击器主类 (使用我们自己的封装器，无需改动) ---
class AttackerWrapper:
    def __init__(self, predictor):
        num_labels = 2 if predictor.task == 'sst2' else 4
        
        attackable_model = ClassificationModelForAttack(predictor.model, num_labels, predictor.tokenizer)
        
        self.model_wrapper = CustomPyTorchModelWrapper(attackable_model, predictor.tokenizer)

    def attack(self, dataset_for_attack, attack_recipe_name='textbugger'):
        if attack_recipe_name == 'textbugger':
            recipe = TextBuggerLi2018.build(self.model_wrapper)
        elif attack_recipe_name == 'deepwordbug':
            recipe = DeepWordBugGao2018.build(self.model_wrapper)
        else:
            raise ValueError(f"未知的攻击方法: {attack_recipe_name}")

        dataset_tuples = []
        for item in dataset_for_attack:
            text = item['sentence'] if 'sentence' in item else item['text']
            label = item['label']
            dataset_tuples.append((text, label))

        attack_dataset = textattack.datasets.Dataset(dataset_tuples, shuffle=False)
        attacker = Attacker(recipe, attack_dataset)
        results_iterable = attacker.attack_dataset()

        attacked_data = []
        print(f"正在使用 {attack_recipe_name} 生成对抗样本...")
        for result in tqdm(results_iterable, total=len(dataset_for_attack)):
            if isinstance(result, SkippedAttackResult):
                original_text = result.original_text
                perturbed_text = result.original_text
                ground_truth_label = result.original_result.ground_truth_output
            else:
                original_text = result.original_text
                perturbed_text = result.perturbed_text
                ground_truth_label = result.ground_truth_output
            
            attacked_data.append({
                "original_text": original_text,
                "perturbed_text": perturbed_text,
                "ground_truth_id": ground_truth_label,
            })
            
        return pd.DataFrame(attacked_data)