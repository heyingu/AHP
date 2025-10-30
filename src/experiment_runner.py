# src/experiment_runner.py
import logging
import os
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import textattack # 导入 TextAttack 基础库
from typing import List, Dict, Optional
import torch # <--- 添加 import torch

# --- 导入 TextAttack 相关组件 ---
from textattack.attack_recipes import (
    TextBuggerLi2018, TextFoolerJin2019, PWWSRen2019, BAEGarg2019, DeepWordBugGao2018
)
from textattack.attack_recipes.bert_attack_li_2020 import BERTAttackLi2020 # <--- 将 BertAttack 改为 BERTAttack
from textattack.models.wrappers import ModelWrapper
# textattack.datasets.Dataset is imported in data_loader utils now
from textattack.attack_args import AttackArgs
from textattack.attacker import Attacker
# from textattack.loggers import AttackLogManager, CSVLogger
from textattack.goal_function_results import GoalFunctionResultStatus # 用于判断攻击是否成功

# --- 导入本项目模块 ---
try:
    from .args_config import AHPSettings # 导入参数配置类
    from .models.model_loader import AlpacaModel # 导入我们重构的模型加载和防御类
    from .utils.data_loader import load_dataset, create_textattack_dataset # 导入数据加载辅助函数
    # from .utils.metrics import SimplifidAttackResult # 如果您有自定义的简化结果统计类，可以导入
except ImportError as e:
     logging.error(f"无法导入项目模块，请检查 experiment_runner.py 中的导入路径: {e}")
     raise # 导入失败是严重错误，直接抛出


# --- 定义 TextAttack 模型包装器 ---
class AlpacaModelWrapper(ModelWrapper):
    """
    为 TextAttack 包装 AlpacaModel 类。
    它提供 TextAttack 所需的 __call__ 接口，内部调用 AlpacaModel 的预测逻辑。
    """
    def __init__(self, model: AlpacaModel):
        """
        初始化包装器。

        Args:
            model (AlpacaModel): 已初始化的 AlpacaModel 实例。
        """
        self.model = model
        # TextAttack 的某些组件可能需要访问分词器
        self.tokenizer = model.tokenizer

    def __call__(self, text_input_list: List[str]) -> np.ndarray:
        """
        TextAttack 调用此方法进行模型预测。

        Args:
            text_input_list (List[str]): 需要预测的文本列表。

        Returns:
            np.ndarray: 模型预测的概率分布数组，形状为 [batch_size, num_labels]。
        """
        # 简单检查输入是否为空
        if not text_input_list:
            logging.warning("AlpacaModelWrapper 收到了空输入列表。")
            # 返回一个空的或者形状正确的空数组
            return np.zeros((0, self.model.num_labels)) # 假设模型有 num_labels 属性
        # 直接调用 AlpacaModel 实例的 __call__ 方法，该方法已实现 TextAttack 接口
        return self.model(text_input_list)

    def get_grad(self, text_input: str) -> Dict[str, torch.Tensor]:
        """
        获取输入文本相对于模型输出的梯度 (用于基于梯度的攻击)。
        对于通过 API 或无法直接访问内部计算图的 LLM，通常不支持此功能。
        """
         # 对于 Alpaca 这类大型语言模型，通过 TextAttack 获取梯度通常很困难或不可能
        logging.error("当前模型包装器不支持基于梯度的攻击。")
        raise NotImplementedError("此模型包装器不支持梯度计算。")


class ExperimentRunner:
    """负责协调整个实验流程，包括数据加载、模型初始化、运行评估或攻击。"""
    def __init__(self, args: AHPSettings):
        """
        初始化实验运行器。

        Args:
            args (AHPSettings): 包含所有实验配置的参数对象。
        """
        self.args = args
        # 初始化 Alpaca 模型，模型内部会根据 args.defense_method 处理防御逻辑
        self.alpaca_model = AlpacaModel(args)
        self.dataset: Optional[textattack.datasets.Dataset] = None # 用于存储加载的 TextAttack 数据集

    def _load_data(self):
        """按需加载数据集，并将其转换为 TextAttack 格式。"""
        # 只有在数据集尚未加载时才执行加载操作
        if self.dataset is None:
             logging.info(f"正在加载数据集: {self.args.dataset_name} (最多加载 {self.args.num_examples} 个样本)")
             # 调用 utils.data_loader 中的函数加载原始数据
             if self.args.dataset_name == 'sst2':
                 raw_data = load_dataset(self.args.dataset_full_path, self.args.dataset_name, split='validation', num_examples=self.args.num_examples)
             elif self.args.dataset_name == 'agnews':
                 raw_data = load_dataset(self.args.dataset_full_path, self.args.dataset_name, split='test', num_examples=self.args.num_examples)
             # 将原始数据转换为 TextAttack Dataset 对象
             self.dataset = create_textattack_dataset(raw_data, self.args.dataset_name)
             logging.info(f"已加载 {len(self.dataset)} 个样本供 TextAttack 使用。")

    def _get_attack_recipe(self) -> type:
        """根据配置参数返回相应的 TextAttack 攻击配方类。"""
        attack_name = self.args.attack_method.lower() # 转为小写以匹配
        if attack_name == 'textbugger':
            return TextBuggerLi2018
        elif attack_name == 'textfooler':
            return TextFoolerJin2019
        elif attack_name == 'pwws':
            return PWWSRen2019
        elif attack_name == 'bae':
            return BAEGarg2019
        elif attack_name == 'deepwordbug':
            return DeepWordBugGao2018
        elif attack_name == 'bertattack': 
            return BERTAttackLi2020
        else:
            # 如果配置了未知的攻击方法，则抛出错误
            raise ValueError(f"未知的攻击方法: {self.args.attack_method}")

    def run(self):
        """根据配置的模式 ('attack' 或 'evaluate') 执行相应的实验流程。"""
        if self.args.mode == 'attack':
            self.attack() # 执行对抗攻击流程
        elif self.args.mode == 'evaluate':
            self.evaluate() # 执行在干净样本上的评估流程
        else:
            raise ValueError(f"无效的运行模式: {self.args.mode}")

    def evaluate(self):
        """
        评估模型在干净（未被攻击）的测试样本上的准确率。
        评估时会应用配置中指定的防御方法。
        """
        # --- 修改开始 ---
        # 不再调用 self._load_data() 来创建 TextAttack Dataset 对象
        # 而是直接加载原始数据
        logging.info(f"正在加载原始数据用于评估: {self.args.dataset_name} (最多 {self.args.num_examples} 个样本)")
        try:
            # 直接调用 load_dataset 函数获取原始 (文本, 标签) 列表
            raw_data = load_dataset(self.args.dataset_full_path, self.args.dataset_name, split='test', num_examples=self.args.num_examples)
            if not raw_data:
                logging.error("未能加载到用于评估的数据。")
                return # 无法继续评估
        except Exception as e:
            logging.error(f"加载评估数据时出错: {e}", exc_info=True)
            return # 无法继续评估

        logging.info(f"已加载 {len(raw_data)} 个原始样本用于评估。")
        logging.info(f"正在评估干净样本准确率，使用防御方法: {self.args.defense_method}")
        # --- 修改结束 ---

        correct_count = 0 # 正确预测的数量
        total_count = 0   # 总样本数量
        batch_size = self.args.model_batch_size # 使用模型推理的批次大小

        # --- 修改开始 ---
        # 直接从 raw_data 中提取文本和标签
        texts = [item[0] for item in raw_data]
        true_labels = [item[1] for item in raw_data]
        # --- 修改结束 ---

        # 使用 tqdm 创建评估进度条
        eval_iterator = tqdm(range(0, len(texts), batch_size), desc="正在评估", ncols=100)
        all_preds = [] # 存储所有预测结果
        # 按批次进行评估
        for i in eval_iterator:
            batch_texts = texts[i : i + batch_size] # 当前批次的文本
            batch_labels = true_labels[i : i + batch_size] # 当前批次的真实标签

            # 调用 AlpacaModel 的 predict_batch 方法获取预测概率列表
            # 这个方法会根据 self.args.defense_method 自动应用防御
            batch_probs_list = self.alpaca_model.predict_batch(batch_texts) # 返回 List[np.ndarray]

            # 从概率分布中获取预测的类别 ID (概率最高的那个)
            batch_preds = [np.argmax(p) for p in batch_probs_list]
            all_preds.extend(batch_preds) # 收集预测结果

            # 比较预测结果和真实标签，统计正确数量
            for pred, true in zip(batch_preds, batch_labels):
                 if pred == true:
                     correct_count += 1
            total_count += len(batch_texts) # 更新已处理的总样本数

            # 计算当前批次的累积准确率，并更新进度条显示
            accuracy = correct_count / total_count if total_count > 0 else 0
            eval_iterator.set_postfix({"准确率": f"{accuracy:.2%}"})

        # ... (后续的日志记录和保存结果部分保持不变) ...
        final_accuracy = correct_count / total_count if total_count > 0 else 0
        logging.info(f"评估完成。最终准确率 ({self.args.defense_method} 防御下): {final_accuracy:.2%}")
        # ... (保存结果的代码) ...


    def attack(self):
        """使用 TextAttack 执行对抗攻击流程。"""
        self._load_data() # 确保数据已加载
        logging.info(f"开始攻击: {self.args.attack_method} on {self.args.dataset_name}, 使用防御: {self.args.defense_method}")

        # --- 1. 包装模型 ---
        # 创建 AlpacaModelWrapper 实例，将我们的 AlpacaModel 包装成 TextAttack 可用的格式
        model_wrapper = AlpacaModelWrapper(self.alpaca_model)

        # --- 2. 选择攻击配方 ---
        # 获取与配置参数对应的 TextAttack 攻击配方类
        attack_recipe_class = self._get_attack_recipe()
        # 使用配方类的 .build() 方法构建攻击实例，传入包装后的模型
        attack = attack_recipe_class.build(model_wrapper)

        # --- 3. 配置攻击参数 (例如查询预算) ---
        # 尝试设置查询预算。注意：并非所有攻击配方或搜索方法都支持查询预算限制。
        # TextAttack 中通常在 GoalFunction 或 SearchMethod 中限制查询。
        # 这里尝试设置 GoalFunction 的 query_budget (较新版本 TextAttack 的做法)
        if hasattr(attack, 'goal_function') and hasattr(attack.goal_function, 'query_budget'):
             attack.goal_function.query_budget = self.args.attack_query_budget
             logging.info(f"已设置攻击目标函数的查询预算为: {self.args.attack_query_budget}")
        # 也尝试设置 SearchMethod 的 max_queries (一些旧的或特定的搜索方法可能使用这个)
        elif hasattr(attack, 'search_method') and hasattr(attack.search_method, 'max_queries'):
             attack.search_method.max_queries = self.args.attack_query_budget
             logging.info(f"已设置搜索方法的最大查询数为: {self.args.attack_query_budget}")
        # 如果设置了查询预算但无法在攻击对象中找到相应属性，则发出警告
        elif self.args.attack_query_budget < float('inf'): # 检查是否设置了有限预算
            logging.warning(f"设置了查询预算 ({self.args.attack_query_budget}), 但攻击配方/搜索方法可能不支持此限制。")


        # --- 4. 配置 TextAttack 的 AttackArgs ---
        # 定义详细攻击日志的 CSV 文件路径
        attack_log_csv_path = os.path.join(self.args.attack_log_path, f"{self.args.dataset_name}_{self.args.attack_method}_{self.args.defense_method}_log.csv")
        attack_args = AttackArgs(
            num_examples=len(self.dataset),
            log_to_csv=attack_log_csv_path, # <-- 指定 CSV 日志路径
            disable_stdout=True,           # 禁用 TextAttack 默认控制台输出
            silent=True                    # 禁用 TextAttack 默认 tqdm
            # TextAttack 会根据 log_to_csv 参数自动设置 CSVLogger
        )

        # --- 5. 设置日志记录器 ---
        # log_manager = AttackLogManager(metrics=[]) # 管理所有日志记录器
        # # 创建 CSVLogger，用于将每个样本的详细攻击结果记录到 CSV 文件
        # csv_logger = CSVLogger(filename=attack_args.log_to_csv, color_method="file") # color_method='file' 可以在文件中保留颜色标记
        # log_manager.add_logger(csv_logger) # 将 CSVLogger 添加到管理器
        # 注意：我们将在攻击结束后手动记录摘要信息

        # --- 6. 执行攻击 ---
        # 创建 Attacker 实例，传入攻击对象、数据集和攻击参数
        attacker = Attacker(attack, self.dataset, attack_args)
        results = [] # 存储每个样本的攻击结果对象
        # 使用 tqdm 创建攻击进度条，迭代执行攻击
        attack_iterator = tqdm(attacker.attack_dataset(), total=len(self.dataset), desc="正在攻击", ncols=100)
        # 遍历每个样本的攻击结果
        for result in attack_iterator:
             results.append(result) # 保存结果对象
             # log_manager.log_result(result) # 将当前结果写入配置的日志文件 (CSV)

        # --- 7. 处理和记录攻击摘要结果 ---
        logging.info("攻击完成。正在汇总结果...")
        num_results = len(results) # 总样本数
        # 统计攻击成功、失败和跳过 (例如原始预测就错误) 的数量
        # --- 修改这里：使用类属性访问状态，并根据可用状态计算失败数 ---
        # 统计攻击成功数 (SUCCEEDED)
        num_successes = sum(r.perturbed_result.goal_status == GoalFunctionResultStatus.SUCCEEDED for r in results)
        # 统计跳过样本数 (SKIPPED)
        num_skipped = sum(r.perturbed_result.goal_status == GoalFunctionResultStatus.SKIPPED for r in results)
        # 计算攻击失败数 (总数 - 成功数 - 跳过数)
        num_failures = num_results - num_successes - num_skipped
        # --- 修改结束 ---

        # 计算指标
        # 在计算准确率和攻击成功率时，通常排除被跳过的样本
        valid_examples_count = num_successes + num_failures # 未被跳过的样本总数

        # 原始准确率 (在未被跳过的样本中，攻击失败的比例)
        original_accuracy = num_failures / valid_examples_count if valid_examples_count > 0 else 0
        # 攻击后准确率 (同上，因为攻击失败意味着模型在对抗样本上预测正确)
        accuracy_under_attack = original_accuracy

        # 攻击成功率 (在未被跳过的样本中，攻击成功的比例)
        attack_success_rate = num_successes / valid_examples_count if valid_examples_count > 0 else 0

        # 计算平均扰动词数 (仅对攻击成功的样本) 和平均查询次数 (对所有未跳过样本)
        perturbed_word_counts = []
        query_counts = []
        for r in results:
             # 只统计未跳过的样本的查询次数
             if r.perturbed_result.goal_status != GoalFunctionResultStatus.SKIPPED:
                 query_counts.append(r.num_queries)
                 # 只统计攻击成功的样本的扰动词数
                 if r.perturbed_result.goal_status == GoalFunctionResultStatus.SUCCEEDED:
                     try:
                        # 尝试调用 TextAttack 的方法获取修改词数 (方法名可能随版本变化)
                        # 这是一个常用的计算方式：比较原始文本和扰动后文本的差异
                        words_changed = len(r.original_result.attacked_text.all_words_diff(r.perturbed_result.attacked_text))
                        perturbed_word_counts.append(words_changed)
                     except Exception as e: # 如果方法不存在或出错，记录警告
                        logging.warning(f"无法计算样本的扰动词数: {e}")
                        perturbed_word_counts.append(0) # 记为 0

        avg_perturbed_words = np.mean(perturbed_word_counts) if perturbed_word_counts else 0
        avg_queries = np.mean(query_counts) if query_counts else 0

        # 打印摘要日志
        logging.info(f"总样本数: {num_results}")
        logging.info(f"攻击成功数: {num_successes}")
        logging.info(f"攻击失败数: {num_failures}")
        logging.info(f"跳过样本数 (原始预测错误): {num_skipped}")
        logging.info(f"原始准确率 (未跳过样本): {original_accuracy:.2%}")
        logging.info(f"攻击后准确率 (未跳过样本): {accuracy_under_attack:.2%}")
        logging.info(f"攻击成功率 (未跳过样本): {attack_success_rate:.2%}")
        logging.info(f"平均扰动词数 (成功样本): {avg_perturbed_words:.2f}")
        logging.info(f"平均查询次数 (未跳过样本): {avg_queries:.2f}")

        # --- 8. 将摘要结果追加到主 CSV 文件 ---
        results_summary = {
            'dataset': self.args.dataset_name,
            'model': os.path.basename(self.args.model_path),
            'defense': self.args.defense_method,
            'attack': self.args.attack_method,
            'num_examples': num_results, # 可以记录总数或有效数，这里用总数
            'accuracy': accuracy_under_attack, # 记录攻击后的准确率
            'attack_success_rate': attack_success_rate,
            'avg_perturbed_words': avg_perturbed_words,
            'avg_queries': avg_queries,
            'query_budget': self.args.attack_query_budget # 记录查询预算设置
        }
        # 添加防御相关参数
        if self.args.defense_method != 'none':
             results_summary['mask_rate'] = self.args.mask_rate
             if self.args.defense_method == 'selfdenoise':
                 results_summary.update({
                     'denoiser': self.args.selfdenoise_denoiser,
                     'ensemble_size': self.args.selfdenoise_ensemble_size
                 })
             elif self.args.defense_method == 'ahp':
                 results_summary.update({
                     'ahp_pruning': self.args.ahp_pruning_method,
                     'ahp_aggregation': self.args.ahp_aggregation_strategy
                 })

        # (追加逻辑同 evaluate 方法)
        df_new = pd.DataFrame([results_summary])
        try:
            if os.path.exists(self.args.results_file):
                 df_existing = pd.read_csv(self.args.results_file)
                 df_combined = pd.concat([df_existing, df_new], ignore_index=True)
            else:
                 df_combined = df_new
            df_combined.to_csv(self.args.results_file, index=False)
            logging.info(f"攻击摘要结果已追加到: {self.args.results_file}")
        except Exception as e:
            logging.error(f"保存攻击摘要结果失败: {e}")

        # log_manager.flush() # 确保所有日志都已写入文件