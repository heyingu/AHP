# src/utils/data_loader.py
import os
import csv
import logging
from typing import List, Tuple, Dict
from datasets import load_dataset as hf_load_dataset # 使用 Hugging Face 的 datasets 库简化加载
from textattack.datasets import Dataset
from collections import OrderedDict # TextAttack 需要 OrderedDict

def load_dataset(dataset_path: str, dataset_name:str, split: str = 'test', num_examples: int = -1) -> List[Tuple[str, int]]:
    """
    使用 Hugging Face datasets 库加载 SST-2 或 AG News 数据集。

    Args:
        dataset_path (str): 数据集的基础路径 (如果使用 HF 库加载标准数据集可能不需要)。
        dataset_name (str): 数据集名称，支持 'sst2' 或 'agnews'。
        split (str): 数据集划分，例如 'test', 'train', 'validation'/'dev'。
        num_examples (int): 加载的最大样本数量，-1 表示加载全部。

    Returns:
        List[Tuple[str, int]]: 一个包含 (文本, 标签ID) 元组的列表。
    """
    raw_data = []
    # Hugging Face datasets 可能使用 'validation' 而不是 'dev'
    hf_split = 'test' if split == 'test' else ('train' if split == 'train' else 'validation')

    try:
        logging.info(f"尝试使用 Hugging Face datasets 加载 '{dataset_name}' 数据集...")
        if dataset_name == 'sst2':
            # 使用标准的 Hugging Face 数据集标识符
            dataset = hf_load_dataset("sst2", split=hf_split)
            text_key = 'sentence' # SST-2 数据集中的文本字段名
            label_key = 'label'   # SST-2 数据集中的标签字段名
            label_map = {0: 0, 1: 1} # SST-2 标签本身就是 0 (negative) 和 1 (positive)
        elif dataset_name == 'agnews':
            # 使用标准的 Hugging Face 数据集标识符
            dataset = hf_load_dataset("ag_news", split=hf_split)
            # AG News 通常结合标题和描述，与 SelfDenoise/AHP 的 Prompt 类似
            # 定义一个函数来合并 title 和 text 字段
            text_key = lambda x: f"Title: {x['title']}\nDescription: {x['text']}"
            label_key = 'label'   # AG News 数据集中的标签字段名
            # AG News 标签: 0: World, 1: Sports, 2: Business, 3: Sci/Tech
            # 这个映射关系基于 model_loader.py 中的 DATASET_INSTRUCTIONS
            label_map = {0: 0, 1: 1, 2: 2, 3: 3} # 假设与 HF 默认一致
        else:
            raise ValueError(f"此加载器不支持数据集 '{dataset_name}'。")

        logging.info(f"加载的数据集结构: {dataset.features}")

        count = 0
        # 遍历加载的数据集
        for item in dataset:
            # 如果达到了 num_examples 限制，则停止
            if num_examples != -1 and count >= num_examples:
                break
            try:
                # 获取文本，根据 text_key 是字符串还是函数
                if callable(text_key):
                    text = text_key(item)
                else:
                    text = item[text_key]

                label = item[label_key] # 获取原始标签
                label_id = label_map[label] # 将原始标签映射到整数 ID
                raw_data.append((text, label_id))
                count += 1
            except KeyError as e:
                logging.warning(f"跳过样本，因为缺少键: {e}。样本: {item}")
            except Exception as e:
                 logging.warning(f"跳过样本，因为发生错误: {e}。样本: {item}")


    except Exception as e:
        # 如果使用 HF datasets 失败，记录错误并提示用户需要手动实现加载逻辑
        logging.error(f"使用 Hugging Face datasets 加载数据集 '{dataset_name}' 失败: {e}")
        logging.info("将回退到手动加载（如果需要，请在此处实现）...")
        # 在这里添加手动从 TSV/CSV 文件加载的代码，需要根据您在 dataset_path 下的文件格式进行适配
        # 例如，解析 SST-2 的 test.txt (每行一个句子，测试集通常无标签)
        # 例如，解析 AG News 的 test.tsv (label \t title \t description)
        raise NotImplementedError("需要根据您的具体文件格式实现手动数据集加载逻辑。")

    if not raw_data:
         raise RuntimeError(f"未能为 {dataset_name} 的 {split} 部分加载任何数据。请检查路径和格式。")

    logging.info(f"成功为 {dataset_name} 的 {split} 部分加载了 {len(raw_data)} 个样本。")
    return raw_data


def create_textattack_dataset(raw_data: List[Tuple[str, int]], dataset_name: str) -> Dataset:
    """
    将加载的原始数据转换为 TextAttack 所需的 Dataset 格式。

    Args:
        raw_data (List[Tuple[str, int]]): 包含 (文本, 标签ID) 元组的列表。
        dataset_name (str): 数据集名称 (可能影响输入列的名称)。

    Returns:
        Dataset: 一个 TextAttack Dataset 对象。
    """
    # TextAttack 期望的数据格式是 [(OrderedDict, label), ...]
    # OrderedDict 用于映射输入列名到对应的文本内容
    # 对于单文本输入任务，通常使用 {'text': text_content}
    processed_data = []
    for text, label_id in raw_data:
        # 假设所有任务都使用 'text' 作为输入列名
        # 如果您的模型或 TextAttack 配置需要不同的列名，请在这里修改
        input_dict = OrderedDict([('text', text)])
        processed_data.append((input_dict, label_id))

    # 创建 Dataset 对象时指定输入列名
    # 注意：input_columns 必须是一个列表，即使只有一个输入列
    return Dataset(processed_data, input_columns=['text'])