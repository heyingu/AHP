# ahp_robustness/src/utils/data_loader.py

from datasets import load_dataset
import pandas as pd

def load_sst2_dataset(split="test"):
    """
    加载SST-2数据集。

    Args:
        split (str): 要加载的数据集部分 ('train', 'validation', 'test')。

    Returns:
        datasets.Dataset: 加载好的数据集对象，包含'sentence'和'label'列。
    """
    print(f"正在加载SST-2数据集 ({split} split)...")
    try:
        # SST-2在GLUE基准测试中
        dataset = load_dataset("glue", "sst2", split=split)
        # 将label从数字(0, 1)转换为字符串('negative', 'positive')
        def map_labels(example):
            example['label_text'] = dataset.features['label'].names[example['label']]
            return example
        dataset = dataset.map(map_labels)
        print("SST-2数据集加载成功。")
        return dataset
    except Exception as e:
        print(f"加载SST-2数据集失败: {e}")
        return None

def load_agnews_dataset(split="test"):
    """
    加载AG News数据集。

    Args:
        split (str): 要加载的数据集部分 ('train', 'test')。

    Returns:
        datasets.Dataset: 加载好的数据集对象，包含'text'和'label'列。
    """
    print(f"正在加载AG News数据集 ({split} split)...")
    try:
        dataset = load_dataset("ag_news", split=split)
        # 添加一个label的文本描述列
        def map_labels(example):
            example['label_text'] = dataset.features['label'].names[example['label']]
            return example
        dataset = dataset.map(map_labels)
        print("AG News数据集加载成功。")
        return dataset
    except Exception as e:
        print(f"加载AG News数据集失败: {e}")
        return None

# 示例：
if __name__ == '__main__':
    # 测试加载功能
    sst2_test = load_sst2_dataset()
    if sst2_test:
        print("\nSST-2 示例数据:")
        print(sst2_test[0])

    print("-" * 30)
    
    agnews_test = load_agnews_dataset()
    if agnews_test:
        print("\nAG News 示例数据:")
        print(agnews_test[0])