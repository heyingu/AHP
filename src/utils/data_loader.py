# src/utils/data_loader.py
import os
import csv
import logging
from typing import List, Tuple, Dict
# from datasets import load_dataset as hf_load_dataset # <-- 已移除，不再访问 Hugging Face
from textattack.datasets import Dataset
# from collections import OrderedDict # <-- 不再需要
import random

def load_dataset(dataset_path: str, dataset_name:str, split: str = 'test', num_examples: int = -1) -> List[Tuple[str, int]]:
    """
    [已修改] 强制从本地文件加载 SST-2 或 AG News 数据集。
    不再尝试访问 Hugging Face Hub。
    AG News 加载器已更新为处理 2 列 TSV (text \t label)。
    SST-2 加载器已更新为处理 TSV 格式 (text \t label) for train/dev/validation。
    """
    raw_data = []
    
    logging.info(f"将直接从本地路径手动加载数据集: {dataset_path}")

    # 确定要读取的文件名
    filename = f"{split}.tsv" if dataset_name == 'agnews' else f"{split}.txt"
    filepath = os.path.join(dataset_path, filename) 

    logging.info(f"尝试读取文件: {filepath}")

    if not os.path.exists(filepath):
        logging.error(f"手动加载失败：文件 {filepath} 不存在。")
        raise FileNotFoundError(f"数据集文件 {filepath} 未找到。请确保 --dataset_path 指向正确的目录 (例如 ../data/sst2) 并且文件存在。")

    try:
        count = 0
        if dataset_name == 'agnews' and filename.endswith('.tsv'):
            # --- 手动加载 AG News (TSV格式: 假设 2 列: text \t label) ---
            with open(filepath, 'r', encoding='utf-8') as f:
                reader = csv.reader(f, delimiter='\t', quoting=csv.QUOTE_NONE)
                try:
                    header = next(reader)
                    logging.info(f"跳过 AG News 表头: {header}")
                    if header[0].lower() != 'text' or header[1].lower() != 'label':
                         logging.warning(f"AG News TSV 表头 ({header}) 与预期的 ['text', 'label'] 不符。")
                except StopIteration:
                    logging.warning("AG News TSV 文件为空。")
                    pass 
                except Exception as e:
                    logging.warning(f"读取 AG News 表头时出错 (可能没有表头): {e}")

                lines = list(reader) 
                
                if num_examples != -1 and len(lines) > num_examples:
                     logging.info(f"从 {len(lines)} 个样本中随机抽取 {num_examples} 个。")
                     lines = random.sample(lines, num_examples)

                for row in lines:
                    if len(row) == 2: # 格式: text \t label
                        try:
                            description = row[0]       
                            label_id_raw_str = row[1]  
                            label_id = int(label_id_raw_str) 
                            if label_id not in [0, 1, 2, 3]:
                                 logging.warning(f"跳过 AG News (2列) 行，无效标签值: {label_id_raw_str}")
                                 continue
                            text = f"Title: \nDescription: {description}" 
                            raw_data.append((text, label_id))
                            count += 1
                        except ValueError:
                            logging.warning(f"跳过 AG News (2列) 行，无法解析标签: {row}")
                        except Exception as inner_e:
                            logging.warning(f"处理 AG News (2列) 行时出错: {inner_e}, 行: {row}")
                    # ... (3-column logic ...)
                    elif len(row) == 3: 
                        logging.debug("检测到 3 列格式。")
                        try:
                            label_map_3col = {1: 0, 2: 1, 3: 2, 4: 3}
                            label_id_raw = int(row[0])
                            label_id = label_map_3col.get(label_id_raw)
                            if label_id is None:
                                 logging.warning(f"跳过 AG News (3列) 行，无效标签: {row[0]}")
                                 continue
                            title = row[1]
                            description = row[2]
                            text = f"Title: {title}\nDescription: {description}"
                            raw_data.append((text, label_id))
                            count += 1
                        except ValueError:
                            logging.warning(f"跳过 AG News (3列) 行，无法解析标签: {row}")
                    else:
                         logging.warning(f"跳过 AG News 行，列数不为 2 或 3: {row}")

        elif dataset_name == 'sst2' and filename.endswith('.txt'):
             # --- [修复] 手动加载 SST-2 (TXT格式，但内容是 TSV: text \t label) ---
             with open(filepath, 'r', encoding='utf-8') as f:
                 lines = f.readlines()
                 
                 # 检查并跳过表头 (例如 'sentence\tlabel')
                 if lines and ("sentence\tlabel" in lines[0] or "sentence" in lines[0]):
                     logging.info(f"跳过 SST-2 表头: {lines[0].strip()}")
                     lines = lines[1:]

                 if num_examples != -1 and len(lines) > num_examples:
                      logging.info(f"从 {len(lines)} 个样本中随机抽取 {num_examples} 个。")
                      lines = random.sample(lines, num_examples)

                 for line in lines:
                     line = line.strip()
                     if not line: continue

                     # test.txt (假设没有标签)
                     if split == 'test':
                         text = line
                         label_id = 0 # 伪标签
                         raw_data.append((text, label_id))
                         count += 1
                     else: 
                         # train.txt, validation.txt 等 (假设格式为 "text\tlabel")
                         # --- 修复开始 ---
                         parts = line.split('\t', 1) # <--- 修复：按制表符 Tab 分割
                         
                         if len(parts) == 2:
                             try:
                                 text = parts[0] # <--- 修复：第一列是 text
                                 label_id = int(parts[1]) # <--- 修复：第二列是 label
                                 
                                 if label_id not in [0, 1]:
                                     logging.warning(f"跳过 SST-2 行，无效标签: {parts[1]}")
                                     continue
                                 raw_data.append((text, label_id))
                                 count += 1
                             except ValueError:
                                 # 这会捕获 int(parts[1]) 失败的情况
                                 logging.warning(f"跳过 SST-2 行，无法解析标签: {line}")
                         else:
                             # 捕获格式错误的行
                             logging.warning(f"跳过 SST-2 行，格式不符合 'text\tlabel': {line}")
                         # --- 修复结束 ---
        else:
             raise NotImplementedError(f"未实现对 {dataset_name} ({filename}) 的手动加载逻辑。")

        if not raw_data:
             raise RuntimeError(f"手动加载后未能从 {filepath} 读取到任何有效数据。请检查文件内容和格式。")

        logging.info(f"成功从 {filepath} 手动加载了 {len(raw_data)} 个样本。")

    except Exception as manual_e:
        logging.error(f"手动加载数据时发生严重错误: {manual_e}", exc_info=True)
        raise RuntimeError(f"尝试手动加载数据失败: {manual_e}")

    if not raw_data:
         raise RuntimeError(f"未能为 {dataset_name} 的 {split} 部分加载任何数据。")

    logging.info(f"最终为 {dataset_name} split {split} 加载了 {len(raw_data)} 个样本。")
    return raw_data


def create_textattack_dataset(raw_data: List[Tuple[str, int]], dataset_name: str) -> Dataset:
    """
    将加载的原始数据转换为 TextAttack 所需的 Dataset 格式。
    (使用元组代替 OrderedDict)
    """
    processed_data = []
    for text, label_id in raw_data:
        input_tuple = (text,)
        processed_data.append((input_tuple, label_id))
    return Dataset(processed_data, input_columns=['text'])