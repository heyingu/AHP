# src/components/result_aggregation.py
import numpy as np
import logging
from typing import List

def aggregate_results(candidate_probs: np.ndarray, strategy: str = 'majority_vote') -> np.ndarray:
    """
    聚合来自多个候选预测的概率分布。

    Args:
        candidate_probs (np.ndarray): 一个 Numpy 数组，形状为 [num_candidates, num_labels]，
                                     包含每个候选的预测概率分布。
        strategy (str): 聚合策略，例如 'majority_vote' 或 'weighted_vote'。

    Returns:
        np.ndarray: 聚合后的最终概率分布，形状为 [num_labels]。
    """
    # 检查输入数组是否为空或形状不正确
    if candidate_probs is None or candidate_probs.ndim != 2 or candidate_probs.shape[0] == 0:
        logging.warning("aggregate_results 收到无效的候选概率数组，返回均匀分布。")
        # 需要知道 num_labels，如果无法从输入推断，需要传递或设为默认值
        # 尝试从形状获取，如果失败则假设为 2 (例如情感分类)
        num_labels = candidate_probs.shape[1] if candidate_probs is not None and candidate_probs.ndim == 2 else 2
        return np.ones(num_labels) / num_labels

    num_candidates, num_labels = candidate_probs.shape

    logging.debug(f"正在使用 '{strategy}' 策略聚合 {num_candidates} 个候选的概率。")

    if strategy == 'majority_vote':
        # --- 多数投票逻辑 ---
        # 1. 获取每个候选预测的类别
        predictions = np.argmax(candidate_probs, axis=1)
        # 2. 统计每个类别的票数
        votes = np.bincount(predictions, minlength=num_labels)
        # 3. 找到得票最多的类别 (可能存在平票，argmax 会返回第一个最大值的索引)
        majority_class = np.argmax(votes)
        # 4. 创建一个 one-hot 向量表示最终预测
        aggregated_prob = np.zeros(num_labels)
        aggregated_prob[majority_class] = 1.0
        logging.debug(f"多数投票结果: 票数={votes}, 胜出类别={majority_class}")
        return aggregated_prob

    elif strategy == 'weighted_vote':
        # --- 加权投票逻辑 (示例：按置信度加权) ---
        # 权重可以是每个候选的最大预测概率 (置信度)
        weights = np.max(candidate_probs, axis=1) # 获取每个候选的最大概率作为权重
        # 对权重进行归一化 (可选，但通常是个好主意)
        normalized_weights = weights / np.sum(weights) if np.sum(weights) > 0 else np.ones_like(weights) / len(weights)

        # 计算加权平均概率
        # 使用 einsum 进行高效计算: sum(weights[i] * probs[i, j] for i) for each j
        aggregated_prob = np.einsum('i,ij->j', normalized_weights, candidate_probs)

        # 确保概率和为 1
        aggregated_prob /= np.sum(aggregated_prob)
        logging.debug(f"加权投票聚合后概率: {aggregated_prob}")
        return aggregated_prob

    # elif strategy == 'average_prob':
    #     # --- 平均概率逻辑 ---
    #     aggregated_prob = np.mean(candidate_probs, axis=0)
    #     # 确保概率和为 1
    #     aggregated_prob /= np.sum(aggregated_prob)
    #     logging.debug(f"平均概率聚合后概率: {aggregated_prob}")
    #     return aggregated_prob

    else:
        logging.error(f"未知的聚合策略: '{strategy}'。将返回均匀分布。")
        return np.ones(num_labels) / num_labels

# 你可以在这里添加其他聚合相关的辅助函数 (如果需要)