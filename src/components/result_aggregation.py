# ahp_robustness/src/components/result_aggregation.py

from collections import Counter

def majority_voting(predictions):
    """
    对一组预测结果进行多数投票。

    Args:
        predictions (list): 包含K个预测结果的列表 (例如 ['positive', 'negative', 'positive'])。

    Returns:
        any: 投票后得出的最终预测结果。
    """
    if not predictions:
        return None
    
    # 使用Counter来统计每个预测结果出现的次数
    vote_counts = Counter(predictions)
    
    # 找到出现次数最多的结果
    # most_common(1) 返回一个列表，其中包含一个元组 (元素, 次数)
    top_prediction = vote_counts.most_common(1)[0][0]
    
    return top_prediction