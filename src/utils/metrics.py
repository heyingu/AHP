# ahp_robustness/src/utils/metrics.py

from sklearn.metrics import accuracy_score

def calculate_accuracy(true_labels, predictions):
    """
    计算分类准确率。

    Args:
        true_labels (list): 真实的标签列表。
        predictions (list): 模型的预测列表。

    Returns:
        float: 准确率，值在0.0到1.0之间。
    """
    return accuracy_score(true_labels, predictions)

def calculate_asr(original_preds, attack_preds, true_labels):
    """
    计算攻击成功率 (Attack Success Rate, ASR)。
    ASR = 原始预测正确、攻击后预测错误的样本 / 原始预测正确的样本

    Args:
        original_preds (list): 对原始样本的预测。
        attack_preds (list): 对攻击样本的预测。
        true_labels (list): 真实的标签。

    Returns:
        float: 攻击成功率。
    """
    correct_before_attack = 0
    successful_attacks = 0
    
    for i in range(len(true_labels)):
        if original_preds[i] == true_labels[i]:
            correct_before_attack += 1
            if attack_preds[i] != true_labels[i]:
                successful_attacks += 1
    
    if correct_before_attack == 0:
        return 0.0
        
    return successful_attacks / correct_before_attack