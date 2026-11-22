"""Two unchanged functions from the local analyse_result.py used by train.py."""

import numpy as np
import torch
from torch.nn import functional as F


def cal_nma_longtail(y_true, y_hat, lamda=0.5):
    y_true = np.array(y_true)
    y_hat = np.array(y_hat)
    all_classes = sorted(set(y_true))

    class_acc = []
    for cls in all_classes:
        cls_mask = (y_true == cls)
        if cls_mask.sum() == 0:
            class_acc.append(0.0)
            continue
        correct = (y_hat[cls_mask] == cls).sum()
        class_acc.append(correct / cls_mask.sum())
    macro_acc = np.mean(class_acc)

    micro_acc = (y_hat == y_true).sum() / len(y_true)

    nma = lamda * macro_acc + (1 - lamda) * micro_acc
    return nma, macro_acc, micro_acc


def extract_features_during_training(net, test_iter, device='cuda:0'):
    """
    训练时提取特征:直接复用模型forward流程,在目标层(layer4)输出特征
    无需复现预处理流程——直接调用模型的中间层输出
    """
    net.eval()
    all_features = []
    all_labels = []

    # 定义一个临时的"特征提取钩子"，或直接修改模型forward返回中间特征（二选一）
    # 方案1：用钩子（无需修改模型结构，更灵活）
    target_features = None
    def hook_fn(module, input, output):
        nonlocal target_features
        target_features = output  # 保存layer4的输出特征

    # 给ResNet18的layer4注册钩子（提取该层特征）
    hook = net.resnet18.layer4.register_forward_hook(hook_fn)

    with torch.no_grad():
        for batch in test_iter:
            *X, targets = batch
            X = tuple(part.to(device) for part in X)
            targets = targets.to(device)

            # 正常调用模型forward——钩子会自动保存layer4特征
            _ = net(*X)  # 模型输出无用，只需要钩子保存的特征

            # 对layer4特征做全局平均池化（[B, 512, 7, 7] → [B, 512]）
            pooled_feat = F.adaptive_avg_pool2d(target_features, (1, 1)).squeeze()

            all_features.extend(pooled_feat.cpu().numpy())
            all_labels.extend(targets.cpu().numpy())

    # 移除钩子（避免影响后续测试）
    hook.remove()

    return np.array(all_features), np.array(all_labels)
