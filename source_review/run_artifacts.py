# coding: utf-8
"""
run_artifacts.py — 单 run 规范工件落盘。

被 train.py 在训练+旧式写入完成后调用(write_run_artifacts)。把本次 run 的
配置、指标、原始预测(logits/y_true/y_pred/probabilities/样本 ID)、逐类指标、
混淆矩阵、训练历史、checkpoint 统一写入 canonical 目录(由 --exp_out 指定)。

不改变任何训练/数据/指标口径；只负责"存得全、存得规范、可校验"。
"""
import os
import json
import csv
import shutil
import numpy as np
from sklearn.metrics import (
    f1_score, accuracy_score, balanced_accuracy_score,
    precision_score, recall_score, precision_recall_fscore_support,
    confusion_matrix,
)

from my_utils import cal_gmean, load_test_sample_ids
from paper_metrics import cal_nma_longtail


CODE_DIR = os.path.dirname(os.path.abspath(__file__))


def _infer_protocol(args, exp_out):
    """从 --exp_out 路径推断 protocol(即 results/<protocol>/<group>/<config>/run_XX 的第 1 段)。

    不能只看 split_prefix: FLSMDD 的 split_prefix 同样是 '' ,会被误标成 main30。
    仅当路径不在 results/ 下时才回退到 split_prefix 推断。
    """
    norm = os.path.normpath(os.path.abspath(exp_out)).replace('\\', '/')
    marker = '/results/'
    if marker in norm:
        first = norm.split(marker, 1)[1].split('/')[0]
        if first:
            return first
    return 'htpl20' if args.split_prefix else 'main30'


def _softmax(logits):
    a = np.asarray(logits, dtype=np.float64)
    a = a - np.max(a, axis=1, keepdims=True)
    e = np.exp(a)
    return e / np.sum(e, axis=1, keepdims=True)


def _best_epoch_from_log(log_path):
    """从训练日志解析 (best_epoch, best_val_gmean, history_rows)。epoch 为 0 基。"""
    rows = []
    best_epoch = None
    best_val = -np.inf
    with open(log_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    for line in lines[1:]:  # 跳过表头 epoch,train_loss,val_loss,val_gmean
        line = line.strip()
        if not line:
            continue
        parts = line.split(',')
        if len(parts) < 4:
            continue
        epoch = int(parts[0])
        val_gmean = float(parts[3])
        rows.append(parts)
        if val_gmean > best_val:
            best_val = val_gmean
            best_epoch = epoch
    return best_epoch, best_val, rows


def write_run_artifacts(args, *, y_true, y_pred, logits, best_val_gmean,
                        num_classes, class_names, group_names, rgb_cnn_in_ch,
                        wavelet_cnn_in_chs, train_num, val_num, test_num,
                        train_per_class, betas=None):
    """写入单 run 的 canonical 工件到 args.exp_out 目录。"""
    exp_out = args.exp_out
    if not exp_out:
        return
    exp_out = os.path.abspath(exp_out)
    os.makedirs(exp_out, exist_ok=True)

    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    logits_arr = np.asarray(logits, dtype=np.float32)

    # ---------- 指标 ----------
    gmean = float(cal_gmean(y_true, y_pred))
    nma, macro_acc, micro_acc = cal_nma_longtail(y_true, y_pred)
    metrics = {
        'gmean': gmean,
        'macro_f1': float(f1_score(y_true, y_pred, average='macro')),
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'balanced_accuracy': float(balanced_accuracy_score(y_true, y_pred)),
        'macro_precision': float(precision_score(y_true, y_pred, average='macro', zero_division=0)),
        'macro_recall': float(recall_score(y_true, y_pred, average='macro', zero_division=0)),
        'nma': float(nma),
        'macro_acc': float(macro_acc),
        'micro_acc': float(micro_acc),
    }

    # ---------- 训练历史 / best_epoch ----------
    log_path = os.path.join(CODE_DIR, '..', 'output', 'log', args.dataset, args.method_name,
                            f'log_p{args.p_value}_k{args.k_value}_seed{args.seed}.txt')
    best_epoch, best_val_from_log, history_rows = _best_epoch_from_log(log_path)
    best_epoch = best_epoch if best_epoch is not None else -1
    metrics['best_epoch'] = best_epoch
    metrics['best_val_gmean'] = best_val_gmean

    with open(os.path.join(exp_out, 'history.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['epoch', 'train_loss', 'val_loss', 'val_gmean'])
        w.writerows(history_rows)

    # ---------- 原始预测 + 样本 ID ----------
    probabilities = _softmax(logits_arr).astype(np.float32)
    sample_ids = load_test_sample_ids(
        os.path.join(CODE_DIR, '..', 'data', args.dataset), args.split_prefix)
    if len(sample_ids) != len(y_true):
        raise ValueError(f'sample_ids({len(sample_ids)}) 与 y_true({len(y_true)}) 长度不一致')

    np.save(os.path.join(exp_out, 'y_true.npy'), y_true)
    np.save(os.path.join(exp_out, 'y_pred.npy'), y_pred)
    np.save(os.path.join(exp_out, 'logits.npy'), logits_arr)
    np.save(os.path.join(exp_out, 'probabilities.npy'), probabilities)
    with open(os.path.join(exp_out, 'sample_ids.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(sample_ids) + '\n')

    # ---------- 逐类指标 / 混淆矩阵 ----------
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(num_classes)), zero_division=0)
    with open(os.path.join(exp_out, 'per_class_metrics.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['class', 'class_name', 'precision', 'recall', 'f1', 'support'])
        for c in range(num_classes):
            w.writerow([c, class_names[c], precision[c], recall[c], f1[c], int(support[c])])
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    np.save(os.path.join(exp_out, 'confusion_matrix.npy'), cm)

    # ---------- Branch-Adaptive Gate 权重(仅 branch_gate 时) ----------
    if betas is not None:
        betas_arr = np.asarray(betas, dtype=np.float32)
        if betas_arr.ndim != 2 or betas_arr.shape[0] != len(y_true):
            raise ValueError(f'betas 形状 {betas_arr.shape} 与 y_true({len(y_true)}) 不一致')
        n_branch = betas_arr.shape[1]
        if len(group_names) != n_branch:
            raise ValueError(f'branch 名({group_names}) 与 beta 列数({n_branch}) 不一致')
        np.save(os.path.join(exp_out, 'gate_betas.npy'), betas_arr)
        # 逐样本 CSV: sample_id / true_label / pred_label / beta_{branch}
        with open(os.path.join(exp_out, 'gate_weights.csv'), 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['sample_id', 'true_label', 'pred_label'] + [f'beta_{n}' for n in group_names])
            for i in range(len(y_true)):
                w.writerow([sample_ids[i], int(y_true[i]), int(y_pred[i])]
                           + [float(x) for x in betas_arr[i]])
        # 统计: 整体 mean±std + 逐类 mean(该 seed 内)
        overall_mean = betas_arr.mean(axis=0)
        overall_std = betas_arr.std(axis=0)
        per_class_mean = {}
        for c in range(num_classes):
            mask = y_true == c
            per_class_mean[str(c)] = betas_arr[mask].mean(axis=0).tolist() if mask.sum() else None
        gate_stats = {
            'gate_type': 'static' if getattr(args, 'static_branch_gate', False) else 'baf',
            'branch_names': group_names,
            'class_names': class_names,
            'overall_mean': overall_mean.tolist(),
            'overall_std': overall_std.tolist(),
            'per_class_mean': per_class_mean,
        }
        # 静态分支加权: 额外记录最终 4 个全局权重(与 overall_mean 相同, 跨样本恒定)
        if getattr(args, 'static_branch_gate', False):
            gate_stats['static_beta'] = overall_mean.tolist()
        with open(os.path.join(exp_out, 'gate_weights_stats.json'), 'w', encoding='utf-8') as f:
            json.dump(gate_stats, f, indent=2, ensure_ascii=False)

    # ---------- checkpoint 复制 ----------
    best_model_src = os.path.join(
        CODE_DIR, '..', 'output', 'model', args.dataset, args.method_name,
        f'best_model_p{args.p_value}_k{args.k_value}_seed{args.seed}.pth')
    if os.path.exists(best_model_src):
        shutil.copy2(best_model_src, os.path.join(exp_out, 'best_model.pth'))
    else:
        print(f'⚠️ 未找到 best_model 源文件(将不复制): {best_model_src}')

    # ---------- split / run / config 元数据 ----------
    protocol = _infer_protocol(args, exp_out)
    test_per_class = [int((y_true == c).sum()) for c in range(num_classes)]
    split_info = {
        'protocol': protocol,
        'split_prefix': args.split_prefix,
        'dataset': args.dataset,
        'p': args.p_value, 'k': args.k_value, 'seed': args.seed,
        'train_num': train_num, 'val_num': val_num, 'test_num': test_num,
        'train_per_class': [int(x) for x in train_per_class],
        'test_per_class': test_per_class,
        'files': {
            'train_abs': f'train_abs{args.split_prefix}.txt',
            'test_abs': f'test_abs{args.split_prefix}.txt',
            'kfold_train': f'kfold_train{args.split_prefix}.txt',
            'kfold_val': f'kfold_val{args.split_prefix}.txt',
        },
    }
    run_info = {
        'dataset': args.dataset,
        'method_name': args.method_name,
        'protocol': protocol,
        'p': args.p_value, 'k': args.k_value, 'seed': args.seed,
        'groups': group_names,
        'rgb_cnn_in_ch': rgb_cnn_in_ch,
        'wavelet_cnn_in_chs': wavelet_cnn_in_chs,
        'backbone': args.backbone,
        'lr': args.lr, 'bs': args.bs, 'es': args.es,
        'loss': args.loss, 'no_plud': args.no_plud,
        'sampler': args.sampler, 'aug': args.aug,
        'plain_resnet': args.plain_resnet,
        'base_fusion': args.base_fusion,
        'dwt_cnn': getattr(args, 'dwt_cnn', False),
        'no_branch_refine': args.no_branch_refine,
        'no_att_fusion': args.no_att_fusion,
        'branch_gate': args.branch_gate,
        'static_branch_gate': args.static_branch_gate,
        'no_phys_init': args.no_phys_init,
        'group_att': args.group_att,
        'best_val_gmean': best_val_gmean,
        'best_epoch': best_epoch,
        'output_dir': exp_out,
    }
    config_dict = dict(vars(args))
    config_dict['_parsed_groups'] = group_names
    config_dict['_rgb_cnn_in_ch'] = rgb_cnn_in_ch
    config_dict['_wavelet_cnn_in_chs'] = wavelet_cnn_in_chs

    with open(os.path.join(exp_out, 'metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    with open(os.path.join(exp_out, 'split_info.json'), 'w', encoding='utf-8') as f:
        json.dump(split_info, f, indent=2, ensure_ascii=False)
    with open(os.path.join(exp_out, 'run_info.json'), 'w', encoding='utf-8') as f:
        json.dump(run_info, f, indent=2, ensure_ascii=False)
    with open(os.path.join(exp_out, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(config_dict, f, indent=2, ensure_ascii=False)

    print(f'✅ 规范工件已写入: {exp_out}')
