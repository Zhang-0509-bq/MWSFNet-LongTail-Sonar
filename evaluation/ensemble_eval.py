# coding: utf-8
"""
ensemble_eval.py — 3-model(seed=0,1,2) 等权 probability soft-voting ensemble 评估。

复用 BAF ensemble 的完整逻辑, 通过参数定位任意模型家族的 run 目录。
只读 + 聚合, 不重新训练、不修改模型、不调整 voting 权重。

流程(与 BAF 版完全一致):
  1. 定位 {run_parent}/run_0{1,2,3} (seed=0,1,2)
  2. 读 sample_ids.txt / y_true.npy / logits.npy(softmax 得 probability, 并与 probabilities.npy 交叉校验)
  3. 硬校验: 样本数==523, sample_id 集合全等, y_true 一致, class_names 顺序一致; 顺序不同则按 sample_id 对齐
  4. p_ens=(p0+p1+p2)/3, y_pred=argmax(p_ens)
  5. G-mean / Macro-F1 / Accuracy / NMA / 混淆矩阵 / 逐类 precision/recall/F1/support
  6. 写 artifacts 到 --out_dir
  7. sanity check + 汇总表(对照常量) + 相对 --delta_ref 基线的 G-mean 差值(pp)

BAF 默认(不带参数)= 与原 ensemble_baf.py 一致。

用法:
  # A3 Sym4-only
  python ensemble_eval.py \
    --run_parent results/htpl20/representation/sym4_only \
    --out_dir results/htpl20/representation/a3_ensemble_3seed \
    --label "A3 Sym4-only" \
    --ref "A3 Sym4-only:95.84:95.56" "A5 Original Full:93.26:92.07" "BAF single-model:95.60:94.41" "SBW:90.68:89.31" "BAF 3-model ensemble:97.02:96.13" \
    --delta_ref "BAF 3-model ensemble(97.02)" "A3 Sym4-only(95.84)" "HTPL-JB reported(94.77)"
"""
import os
import json
import csv
import argparse
import numpy as np
from sklearn.metrics import (
    f1_score, accuracy_score, precision_recall_fscore_support,
    confusion_matrix,
)
from scipy.stats.mstats import gmean

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_ROOT = os.path.abspath(os.path.join(CODE_DIR, '..', 'results'))

SEEDS = [0, 1, 2]
PROTOCOL = 'htpl20'
TEST_SIZE = 523

# BAF 默认(与 ensemble_baf.py 完全一致)
BAF_RUN_PARENT = os.path.join(RESULTS_ROOT, 'htpl20', 'baf', 'baf_mwsfnet')
BAF_OUT_DIR = os.path.join(RESULTS_ROOT, 'htpl20', 'baf', 'baf_ensemble_3seed')

# 与 run 的 per_class_metrics.csv class_name 列一致(NKSID)
CLASS_NAMES = ['Big Propeller', 'Cylinder', 'Fishing Net', 'Floats',
               'Iron Pipeline', 'Small Propeller', 'Soft Pipeline', 'Tire']
N_CLASSES = len(CLASS_NAMES)


def softmax(z):
    z = np.asarray(z, dtype=np.float64)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def cal_gmean(y_true, y_pred):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    cm = confusion_matrix(y_true, y_pred)
    diag = np.diagonal(cm); n = cm.sum(axis=1)
    recall = diag[n > 0] / n[n > 0]
    return float(gmean(recall)) if len(recall) else 0.0


def load_member(seed, run_parent):
    rd = os.path.join(run_parent, f'run_{seed + 1:02d}')
    if not os.path.isdir(rd):
        raise FileNotFoundError(f'缺 run 目录: {rd}')
    logits_path = os.path.join(rd, 'logits.npy')
    prob_path = os.path.join(rd, 'probabilities.npy')
    yt_path = os.path.join(rd, 'y_true.npy')
    sid_path = os.path.join(rd, 'sample_ids.txt')
    for p in (logits_path, yt_path, sid_path):
        if not os.path.exists(p):
            raise FileNotFoundError(f'{seed} run 缺工件: {p}')
    logits = np.load(logits_path)
    probs = softmax(logits)
    if os.path.exists(prob_path):
        saved = np.load(prob_path).astype(np.float64)
        if saved.shape != probs.shape or not np.allclose(saved, probs, atol=1e-4):
            raise ValueError(f'{seed} probabilities.npy 与 logits 现算 softmax 不一致')
    y_true = np.load(yt_path).astype(np.int64)
    with open(sid_path, encoding='utf-8') as f:
        sample_ids = [ln.strip() for ln in f if ln.strip()]
    return {'seed': seed, 'run_dir': rd, 'sample_ids': sample_ids,
            'y_true': y_true, 'probs': probs}


def parse_ref(items):
    """'label:gmean:f1' 或 'label:gmean:f1:std' -> [{label,gmean,f1,std}]"""
    rows = []
    for it in items:
        parts = it.split(':')
        label = parts[0]; gmean = float(parts[1]); f1 = float(parts[2])
        std = float(parts[3]) if len(parts) > 3 else None
        rows.append({'label': label, 'gmean': gmean, 'f1': f1, 'std': std})
    return rows


def resolve_dir(p):
    """路径解析: 绝对路径直接用; 相对路径一律挂到 RESULTS_ROOT 下(所有结果都在 results/ 内)。"""
    if not p:
        return None
    p = os.path.expanduser(p)
    if os.path.isabs(p):
        return os.path.abspath(p)
    return os.path.abspath(os.path.join(RESULTS_ROOT, p))


def parse_delta(items):
    """'label(gmean)' -> [{label,gmean}]"""
    rows = []
    for it in items:
        i = it.rfind('(')
        label = it[:i].strip(); gmean = float(it[i + 1:-1])
        rows.append({'label': label, 'gmean': gmean})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run_parent', default=None, type=str)
    ap.add_argument('--out_dir', default=None, type=str)
    ap.add_argument('--label', default='BAF-MWSFNet', type=str)
    ap.add_argument('--ref', nargs='*', default=None,
                    help='对照表固定行, 每个 "label:gmean:f1[:std]"')
    ap.add_argument('--delta_ref', nargs='*', default=None,
                    help='delta 基线, 每个 "label(gmean)"')
    args = ap.parse_args()

    run_parent = resolve_dir(args.run_parent) if args.run_parent else BAF_RUN_PARENT
    out_dir = resolve_dir(args.out_dir) if args.out_dir else BAF_OUT_DIR

    # 默认对照常量 = 与 BAF 版一致(带参数时由 --ref 覆盖)
    ref = parse_ref(args.ref) if args.ref else parse_ref([
        'A3 Sym4 only:95.84:95.56', 'A5 Original Full:93.26:92.07',
        'BAF single-model:95.60:94.41', 'SBW:90.68:89.31',
        'BAF 3-model ensemble:97.02:96.13',
    ])
    delta_ref = parse_delta(args.delta_ref) if args.delta_ref else parse_delta([
        'BAF single-model mean(95.60)', 'A3 Sym4-only mean(95.84)', 'HTPL-JB reported(94.77)',
    ])

    members = [load_member(s, run_parent) for s in SEEDS]
    for m in members:
        if len(m['sample_ids']) != TEST_SIZE or len(m['y_true']) != TEST_SIZE or m['probs'].shape[0] != TEST_SIZE:
            raise SystemExit(f"seed{m['seed']} 样本数 != {TEST_SIZE}")
        if m['probs'].shape[1] != N_CLASSES:
            raise SystemExit(f"seed{m['seed']} class 数 != {N_CLASSES}")

    sid_sets = [set(m['sample_ids']) for m in members]
    if len(set(frozenset(s) for s in sid_sets)) != 1:
        raise SystemExit('三 run sample_id 集合不一致, 停止(不允许直接平均)')

    base_ids = members[0]['sample_ids']
    base_index = {sid: i for i, sid in enumerate(base_ids)}
    if len(base_index) != len(base_ids):
        raise SystemExit('sample_id 重复, 无法对齐')

    aligned = []
    for m in members:
        sid_to_row = {sid: i for i, sid in enumerate(m['sample_ids'])}
        y_true = np.zeros(TEST_SIZE, dtype=np.int64)
        probs = np.zeros((TEST_SIZE, N_CLASSES), dtype=np.float64)
        for j, sid in enumerate(base_ids):
            src = sid_to_row.get(sid)
            if src is None:
                raise SystemExit(f"seed{m['seed']} 缺失 sample_id: {sid}")
            y_true[j] = m['y_true'][src]; probs[j] = m['probs'][src]
        aligned.append({'seed': m['seed'], 'run_dir': m['run_dir'],
                        'sample_ids': base_ids, 'y_true': y_true, 'probs': probs})

    y0 = aligned[0]['y_true']
    for a in aligned[1:]:
        if not np.array_equal(a['y_true'], y0):
            raise SystemExit(f"y_true 跨 run 不一致(seed{a['seed']})")

    # class_names 顺序校验(仅当有 per_class_metrics.csv)
    pc0 = os.path.join(aligned[0]['run_dir'], 'per_class_metrics.csv')
    if os.path.exists(pc0):
        with open(pc0, newline='') as f:
            names = [r['class_name'] for r in csv.DictReader(f)]
        if names != CLASS_NAMES:
            raise SystemExit(f'class_names 顺序与预期不一致: {names}')

    p_ens = (aligned[0]['probs'] + aligned[1]['probs'] + aligned[2]['probs']) / 3.0
    y_ens = np.argmax(p_ens, axis=1)
    y_true = y0

    gmean_ens = cal_gmean(y_true, y_ens)
    macro_f1 = float(f1_score(y_true, y_ens, average='macro'))
    accuracy = float(accuracy_score(y_true, y_ens))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_ens, labels=list(range(N_CLASSES)), zero_division=0)
    cm = confusion_matrix(y_true, y_ens, labels=list(range(N_CLASSES)))
    macro_recall = float(np.mean(recall))
    nma = 0.5 * macro_recall + 0.5 * accuracy

    member_metrics = []
    for a in aligned:
        yp = np.argmax(a['probs'], axis=1)
        member_metrics.append({'seed': a['seed'], 'run_dir': a['run_dir'],
                               'gmean': cal_gmean(a['y_true'], yp),
                               'macro_f1': float(f1_score(a['y_true'], yp, average='macro'))})

    os.makedirs(out_dir, exist_ok=True)
    metrics = {'gmean': gmean_ens, 'macro_f1': macro_f1, 'accuracy': accuracy,
               'balanced_accuracy': macro_recall, 'macro_precision': float(np.mean(precision)),
               'macro_recall': macro_recall, 'nma': float(nma),
               'n_models': 3, 'voting': 'equal-weight probability soft voting'}

    with open(os.path.join(out_dir, 'ensemble_metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    with open(os.path.join(out_dir, 'ensemble_predictions.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['sample_id', 'true_label', 'pred_label'] + [f'prob_{c}' for c in range(N_CLASSES)])
        for i in range(TEST_SIZE):
            w.writerow([base_ids[i], int(y_true[i]), int(y_ens[i])] + [f'{x:.6f}' for x in p_ens[i]])
    with open(os.path.join(out_dir, 'ensemble_confusion_matrix.csv'), 'w', newline='') as f:
        w = csv.writer(f); w.writerow([''] + CLASS_NAMES)
        for r, row in enumerate(cm):
            w.writerow([CLASS_NAMES[r]] + [int(x) for x in row])
    with open(os.path.join(out_dir, 'ensemble_per_class.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['class', 'class_name', 'precision', 'recall', 'f1', 'support'])
        for c in range(N_CLASSES):
            w.writerow([c, CLASS_NAMES[c], precision[c], recall[c], f1[c], int(support[c])])
    with open(os.path.join(out_dir, 'ensemble_info.json'), 'w', encoding='utf-8') as f:
        json.dump({'member_seeds': SEEDS, 'member_runs': [a['run_dir'] for a in aligned],
                   'n_models': 3, 'voting': 'equal-weight probability soft voting',
                   'test_protocol': PROTOCOL, 'test_size': TEST_SIZE,
                   'class_names': CLASS_NAMES, 'n_classes': N_CLASSES,
                   'member_metrics': member_metrics,
                   'label': args.label, 'out_dir': out_dir}, f, indent=2, ensure_ascii=False)

    gmean_pct = gmean_ens * 100
    L = []
    L.append(f'# {args.label} 3-model soft-voting ensemble (htpl20, p0·k0, seeds 0/1/2)')
    L.append('')
    L.append('## 汇总对照表 (G-mean / Macro-F1)')
    L.append('')
    L.append('| Model | G-mean | Macro-F1 | note |')
    L.append('|---|---|---|---|')
    for r in ref:
        is_computed = r['label'] == args.label
        g = r['gmean']; f1_ = r['f1']
        gs = f'{g} ± {r["std"]}' if r['std'] is not None else f'{g}'
        note = 'computed' if is_computed else 'fixed'
        bold = '**' if is_computed else ''
        L.append(f'| {bold}{r["label"]}{bold} | {gs} | {f1_} | {note} |')
    L.append('')
    L.append('## ensemble 相对基线差 (G-mean, pp)')
    L.append('')
    L.append('| vs | ΔG-mean (pp) |')
    L.append('|---|---|')
    for d in delta_ref:
        L.append(f'| {d["label"]} | {gmean_pct - d["gmean"]:+.2f} |')
    L.append('')
    L.append('## 逐类 (ensemble)')
    L.append('')
    L.append('| class | precision | recall | f1 | support |')
    L.append('|---|---|---|---|---|')
    for c in range(N_CLASSES):
        L.append(f'| {c} {CLASS_NAMES[c]} | {precision[c] * 100:.2f} | {recall[c] * 100:.2f} '
                 f'| {f1[c] * 100:.2f} | {int(support[c])} |')
    L.append('')
    L.append('## sanity check: 成员单模型(现算)')
    L.append('')
    L.append('| seed | G-mean | Macro-F1 | run_dir |')
    L.append('|---|---|---|---|')
    for mm in member_metrics:
        L.append(f'| {mm["seed"]} | {mm["gmean"] * 100:.2f} | {mm["macro_f1"] * 100:.2f} | {mm["run_dir"]} |')
    L.append('')
    L.append('## 一致性校验')
    L.append(f'- test_size = {TEST_SIZE} (three runs all == {TEST_SIZE})')
    L.append('- sample_ids: 三 run 集合全等, 已按 sample_id 对齐(基准=seed0 顺序)')
    L.append('- y_true: 跨 run 完全一致')
    L.append('- class_names: 顺序一致(8 类)')
    L.append('- voting = equal-weight probability soft voting (不改权重, 不挑 seed)')

    with open(os.path.join(out_dir, 'ENSEMBLE_SUMMARY.md'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(L) + '\n')

    print('================ SANITY CHECK ================')
    for mm in member_metrics:
        print(f"member seed{mm['seed']}: G-mean {mm['gmean'] * 100:.2f}  Macro-F1 {mm['macro_f1'] * 100:.2f}")
    print(f"\n{args.label} 3-model ensemble: G-mean {gmean_ens * 100:.2f}  Macro-F1 {macro_f1 * 100:.2f}"
          f"  Acc {accuracy * 100:.2f}  NMA {nma * 100:.2f}")
    for d in delta_ref:
        print(f"  vs {d['label']}: {gmean_pct - d['gmean']:+.2f} pp")
    print(f'\n✅ artifacts 写入: {out_dir}')


if __name__ == '__main__':
    main()
