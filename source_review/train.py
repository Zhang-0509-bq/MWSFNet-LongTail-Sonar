# coding: utf-8
import argparse
import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
import random
from my_utils import *
import sys
from My_Dataset import MyDataset
from config import *
from paper_metrics import cal_nma_longtail, extract_features_during_training
from collections import Counter
from sklearn.metrics import f1_score
from run_artifacts import write_run_artifacts
def push_logit_loss(logit, gamma=1.):
    sigmoid = nn.Sigmoid()
    loss = sigmoid(gamma * logit)
    return loss.mean()

def train_mwsfnet_one_epoch(args, epoch, net, train_iter, criterion, optimizer, loss1_weight, loss2_weight, device='cuda:0', use_plud=True):
# def train_plud_one_epoch(args, epoch, net, train_iter, criterion, optimizer,  device='cuda:0'):
    net.train()
    train_loss = 0
    fc_layer = net.resnet18.fc

    # 接收3个输入 + 标签（而非单inputs）
    for batch_idx, (*X_parts, targets) in enumerate(train_iter):
        X = tuple(part.to(device) for part in X_parts)
        targets = targets.to(device)
        optimizer.zero_grad()

        outputs = net(*X)
        labels = targets

        loss1 = criterion(outputs, labels)
        if use_plud:
            # MWSFNet 特有：推高真类 logit、压低最大错误类 logit（源于 PLUD 论文的推间距项）
            true_logits = outputs[torch.arange(outputs.shape[0]), labels]
            mask = torch.ones_like(outputs, dtype=bool)
            mask[torch.arange(outputs.shape[0]), labels] = False
            max_wrong_logits = outputs[mask].view(outputs.shape[0], -1).max(dim=1)[0]
            right_close_logit = true_logits - max_wrong_logits
            loss2 = push_logit_loss(-right_close_logit)
            loss = (loss1 * loss1_weight) + (loss2 * loss2_weight)
        else:
            # 纯基线（CE/Focal/CB/LDAM）：只使用分类损失，不含推间距项
            loss = loss1

        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    print(f'----- Epoch {epoch} train loss is {train_loss} -----')
    return train_loss

def train_mwsfnet(args, net, train_iter, val_iter, test_iter, criterion, optimizer, scheduler, device='cuda:0', use_plud=True):
    net = net.to(device)
    FineTune_MAX_EPOCH = args.es

    # ----------------------早停与最优模型相关参数 -----------------------------
    best_val_gmean = -np.inf  # 记录验证集最好的Gmean(用Gmean更适合长尾数据) 负无穷，确保只有真正提升时才更新
    early_stop_count = 0  # 连续几轮验证集没提升的计数
    early_stop_patience = 100  # 容忍x轮没提升就停
    absolute_model_root = "../output/model"
    # 按数据集+方法名分目录存放，与 result/log/features 风格一致，不同方法互不覆盖
    best_model_dir = os.path.join(absolute_model_root, args.dataset, args.method_name)
    os.makedirs(best_model_dir, exist_ok=True)
    best_model_path = os.path.join(
        best_model_dir,
        f'best_model_p{args.p_value}_k{args.k_value}_seed{args.seed}.pth'
    )
    # ------------------------------------------------------------------------

    # 初始权重：loss1=0.8（分类），loss2=0.2（拉间距）
    init_loss1_w = 0.8
    init_loss2_w = 0.2
    # 目标权重：最后一个epoch，loss1=0.5，loss2=0.5（强化拉间距）
    target_loss1_w = 0.5
    target_loss2_w = 0.5

    # 训练日志保存
    log_dir = os.path.join('../output/log', args.dataset, args.method_name)
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f'log_p{args.p_value}_k{args.k_value}_seed{args.seed}.txt')
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write('epoch,train_loss,val_loss,val_gmean\n')

    for epoch in range(FineTune_MAX_EPOCH):
        # 动态计算损失权重
        ratio = epoch / FineTune_MAX_EPOCH  # 0→1，随epoch线性过渡
        current_loss1_w = init_loss1_w - (init_loss1_w - target_loss1_w) * ratio
        current_loss2_w = 1.0 - current_loss1_w  # 确保权重和为1

        train_loss = train_mwsfnet_one_epoch(
            args, epoch, net, train_iter, criterion, optimizer,
            current_loss1_w, current_loss2_w, device=device, use_plud=use_plud
        )

        # train_loss = train_mwsfnet_one_epoch(
        #     args, epoch, net, train_iter, criterion, optimizer,
        #     device=device
        # )

        scheduler.step()

        # ---------------------- 每轮训练后评估验证集 --------------------------
        # 用 my_utils.py 里的 evaluate_gmean 计算验证集效果（也可换 evaluate_accuracy）
        val_gmean = evaluate_gmean(val_iter, net, device=device)
        val_loss = evaluate_loss(val_iter, net, criterion, device)
        if isinstance(val_gmean, (tuple, list)):
            val_gmean = val_gmean[0]
        val_gmean = float(val_gmean)
        # 打印当前epoch的验证集效果
        print(f'===== Epoch {epoch+1}/{FineTune_MAX_EPOCH} | Val Gmean: {val_gmean:.4f} | Best Val Gmean: {best_val_gmean:.4f} =====')

         # 写入日志
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(f'{epoch},{train_loss:.4f},{val_loss:.4f},{val_gmean:.4f}\n')
        # ------------------------------------------------------------------------

        # ---------------------- ---早停判断与最优模型保存 ----------------------
        if val_gmean > best_val_gmean:
            # 验证集效果提升：更新最优Gmean、重置计数、保存当前模型
            best_val_gmean = val_gmean
            early_stop_count = 0
            torch.save(net.state_dict(), best_model_path)
            print(f'√ 保存最优模型到：{best_model_path}')
        else:
            # 验证集效果没提升：计数+1，判断是否早停
            early_stop_count += 1
            print(f'x 验证集无提升，已连续 {early_stop_count}/{early_stop_patience} 轮')
            if early_stop_count >= early_stop_patience:
                print(f'❗ 早停触发!停止训练(当前epoch: {epoch+1})')
                break  # 跳出训练循环，结束训练
        # ------------------------------------------------------------------------

    # ---------------------- ---训练结束后加载最优模型 ----------------------
    # 不管是早停还是跑完所有epoch，最后都加载最优模型用于测试
    if os.path.exists(best_model_path):
        net.load_state_dict(torch.load(best_model_path, weights_only=True))
        print(f'✅ 加载最优模型完成，准备评估测试集')

    # 新增：提取并保存测试集特征（目标层：ResNet18 layer4）
        save_feature_dir = os.path.join('../output/features', args.dataset, args.method_name)
        os.makedirs(save_feature_dir, exist_ok=True)
        feature_save_path = os.path.join(
            save_feature_dir,
            f'features_p{args.p_value}_k{args.k_value}_seed{args.seed}.npy'
        )
        label_save_path = os.path.join(
            save_feature_dir,
            f'labels_p{args.p_value}_k{args.k_value}_seed{args.seed}.npy'
        )

        # 调用特征提取函数（直接用模型的forward流程，无需复现）
        test_features, test_labels = extract_features_during_training(net, test_iter, device)
        np.save(feature_save_path, test_features)
        np.save(label_save_path, test_labels)
        print(f'✅ 测试集特征已保存至：{feature_save_path}')
        print(f'✅ 测试集标签已保存至：{label_save_path}')
    else:
        print(f'⚠️  未找到最优模型文件，使用最后一轮模型评估（可能过拟合）')
    # ------------------------------------------------------------------------
    return best_val_gmean

#  计算验证集损失
def evaluate_loss(data_iter, net, criterion, device):
    net.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in data_iter:
            *X_parts, targets = batch
            X = tuple(part.to(device) for part in X_parts)
            targets = targets.to(device)
            outputs = net(*X)
            loss = criterion(outputs, targets)
            total_loss += loss.item() * len(targets)
    return total_loss / len(data_iter.dataset)


def test_mwsfnet(net, testloader, device='cuda:0'):
    net.eval()
    scores, y_true = [], []
    with torch.no_grad():
        for batch_idx, (*X_parts, targets) in enumerate(testloader):
            # 所有输入迁移到设备
            X = tuple(part.to(device) for part in X_parts)
            targets = targets.to(device)

            y_true.append(targets)
            logits = net(*X)
            scores.append(logits)
    scores = torch.cat(scores,dim=0).cpu().numpy().tolist()
    y_true = torch.cat(y_true,dim=0).cpu().numpy().tolist()
    return scores, y_true


def test_mwsfnet_with_beta(net, testloader, device='cuda:0'):
    """测试集前向, 额外返回每个样本的 branch gate 权重 beta [N, n_branch]。
    仅 branch_gate 开启时有意义; 否则 beta 为 None。"""
    net.eval()
    scores, y_true, betas = [], [], []
    with torch.no_grad():
        for batch_idx, (*X_parts, targets) in enumerate(testloader):
            X = tuple(part.to(device) for part in X_parts)
            targets = targets.to(device)
            y_true.append(targets)
            logits, beta = net(*X, return_beta=True)
            scores.append(logits)
            if beta is not None:
                betas.append(beta)
    scores = torch.cat(scores, dim=0).cpu().numpy().tolist()
    y_true = torch.cat(y_true, dim=0).cpu().numpy().tolist()
    betas = torch.cat(betas, dim=0).cpu().numpy() if betas else None
    return scores, y_true, betas

def parse_args():
    # Set arg parser
    parser = argparse.ArgumentParser(description='MWSFNet training reference for forward-looking sonar classification')
    parser.add_argument('--dataset', default='NKSID', type=str, help='the dataset name')
    parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
    parser.add_argument('--backbone', default='resnet18', type=str, help='choosing network')
    parser.add_argument('--bs', default=32, type=int, help='batch size')
    parser.add_argument('--es', default=100, type=int, help='epoch size')
    parser.add_argument('--p_value', default=0, type=int, help='trail number of 5-trail 5-fold cross-validation')
    parser.add_argument('--k_value', default=0, type=int, help='fold number of 5-trail 5-fold cross-validation')
    parser.add_argument('--method_name', default='MWSFNet', type=str,
                        help='Method Name(结果目录名,如 MWSFNet / LDAM 等)')
    parser.add_argument('--loss', default='focal', type=str, choices=['ce', 'focal', 'cb', 'ldam'],
                        help='分类损失: ce / focal / cb / ldam (默认 focal,即 MWSFNet 使用的焦点损失)')
    parser.add_argument('--no_plud', action='store_true',
                        help='关闭 MWSFNet 的推间距(PLUD)项,用于纯基线比较(CE/Focal/CB/LDAM)')
    parser.add_argument('--save_results', default='True', type=str,
                        help='If save evaluation results to file')
    parser.add_argument('--save_models',action='store_true',
                        help='If save evaluation results to file')
    parser.add_argument('--sampler', type=str, default=None,
                       help='是否启用平衡采样,可选值:None 或 "balance"')
    parser.add_argument('--seed', default=0, type=int, help='全局随机种子（控制所有随机性）')
    parser.add_argument('--unfreeze_epoch', default=10, type=int,
                        help='[已弃用/no-op] 旧的渐进解冻轮次参数。冻结-解冻机制已移除, 保留此参数仅兼容既有调用, 不再影响训练')
    parser.add_argument('--group_att', action='store_true',
                        help='使用频带分组感知通道注意力(替代原始单一共享卷积通道注意力)。默认关闭,结果与原始 MWSFNet 一致;开启后用于消融/论文走图')
    parser.add_argument('--rgb_only', action='store_true',
                        help='仅保留 RGB 空域分支(去掉 3 种小波频域分支),即 RGB-only 基线/消融。wavelet_cnn_in_chs 置空,模型忽略小波输入,用于证明频域融合的贡献')
    parser.add_argument('--no_phys_init', action='store_true',
                        help='关闭物理引导初始化:小波通道走默认Kaiming随机(物理初始化消融基底)')
    parser.add_argument('--base_fusion', action='store_true',
                        help='使用 BaseFusionResNet18(15ch直连ResNet,无分支增强/无注意力/无自定义初始化)= "是否只是加参数/复杂度" 的最佳对照。与完整 MWSFNet 及 --rgb_only 配合做消融')
    parser.add_argument('--split_prefix', default='', type=str,
                        help='划分文件前缀。默认""=固定789协议; "_htpl"=HTPL-JB 2094/523 协议(读 test_abs_htpl/train_abs_htpl/kfold_*_htpl)')
    parser.add_argument('--aug', type=str, default='none', choices=['none', 'weak'],
                        help='数据增强模式: none=默认(仅flip+微jitter,与原始MWSFNet一致) / weak=针对相似类对(大小螺旋桨/浮筒-软管)的强增强: RandomResizedCrop尺度抖动+旋转+翻转+颜色抖动,作用于整张图使RGB与小波两分支空间对齐')
    parser.add_argument('--boost_classes', type=str, default='',
                        help='要过采样的弱类(逗号分隔的类别号;NKSID: 0=BigProport 3=Floats 5=SmallPropurt 6=SoftPipeline)。空=不过采样; 仅配合 --aug weak 使用时建议开启')
    parser.add_argument('--boost_factor', type=int, default=3,
                        help='对 boost_classes 每张训练图重复的倍数(过采样强度), 默认3')
    parser.add_argument('--groups', type=str, default=None,
                        help='输入分组(逗号分隔, 子集 of rgb,bior,rbio,sym4; 规范顺序 rgb->bior->rbio->sym4)。'
                             '默认 None=全组(RGB+3小波)。等价旧 --rgb_only 的是 --groups rgb。')
    parser.add_argument('--plain_resnet', action='store_true',
                        help='使用纯 torchvision ResNet18(3ch RGB, 无小波/无注意力/无物理初始化)。用于 ResNet18+CE/Focal/PLUD 基线')
    parser.add_argument('--no_branch_refine', action='store_true',
                        help='去掉分支增强(RGBCNN/WaveletCNN), 输入直接 concat 后走注意力(模块消融 B0)')
    parser.add_argument('--no_att_fusion', action='store_true',
                        help='去掉注意力融合(AttentionFusion), 分支增强后直接 dropout->ResNet18(模块消融 B1)')
    parser.add_argument('--branch_gate', action='store_true',
                        help='在分支增强后、concat 前插入 Softmax Branch-Adaptive Gate(sample-adaptive 分支加权),'
                             '与现有 DSAF 串联(BAF-MWSFNet)。默认关闭=原始 MWSFNet(可复现 A5 Full)')
    parser.add_argument('--static_branch_gate', action='store_true',
                        help='静态分支加权: 4 个全局共享可学习标量(softmax)对所有样本同一权重, 与 DSAF 串联。'
                             '用于对照"逐样本自适应是否必要"(若 SBW≈BAF 则样本自适应无增益)。与 --branch_gate 互斥')
    parser.add_argument('--dwt_cnn', action='store_true',
                        help='纯 DWT + ResNet18 baseline(学名 Sym4-DWT-ResNet18 / Wavelet-CNN Baseline): '
                             'grayscale→single-level DWT(默认 sym4)→LL/LH/HL/HH→[B,4,224,224]→ResNet18(conv1=4ch,fc=8)。'
                             '不含 WaveletCNN/RGBCNN/ChannelAttention/DSAF/BAF/SBW/multi-basis/Gaussian-Sobel 初始化。'
                             '要求 groups 恰好一个 wavelet(如 sym4)且不含 rgb')
    parser.add_argument('--exp_out', type=str, default='',
                        help='canonical 结果目录(写 metrics.json/*.npy/per_class/confusion/split_info 等规范工件)。空=不写')

    args = parser.parse_args()
    return args

def _expand_train_idx_for_boost(train_idx, label_path, boost_classes, boost_factor, old_new_lbl_map):
    """按类别过采样：对属于 boost_classes(新标签空间) 的训练索引重复 boost_factor 次。
    train_idx: 单折 int 列表; boost_classes: 集合{类别号}; 返回扩充后的索引列表。"""
    if not boost_classes or not label_path or boost_factor <= 1:
        return train_idx
    with open(label_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    labels = [int(l.split()[1]) for l in lines if len(l.split()) >= 2]
    new_idx = []
    for i in train_idx:
        new_idx.append(i)
        if old_new_lbl_map[labels[i]] in boost_classes:
            new_idx.extend([i] * (boost_factor - 1))
    return new_idx


if __name__=="__main__":
    dataset_list = DATASET_LIST
    data_root = DATA_ROOT
    old_new_lbl_map_dict = OLD_NEW_LBL_MAP_DICT

    args = parse_args()
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # 输入分组 -> 通道配置(唯一事实源)。--groups 优先, --rgb_only 等价 groups=rgb, 默认全组。
    groups = args.groups
    if groups is None and args.rgb_only:
        groups = 'rgb'
    rgb_cnn_in_ch, wavelet_cnn_in_chs, group_names = parse_groups(groups)

    # 保存本次运行的完整配置(超参数+seed+数据集划分+方法)，用于结果可复现/可追溯
    config_dir = os.path.join('../output/log', args.dataset, args.method_name)
    os.makedirs(config_dir, exist_ok=True)
    config_path = os.path.join(config_dir, f'config_p{args.p_value}_k{args.k_value}_seed{args.seed}.json')
    config_dict = dict(vars(args))
    config_dict['_parsed_groups'] = group_names
    config_dict['_rgb_cnn_in_ch'] = rgb_cnn_in_ch
    config_dict['_wavelet_cnn_in_chs'] = wavelet_cnn_in_chs
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config_dict, f, indent=2, ensure_ascii=False)
    print(f'✅ 运行配置已保存至：{config_path}')

    alpha_result_dict = get_multi_dataset_alpha(
        dataset_list=[args.dataset],
        data_root=data_root,
        old_new_lbl_map_dict=old_new_lbl_map_dict,
        seed=args.seed,
        p_value=args.p_value,
        k_value=args.k_value,
        split_prefix=args.split_prefix
    )
    # Loading data
    print('==> Preparing data..')
    p_v, k_v = args.p_value, args.k_value
    global_seed = args.seed
    init_random_seeds(global_seed)
    batch_size = args.bs
    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]
    data_dir = os.path.join('../data', args.dataset)
    def worker_init_fn(worker_id):
        # 子进程种子 = 全局种子 + 进程ID（避免多进程种子冲突）
        worker_seed = global_seed + worker_id
        torch.manual_seed(worker_seed)
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    dataset_classes_map = {
        'FLSMDD': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        'NKSID': [0, 1, 2, 3, 4, 5, 6, 7],
        # 'OtherDataset': [1, 3, 5, 7, 9]            # 新增其他数据集时
    }

    if args.dataset not in dataset_classes_map:
        raise ValueError(f"未配置数据集{args.dataset}的类别,请在dataset_classes_map中添加")
    all_classes = dataset_classes_map[args.dataset]
    train_class_num = len(all_classes)
    old_new_lbl_map = {cls: i for i, cls in enumerate(all_classes)}

    train_idx, val_idx = get_new_kfold_img_idx(p=p_v, k=k_v, data_dir=data_dir, split_prefix=args.split_prefix)
    # ---- 弱类过采样(针对 G-mean 最拖后腿的类) ----
    boost_classes = set(int(x) for x in args.boost_classes.split(',') if x.strip()) if args.boost_classes else set()
    if boost_classes:
        label_path = os.path.join(data_dir, f'train_abs{args.split_prefix}.txt')
        orig_count = len(train_idx)
        train_idx = _expand_train_idx_for_boost(train_idx, label_path, boost_classes, args.boost_factor, old_new_lbl_map)
        print(f'⚠️ 弱类过采样({sorted(boost_classes)}): 训练集 {orig_count} -> {len(train_idx)} 张(×{args.boost_factor})')
    train_iter, val_iter = get_kfold_img_iters(
        batch_size=batch_size,
        data_dir=data_dir,
        train_idx=train_idx,
        val_idx=val_idx,
        mean=mean,
        std=std,
        old_new_lbl_map=old_new_lbl_map,
        sampler=args.sampler,
        seed=global_seed,
        worker_init_fn=worker_init_fn,
        split_prefix=args.split_prefix,
        aug_mode=args.aug,
        groups=group_names
    )
    val_labels = []
    for batch in val_iter:
        *X_parts, targets = batch
        val_labels.extend(targets.numpy())
    print("验证集类别分布：", Counter(val_labels))

    # 加载独立测试集
    test_iter = get_test_iter(
        data_dir=data_dir,
        batch_size=batch_size,
        mean=mean,
        std=std,
        old_new_lbl_map=old_new_lbl_map,
        worker_init_fn=worker_init_fn,
        split_prefix=args.split_prefix,
        groups=group_names
    )
    # Building model
    print('==> Building model..')
    is_train = True
    backbone = args.backbone
    try:
        if args.plain_resnet:
            # 纯 torchvision ResNet18 (3ch RGB), 用于 ResNet18+CE/Focal/PLUD 基线
            if rgb_cnn_in_ch != 3 or len(wavelet_cnn_in_chs) != 0:
                raise ValueError('--plain_resnet 仅支持 RGB 输入(要求 groups 只含 rgb)')
            net = PlainResNet18(num_classes=train_class_num)
        elif args.dwt_cnn:
            # 纯 DWT + ResNet18 baseline: 单小波基(默认 sym4)4ch → ResNet18, 无 refinement/fusion/phys init
            if rgb_cnn_in_ch != 0 or len(wavelet_cnn_in_chs) != 1:
                raise ValueError('--dwt_cnn 仅支持单一小波基(不含 RGB), 要求 groups 恰好一个 wavelet(如 sym4)')
            net = DWTCNNResNet18(num_classes=train_class_num, wave_channels=wavelet_cnn_in_chs[0])
        elif args.base_fusion:
            # BaseFusionResNet18: 直连 ResNet,无分支增强/无注意力/无自定义初始化 = "是否只是加参数/复杂度"
            net = BaseFusionResNet18(num_classes=train_class_num, wavelet_cnn_in_chs=wavelet_cnn_in_chs, rgb_cnn_in_ch=rgb_cnn_in_ch)
        else:
            net = FusionResNet18(num_classes=train_class_num, group_att=args.group_att,
                                 wavelet_cnn_in_chs=wavelet_cnn_in_chs, rgb_cnn_in_ch=rgb_cnn_in_ch,
                                 use_phys_init=not args.no_phys_init,
                                 use_branch_refine=not args.no_branch_refine,
                                 use_att_fusion=not args.no_att_fusion,
                                 branch_gate=args.branch_gate,
                                 static_branch_gate=args.static_branch_gate)
    except Exception as e:
        print(f'THE INPUT BACKBONE {backbone} IS NOT EXIST!')
        print(f'错误详情：{str(e)}')
        sys.exit(1)

    # ========== 新增：配置类别映射（关键！适配你的数据集） ==========
    # 根据你的数据集调整：key=标签值，value=类别名称（B_Propeller对应你的标签值）
    if args.dataset == 'NKSID':
        class_label_map = {
            0: 'Big Propeller', 1: 'Cylinder', 2: 'Fishing Net',
            3: 'Floats', 4: 'Iron Pipeline', 5: 'Small Propeller',
            6: 'Soft Pipeline', 7: 'Tire'
        }
    elif args.dataset == 'FLSMDD':
        # 标签值 = 类别目录名的字母序索引(由 tool/generate_dir_lbl_list.py 生成,
        # 见 data/FLSMDD/{train,test}_abs.txt 第 2 列)。勿按语义重排,否则类名全错位。
        class_label_map = {
            0: 'Bottle', 1: 'Can', 2: 'Chain', 3: 'Drink Carton', 4: 'Hook',
            5: 'Propeller', 6: 'Shampoo Bottle', 7: 'Standing Bottle', 8: 'Tire',
            9: 'Valve'
        }
    else:
        raise ValueError(f"未配置{args.dataset}的类别映射！")


    # 直接指向嵌套的分类层
    fc_layer = net.resnet18.fc  # 获取分类层

    # Train Settings
    current_alpha, current_num_classes, current_class_counts = alpha_result_dict[args.dataset]
    # 通过 --loss 一键选择分类损失(ce/focal/cb/ldam)；--no_plud 关闭 MWSFNet 的推间距项(纯基线)
    criterion = build_loss(
        args.loss,
        num_classes=current_num_classes,
        alpha=current_alpha,
        class_counts=current_class_counts
    )
    optimizer = optim.SGD(
        net.parameters(),  # 分支+ResNet18参数一起优化
        lr=args.lr,
        momentum=0.9,  # 加momentum更稳定（ResNet训练常用）
        weight_decay=0.005
    )
    scheduler = MultiStepLR(optimizer, [35, 75], gamma=0.5)

    # Fix Random Seed for Stable Training
    init_random_seeds(global_seed)

    # 训练
    best_val_gmean = train_mwsfnet(args, net, train_iter, val_iter, test_iter, criterion, optimizer, scheduler, device=device, use_plud=not args.no_plud)

    # # ---------------------- 调用注意力可视化（核心添加部分）----------------------
    # # 定义可视化保存目录（和你的模型/结果保存目录结构一致）
    # vis_save_dir = os.path.join(
    #     '../output/visualization',
    #     args.dataset,
    #     args.method_name,
    #     f'p{args.p_value}_k{args.k_value}_seed{args.seed}'
    # )
    # # 调用函数（用测试集迭代器，结果更客观）
    # visualize_attention(
    #     data_iter=test_iter,
    #     net=net,
    #     device=device,
    #     save_dir=vis_save_dir,
    #     target_class='Big Propeller',  # 指定要可视化的难分类类别(NKSID 真实存在)
    #     class_label_map=class_label_map  # 新增：传入类别映射
    # )
    # # ----------------------------------------------------------------------------

    # Evaluation
    val_scores, val_y_true = test_mwsfnet(net, val_iter, device=device)
    if args.branch_gate or args.static_branch_gate:
        test_scores, y_true, test_betas = test_mwsfnet_with_beta(net, test_iter, device=device)
    else:
        test_scores, y_true = test_mwsfnet(net, test_iter, device=device)
        test_betas = None
    train_scores, train_y_true = test_mwsfnet(net, train_iter, device=device)

    # 计算三类结果的y_hat
    test_y_hat = np.argmax(test_scores, axis=1).tolist()
    val_y_hat = np.argmax(val_scores, axis=1).tolist()
    train_y_hat = np.argmax(train_scores, axis=1).tolist()


    # ==============================================================================

    # 计算测试集核心指标（适配长尾数据的关键指标）
    test_gmean = cal_gmean(y_true, test_y_hat)       # 几何平均召回率（重点）
    test_nma, _, _ = cal_nma_longtail(y_true, test_y_hat)           # 归一化平均准确率
    test_macro_f1 = f1_score(y_true, test_y_hat, average='macro')

    # 定义汇总文件路径（和之前 select_best_model.py 配置一致）
    absolute_model_root = "../output/model"  # 和 train_mwsfnet 中一致(相对 code/ 目录,按数据集+方法名分目录)
    summary_dir = os.path.join(absolute_model_root, args.dataset, args.method_name, "summary")
    os.makedirs(summary_dir, exist_ok=True)
    summary_path = os.path.join(
        summary_dir,
        f'{args.dataset}_{args.method_name}_cross_val_summary.txt'
    )

    # 5. 写入汇总信息（追加模式，不覆盖之前的组）
    with open(summary_path, 'a', encoding='utf-8') as f:
        # 列格式：p,k,seed,最优验证集G-mean,测试集G-mean,测试集NMA,最优模型路径
        f.write(
            f'{args.p_value},{args.k_value},{args.seed},{best_val_gmean:.4f},'
            f'{test_gmean:.4f},{test_nma:.4f},{test_macro_f1:.4f},'
            f'{os.path.join(absolute_model_root, args.dataset, args.method_name, f"best_model_p{args.p_value}_k{args.k_value}_seed{args.seed}.pth")}\n'
        )
    print(f"✅ 当前组性能已记录到汇总文件：{summary_path}")
    # ==============================================================================


    # 保存结果
    if args.save_results in ['True', 'true', True]:
        curr_dir = os.path.dirname(__file__)
        data_rename = f'{args.dataset}'
        save_dir = os.path.join(curr_dir, '../output/result', data_rename, args.method_name, backbone)
        os.makedirs(save_dir, exist_ok=True)

         #  测试集结果
        test_save_dir = save_dir
        os.makedirs(test_save_dir, exist_ok=True)
        write_result_to_file(
            base_dir=test_save_dir,
            data_type='test',
            y_hat=test_y_hat,
            y_true=y_true,
            logits=test_scores,
            p=p_v,
            k=k_v,
            seed=global_seed
        )

        #  验证集和训练集结果
        aux_results = [
            ('val', val_y_hat, val_y_true, val_scores),
            ('train', train_y_hat, train_y_true, train_scores)
        ]
        for data_type, y_hat, yt, lg in aux_results:
            aux_save_dir = save_dir
            os.makedirs(aux_save_dir, exist_ok=True)
            write_result_to_file(
                base_dir=aux_save_dir,
                data_type=data_type,
                y_hat=y_hat,
                y_true=yt,
                logits=lg,
                p=p_v,
                k=k_v,
                seed=global_seed
            )

    if args.save_models:
        curr_dir = os.path.dirname(__file__)
        model_dir = os.path.join(curr_dir, f'../output/model/{data_rename}/{args.method_name}/')
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        model_path = model_dir + f'p{p_v}_k{k_v}_seed{global_seed}_{backbone}_{args.method_name}.pth'
        torch.save(net.state_dict(), model_path)
        print(f'Model saved to {model_path} successfully!')

    # 写入 canonical 规范工件(仅当 --exp_out 指定; 不改动旧 output/ 体系)
    if args.exp_out:
        class_names = [class_label_map[i] for i in range(train_class_num)]
        write_run_artifacts(
            args,
            y_true=y_true, y_pred=test_y_hat, logits=test_scores,
            best_val_gmean=best_val_gmean,
            num_classes=train_class_num, class_names=class_names,
            group_names=group_names, rgb_cnn_in_ch=rgb_cnn_in_ch,
            wavelet_cnn_in_chs=wavelet_cnn_in_chs,
            train_num=len(train_idx), val_num=len(val_idx), test_num=len(y_true),
            train_per_class=current_class_counts,
            betas=test_betas,
        )
