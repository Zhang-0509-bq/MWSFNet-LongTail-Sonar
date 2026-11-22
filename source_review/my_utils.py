# coding: utf-8
import os
import random
import time
import copy
import torch
from torch import nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from My_Dataset import MyDataset
import numpy as np
from scipy.stats.mstats import gmean
from sklearn.metrics import *
import torchvision.models as ptm
import pickle
from torchvision.models import (
    ResNet18_Weights, ResNet50_Weights,  # ResNet 系列
    AlexNet_Weights, VGG16_Weights,      # AlexNet、VGG 系列
    EfficientNet_B0_Weights              # EfficientNet 系列等
)
from collections import defaultdict

# 设定种子
def init_random_seeds(seed=0):
    # Fix Random Seed for Stable Training
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # 关键：无论 seed 是多少都强制确定性（小数据集，速度影响可忽略）
    # 否则 seed != 0 时 benchmark=True/deterministic=False，同一 seed 重跑结果都可能不同
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------- 输入分组解析(唯一事实源) ----------------------
# 规范顺序: rgb(有则) -> bior -> rbio -> sym4。数据集返回顺序与模型 conv1 通道顺序都以此为准。
GROUP_WAVELET = {'bior': 'bior3.7', 'rbio': 'rbio3.9', 'sym4': 'sym4'}
CANONICAL_GROUPS = ['rgb', 'bior', 'rbio', 'sym4']


def parse_groups(groups):
    """
    解析输入分组 -> (rgb_cnn_in_ch, wavelet_cnn_in_chs, group_names)。

    groups: None 或 '' = 默认全组(RGB + 3 小波, 与旧行为完全一致)；
            或逗号分隔字符串/列表, 子集 of ['rgb','bior','rbio','sym4']。
    返回:
      - rgb_cnn_in_ch: 3 或 0
      - wavelet_cnn_in_chs: 每基 4ch, 按 bior,rbio,sym4 顺序(仅选中者)
      - group_names: 规范顺序的组名列表(含 rgb, 若有)
    """
    if groups is None or (isinstance(groups, str) and groups.strip() == ''):
        groups = CANONICAL_GROUPS
    elif isinstance(groups, str):
        groups = [g.strip() for g in groups.split(',') if g.strip()]
    else:
        groups = list(groups)
    ordered = [g for g in CANONICAL_GROUPS if g in groups]
    if not ordered:
        raise ValueError(f"groups 至少需要一个有效组(可选 {CANONICAL_GROUPS}), 收到 {groups}")
    rgb_cnn_in_ch = 3 if 'rgb' in ordered else 0
    wavelet_cnn_in_chs = [4 for g in ordered if g in GROUP_WAVELET]
    return rgb_cnn_in_ch, wavelet_cnn_in_chs, ordered


def load_test_sample_ids(data_dir, split_prefix=''):
    """按 test_abs 文件行序返回测试样本的绝对路径(真实唯一 ID, 供 sample_ids 对齐/ensemble 校验)。"""
    test_abs_path = os.path.join(data_dir, f'test_abs{split_prefix}.txt')
    with open(test_abs_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    ids = []
    for line in lines:
        parts = line.split()
        if not parts:
            continue
        p = parts[0]
        if not os.path.isabs(p):
            p = p.replace('\\', '/')
            p = os.path.abspath(p)
        ids.append(p)
    return ids


class WaveletCNN(nn.Module):
    """小波特征增强CNN"""
    def __init__(self, in_channels=4):
        super().__init__()
        self.conv = nn.Sequential(
            # 深度可分离卷积 (3×3逐通道卷积 + 1×1跨通道融合)
            # 3×3逐通道卷积：单独捕捉每个小波通道的频域关联 (如LL通道的轮廓连续性)
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels),
            # 1×1跨通道卷积：融合4个小波通道的特征 (如LL与HH的互补信息)
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),

            # 第二步：1×1卷积微调 (保持原有结构,强化特征分布)
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels)
        )
        self.channel_att = ChannelAttention(in_channels=in_channels, reduction_ratio=2)
        self.residual_weight = nn.Parameter(torch.tensor(0.4))

    def forward(self, x):
        # x: [B,4,224,224] (小波特征)
        enhanced = self.conv(x)
        enhanced = enhanced * self.channel_att(enhanced)
        return x + self.residual_weight * enhanced  # 残差融合：原始特征为主,增强特征为辅

class RGBCNN(nn.Module):
    """RGB像素特征增强CNN"""
    def __init__(self, in_channels=3):
        super().__init__()
        self.conv = nn.Sequential(
            # 深度可分离卷积 (适配RGB空间特征)
            # 3×3逐通道卷积：单独提取每个颜色通道的纹理 (如红色通道的边缘)
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels),
            # 1×1跨通道卷积：融合RGB三通道的颜色关联 (如红+绿=黄的物体特征)
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),

            # 第二步：1×1卷积微调 (保持原有结构)
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels)
        )
        self.channel_att = ChannelAttention(in_channels=in_channels, reduction_ratio=3)

        self.residual_weight = nn.Parameter(torch.tensor(0.2))  # 残差权重

    def forward(self, x):
        # 轻量化增强：先提取单通道纹理,再融合三通道颜色信息
        enhanced = self.conv(x)
        enhanced = enhanced * self.channel_att(enhanced)
        return x + self.residual_weight * enhanced  # 原始像素 + 增强特征

 # 通道注意力
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=2):
        """
        Args:
            in_channels: 输入通道数(小波4通道/RGB 3通道)
            reduction_ratio: 通道压缩比(默认2,确保压缩后通道数≥1)
        """
        super().__init__()
        mid_channels = max(in_channels // reduction_ratio, 1)
        # 计算压缩后的通道数 (避免出现0或小数)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.shared_conv = nn.Sequential(  # 共享卷积 (减少参数量)
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, in_channels, kernel_size=1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 双池化结果相加,再通过共享卷积
        avg_out = self.shared_conv(self.avg_pool(x))
        max_out = self.shared_conv(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)  # 融合双池化的权重

class SubbandGroupChannelAttention(nn.Module):
    """
    频带分组感知通道注意力（用于论文走图 / 消融）。

    把融合后的通道按频带分组：[空间RGB, bior, rbio, sym4]，每组用一个独立的小
    1x1 卷积学习通道权重，显式体现"以频带为粒度"的注意力，区别于原始单一共享
    卷积的通道注意力。默认关闭（group_att=False 时仍走原始 AttentionFusion，结果不变）。
    """
    def __init__(self, in_channels=15, group_sizes=None, reduction=4):
        super().__init__()
        if group_sizes is None:
            group_sizes = [3, 4, 4, 4]  # 默认: RGB(3) + 3种小波各4通道
        self.group_sizes = [g for g in group_sizes if g > 0]
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # 每组一个独立 1x1 conv: [B,g,1,1] -> [B,g,1,1]
        self.group_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(g, max(g // reduction, 1), kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(max(g // reduction, 1), g, kernel_size=1),
            ) for g in self.group_sizes
        ])
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: [B, C, H, W] -> 全局平均池化 -> [B,C,1,1]
        pooled = self.avg_pool(x)
        parts = torch.split(pooled, self.group_sizes, dim=1)  # 按频带分组
        attended = [conv(p) for conv, p in zip(self.group_convs, parts)]
        w = self.sigmoid(torch.cat(attended, dim=1))  # [B,C,1,1]
        return w


class AttentionFusion(nn.Module):
    def __init__(self, in_channels=15, group_att=False, group_sizes=None):
        super().__init__()
        self.group_att = group_att
        if group_att:
            # 频带分组感知通道注意力（消融/走图用）
            self.base_channel_attention = SubbandGroupChannelAttention(in_channels, group_sizes)
        else:
            # 原始：单一共享卷积的通道注意力（默认，保持现有结果不变）
            self.base_channel_attention = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),  # [B,15,1,1] (全局平均池化)
                nn.Conv2d(in_channels, in_channels // 2, kernel_size=1),  # 降维压缩
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels // 2, in_channels, kernel_size=1),  # 恢复通道数
                nn.Sigmoid()  # 输出[0,1]的通道权重
            )

        #  多尺度空间注意力（捕捉不同尺寸目标，适配大/小螺旋桨）
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 4, kernel_size=3, padding=1),  # 输入：max_pool+avg_pool（2通道）→ 4通道
            nn.ReLU(inplace=True),
            nn.Conv2d(4, 4, kernel_size=5, padding=2),  # 大感受野卷积（捕捉大螺旋桨）
            nn.ReLU(inplace=True),
            nn.Conv2d(4, 1, kernel_size=1),  # 融合多尺度特征 → 1通道空间权重
            nn.Sigmoid()
        )
        nn.init.xavier_uniform_(self.spatial_attention[0].weight)
        nn.init.constant_(self.spatial_attention[0].bias, 0.1)  # 偏置初始化为0.1,避免前期权重过小

    def forward(self, x, is_visualize=False):
        # x: [B,15,224,224]
        # 1. 通道注意力加权
        batch_size = x.shape[0]

        # 基础通道权重（所有样本通用）
        base_channel_weights = self.base_channel_attention(x)  # [B,15,1,1]

        channel_weights = base_channel_weights

        # 通道加权：每个通道按权重增强/减弱
        x = x * channel_weights  # [B,15,224,224]

        # 2. 空间注意力加权
        max_pool = torch.max(x, dim=1, keepdim=True)[0]  # [B,1,224,224]
        avg_pool = torch.mean(x, dim=1, keepdim=True)    # [B,1,224,224]
        spatial_weights = self.spatial_attention(torch.cat([max_pool, avg_pool], dim=1))  # [B,1,224,224]
        x = x * spatial_weights  # [B,15,224,224] (每个空间位置乘对应权重)

        if is_visualize:
            return x, channel_weights, spatial_weights
        else:
            return x  # 训练/验证/测试时只返回融合特征（单个Tensor）


class BranchAdaptiveGate(nn.Module):
    """Softmax 分支自适应门控(BAF-MWSFNet 的核心新增)。

    输入 N 个"经过 branch refinement 后"的分支特征 [B,C_i,H,W]（各分支 channel 数可不同），
    输出按样本自适应加权后的同形状特征列表 + 门控权重 beta。

    三级注意力分工（互不混用）：
      - 本模块(branch gate)：选"哪个 representation 更可靠"      (branch 级)
      - 现有 DSAF channel attention：选"哪些 channels 更重要"   (channel 级)
      - 现有 DSAF spatial attention：选"哪些 spatial 位置更重要" (spatial 级)

    流程：
      1. 每分支 GAP -> [B,C_i]
      2. 每分支投影 Linear(C_i,d)+ReLU -> h_i ∈ R^d (d 小, 默认 8)
      3. concat -> [B,N*d]
      4. gating: Linear(N*d,hidden)+ReLU+Linear(hidden,N) -> logits
      5. beta = softmax(logits, dim=1)  (sum_j beta_j = 1)
      6. F'_i = beta_i * F_i

    关键初始化：末层 Linear 的 weight=0/bias=0，使初始 beta = 1/N（等权融合），
    训练中再学习 sample-adaptive 分支偏好。softmax 在均匀点的梯度非零，且
    d(logits_i)/d(W_i) = hidden ≠ 0，因此梯度不会被阻断，gate 从第一步即可学习。
    """
    def __init__(self, in_channels_list, d=8, hidden=16):
        super().__init__()
        assert len(in_channels_list) >= 2, 'BranchAdaptiveGate 至少需要 2 个分支'
        self.in_channels_list = list(in_channels_list)
        self.n_branches = len(in_channels_list)
        self.d = d

        # 每分支独立投影到统一 descriptor 维度 d
        self.projections = nn.ModuleList([
            nn.Sequential(nn.Linear(ci, d), nn.ReLU(inplace=True))
            for ci in in_channels_list
        ])
        # 轻量 gating network（末层即输出 branch logits）
        self.gate = nn.Sequential(
            nn.Linear(self.n_branches * d, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, self.n_branches),
        )
        # 关键初始化：末层 weight=0/bias=0 -> 初始 softmax=等权(1/N)
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)

    def forward(self, feats):
        # feats: list of [B,C_i,H,W]
        B = feats[0].shape[0]
        descs = []
        for proj, f in zip(self.projections, feats):
            g = F.adaptive_avg_pool2d(f, 1).view(B, -1)   # [B,C_i]
            descs.append(proj(g))                          # [B,d]
        h = torch.cat(descs, dim=1)                        # [B,N*d]
        logits = self.gate(h)                              # [B,N]
        beta = F.softmax(logits, dim=1)                    # [B,N], 逐样本 sum=1
        weighted = [beta[:, i:i + 1, None, None] * f for i, f in enumerate(feats)]
        return weighted, beta


class StaticBranchGate(nn.Module):
    """静态分支加权(Static Branch Weighting baseline)。

    与 BranchAdaptiveGate 的唯一区别：这里只有 N 个【全局共享】的可学习标量，经 softmax
    得到固定权重，对【所有样本】施加同一组权重，不做逐样本选择。用于对照"逐样本自适应
    是否必要"——若 SBW ≈ BAF，说明 BAF 的样本自适应并无增益，纯靠全局压低 RGB 即可恢复损失。

    流程：
      1. N 个全局标量 logits (nn.Parameter, init 0)
      2. beta = softmax(logits, dim=0) -> [N]，初始 = 1/N = 0.25
      3. beta broadcast 到 batch -> [B,N]
      4. F'_i = beta_i * F_i

    init=0 → softmax=等权，且在均匀点 softmax 梯度非零，可从第一步学习。
    最终这组 beta 就是 gate 学到的全局权重（全网共享，跨样本恒定）。
    """
    def __init__(self, n_branches):
        super().__init__()
        self.n_branches = n_branches
        self.logits = nn.Parameter(torch.zeros(n_branches))  # init 0 -> beta = 1/N

    def forward(self, feats):
        B = feats[0].shape[0]
        beta = F.softmax(self.logits, dim=0)              # [N] 静态权重
        beta_b = beta.unsqueeze(0).expand(B, -1)          # [B,N] 广播到 batch
        weighted = [beta_b[:, i:i + 1, None, None] * f for i, f in enumerate(feats)]
        return weighted, beta_b


class FusionResNet18(nn.Module):
    def __init__(self, num_classes, wavelet_cnn_in_chs=[4, 4, 4], rgb_cnn_in_ch=3, group_att=False,
                 use_phys_init=True, use_branch_refine=True, use_att_fusion=True, branch_gate=False,
                 static_branch_gate=False):
        super().__init__()
        self.rgb_cnn_in_ch = rgb_cnn_in_ch
        self.wavelet_cnn_in_chs = wavelet_cnn_in_chs
        self.num_classes = num_classes
        self.use_phys_init = use_phys_init
        self.use_branch_refine = use_branch_refine
        self.use_att_fusion = use_att_fusion
        if branch_gate and static_branch_gate:
            raise ValueError('--branch_gate 与 --static_branch_gate 只能选一个')
        self.use_branch_gate = branch_gate or static_branch_gate
        self.use_static_branch_gate = static_branch_gate
        if self.use_branch_gate and not use_branch_refine:
            raise ValueError('branch gating 需要分支增强(默认开启), 不能与 --no_branch_refine 同时使用')
        # 1. 动态创建特征增强CNN
        # wavelet_cnn_in_chs
        if use_branch_refine:
            self.wavelet_cnns = nn.ModuleList([WaveletCNN(in_channels=chs) for chs in wavelet_cnn_in_chs])
            self.rgb_cnn = RGBCNN(in_channels=rgb_cnn_in_ch) if rgb_cnn_in_ch > 0 else None
        else:
            # B0 消融: 去掉分支增强(RGBCNN/WaveletCNN), 输入直接 concat 后走注意力
            self.wavelet_cnns = nn.ModuleList()
            self.rgb_cnn = None
        self.dropout = nn.Dropout(p=0.3)

        # 注意力融合模块：传入num_classes
        total_in_channels = sum(wavelet_cnn_in_chs) + rgb_cnn_in_ch
        if use_att_fusion:
            self.attention = AttentionFusion(
                in_channels=total_in_channels,
                group_att=group_att,
                group_sizes=[rgb_cnn_in_ch] + list(wavelet_cnn_in_chs),
            )
        else:
            # B1 消融: 去掉注意力融合, concat 后直接 dropout -> ResNet18
            self.attention = None

        # Branch-Adaptive / Static Branch Gate(可选): 在 concat 前做分支加权, 与 DSAF 串联不替换。
        # in_channels_list 与 forward 里 enhanced_features 的顺序严格一致(rgb 若有 -> bior -> rbio -> sym4)。
        if static_branch_gate:
            # 静态: N 个全局共享标量(softmax), 对所有样本同一权重
            self.branch_gate = StaticBranchGate(len([c for c in ([rgb_cnn_in_ch] + list(wavelet_cnn_in_chs)) if c > 0]))
        elif branch_gate:
            # 自适应: 逐样本 GAP->投影->gating->softmax, 每样本不同权重
            gate_in_chs = [c for c in ([rgb_cnn_in_ch] + list(wavelet_cnn_in_chs)) if c > 0]
            self.branch_gate = BranchAdaptiveGate(gate_in_chs, d=8, hidden=16)
        else:
            self.branch_gate = None

        # 适配ResNet18的通道
        self.resnet18 = ptm.resnet18(weights=ResNet18_Weights.DEFAULT)
        original_conv1 = self.resnet18.conv1
        self.resnet18.conv1 = nn.Conv2d(
            in_channels=total_in_channels,  # 输入通道
            out_channels=original_conv1.out_channels,  # 保持64
            kernel_size=original_conv1.kernel_size,
            stride=original_conv1.stride,
            padding=original_conv1.padding,
            bias=original_conv1.bias
        )
        # 初始化新conv1权重
        with torch.no_grad():
            # 复用前3通道 (RGB增强特征)的预训练权重
            if rgb_cnn_in_ch > 0:
                self.resnet18.conv1.weight[:, :rgb_cnn_in_ch, :, :] = original_conv1.weight.clone()

            # 2. 小波通道分类型初始化 (3~14通道对应3种小波,每种4通道)
            #    use_phys_init=True: 高斯(低频)/Sobel(高频)重投影预训练conv1(物理引导)
            #    use_phys_init=False: 小波通道保持nn.Conv2d默认Kaiming随机(物理引导消融基底)
            if use_phys_init:
                rgb_weights = original_conv1.weight
                rgb_weight_scale = rgb_weights.abs().mean()  # 计算RGB权重的量级（用于统一小波权重）

                # 水平Sobel核 + 垂直Sobel核，捕捉全方位高频边缘
                sobel_h = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=torch.float32).view(1,1,3,3)
                sobel_v = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=torch.float32).view(1,1,3,3)
                sobel_kernel = torch.cat([sobel_h, sobel_v, (sobel_h+sobel_v)/2], dim=0)  # [3,1,3,3]（3个核，对应RGB 3通道）
                sobel_kernel = sobel_kernel / sobel_kernel.abs().sum(dim=(2,3), keepdim=True)  # 每个核归一化（避免量级过大）

                # ---------------------- 高斯核保持低频特性，归一化量级 ----------------------
                gaussian_1d = torch.exp(-(torch.arange(7)-3)**2 / (2*1.0**2))  # 1D高斯：[7]
                gaussian_2d = gaussian_1d.unsqueeze(1) * gaussian_1d.unsqueeze(0)  # 2D高斯：[7,7]
                gaussian_2d = gaussian_2d / gaussian_2d.sum()  # 核归一化
                gaussian_kernel = gaussian_2d.view(1,1,7,7).repeat(3,1,1,1)  # [3,1,7,7]（对应RGB 3通道）

                # ---------------------- 卷积后统一量级，确保和RGB权重一致 ----------------------
                # 生成高频权重（Sobel核卷积RGB核，捕捉边缘特征）
                high_freq_rgb_weights = F.conv2d(rgb_weights, sobel_kernel, padding=1, groups=3)  # [64,3,7,7]
                high_freq_rgb_weights = high_freq_rgb_weights * rgb_weight_scale / high_freq_rgb_weights.abs().mean()  # 统一量级

                # 生成低频权重（高斯核卷积RGB核，捕捉平滑特征）
                low_freq_rgb_weights = F.conv2d(rgb_weights, gaussian_kernel, padding=3, groups=3)  # [64,3,7,7]
                low_freq_rgb_weights = low_freq_rgb_weights * rgb_weight_scale / low_freq_rgb_weights.abs().mean()  # 统一量级

                # 动态初始化小波通道权重 (按wavelet_cnn_in_chs的数量循环)
                wavelet_channel_start = rgb_cnn_in_ch  # 小波通道从RGB通道后开始
                # 对称破缺：初始时所有LL通道共用同一个低频核、所有高频通道共用同一个高频核(仅两套核)，
                # 处于对称点会拖慢收敛；给每个通道的核加少量乘性高斯噪声打破这种对称性。
                sym_noise_std = 0.02
                for cnn_idx, chs in enumerate(wavelet_cnn_in_chs):
                    for ch_idx in range(chs):
                        current_channel = wavelet_channel_start + cnn_idx * 4 + ch_idx
                        if ch_idx == 0:  # LL通道用低频权重
                            base_kernel = low_freq_rgb_weights.mean(dim=1, keepdim=True)
                        else:  # LH/HL/HH通道用高频权重
                            base_kernel = high_freq_rgb_weights.mean(dim=1, keepdim=True)
                        # 乘性噪声：期望保持 base_kernel 不变，仅引入通道间方差以破对称(量级很小)
                        sym_broken_kernel = base_kernel * (1.0 + sym_noise_std * torch.randn_like(base_kernel))
                        self.resnet18.conv1.weight[:, current_channel:current_channel+1, :, :] = sym_broken_kernel
        # 分类层
        self.resnet18.fc = nn.Linear(self.resnet18.fc.in_features, num_classes)

# 随机初始化
        # with torch.no_grad():
        #     # 选项1：Kaiming初始化（推荐，适配ReLU激活）
        #     nn.init.kaiming_normal_(
        #         self.resnet18.conv1.weight,  # 仅初始化修改后的conv1层
        #         mode='fan_out',
        #         nonlinearity='relu'
        #     )

        #     if self.resnet18.conv1.bias is not None:
        #         nn.init.constant_(self.resnet18.conv1.bias, 0.0)
        # self.resnet18.fc = nn.Linear(self.resnet18.fc.in_features, num_classes)
        # nn.init.xavier_uniform_(self.resnet18.fc.weight)
        # nn.init.constant_(self.resnet18.fc.bias, 0.0)

    def forward(self, *args, is_visualize=False, return_beta=False):
        """
        动态接收特征参数:args[0]是RGB特征,args[1:]是所有小波特征
        """
        # 若rgb_cnn_in_ch=0(无RGB),则所有args都是小波特征
        if self.rgb_cnn_in_ch == 0:
            rgb_tensor = None  # RGB特征设为空
            wavelet_feats = args  # 所有输入都是小波特征
        else:
            rgb_tensor = args[0]
            wavelet_feats = args[1:]

        # 2. 统一增强所有特征
        enhanced_features = []
        if self.use_branch_refine:
            if rgb_tensor is not None:
                rgb_enhanced = self.rgb_cnn(rgb_tensor)
                enhanced_features.append(rgb_enhanced)
            for feat, cnn in zip(wavelet_feats, self.wavelet_cnns):
                enhanced_features.append(cnn(feat))
        else:
            # B0 消融: 不做分支增强, 直接用原始输入(后面统一 resize)
            if rgb_tensor is not None:
                enhanced_features.append(rgb_tensor)
            enhanced_features.extend(wavelet_feats)

        # 3. 统一Resize到224×224
        target_size = (224, 224)
        def resize_to_224(x):
            if x.shape[2:] != target_size:
                return F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
            return x
        enhanced_features = [resize_to_224(f) for f in enhanced_features]

        # 3.5 Branch-Adaptive Gate(可选): 逐样本分支加权, 再 concat(与 DSAF 串联)
        beta = None
        if self.use_branch_gate:
            enhanced_features, beta = self.branch_gate(enhanced_features)

        fused = torch.cat(enhanced_features, dim=1)  # [B,15,224,224]

        if self.attention is not None:
            if is_visualize:
                fused_att, channel_weights, spatial_weights = self.attention(fused, is_visualize=True)
            else:
                fused_att = self.attention(fused)  # 只接收融合特征（单个Tensor）
        else:
            # B1 消融: 无注意力融合
            fused_att = fused
            channel_weights, spatial_weights = None, None
        fused_att = self.dropout(fused_att)
        output = self.resnet18(fused_att)

        # 训练时返回分类结果，可视化时返回权重
        if is_visualize:
            return output, channel_weights, spatial_weights
        elif return_beta:
            return output, beta
        else:
            return output


class BaseFusionResNet18(nn.Module):
    def __init__(self, num_classes, wavelet_cnn_in_chs=[], rgb_cnn_in_ch=3):
        super().__init__()
        self.rgb_cnn_in_ch = rgb_cnn_in_ch
        self.wavelet_cnn_in_chs = wavelet_cnn_in_chs

        # 核心修改1：移除特征专属增强（WaveletCNN/RGBCNN）
        self.dropout = nn.Dropout(p=0.3)

        # 核心修改2：移除注意力融合模块
        self.total_in_channels = sum(wavelet_cnn_in_chs) + rgb_cnn_in_ch

        # 核心修改3：使用未修改的原始ResNet18（仅调整输入通道数，不做专属权重初始化）
        self.resnet18 = ptm.resnet18(weights=ResNet18_Weights.DEFAULT)
        original_conv1 = self.resnet18.conv1
        self.resnet18.conv1 = nn.Conv2d(
            in_channels=self.total_in_channels,  # 15通道输入（3小波+RGB）
            out_channels=original_conv1.out_channels,  # 保持64
            kernel_size=original_conv1.kernel_size,
            stride=original_conv1.stride,
            padding=original_conv1.padding,
            bias=original_conv1.bias
        )
        # 核心：不做任何专属权重初始化（不用预训练权重复用、不区分小波/RGB通道）
        # 分类层保持一致
        self.resnet18.fc = nn.Linear(self.resnet18.fc.in_features, num_classes)

    def forward(self, *args):
        """仅做特征拼接，不做增强和注意力融合"""
        if self.rgb_cnn_in_ch == 0:
            rgb_tensor = None
            wavelet_feats = args
        else:
            rgb_tensor = args[0]
            wavelet_feats = args[1:]

        # 不做特征增强，直接收集原始特征
        raw_features = []
        if rgb_tensor is not None:
            raw_features.append(rgb_tensor)  # 直接添加原始RGB特征
        for feat in wavelet_feats:
            raw_features.append(feat)  # 直接添加原始小波特征

        # 统一Resize
        target_size = (224, 224)
        def resize_to_224(x):
            if x.shape[2:] != target_size:
                return F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
            return x

        # 拼接特征（和实验组的拼接逻辑一致，仅无增强和注意力）
        resized_features = [resize_to_224(feat) for feat in raw_features]
        fused = torch.cat(resized_features, dim=1)  # shape: [B, 15, 224, 224]

        fused = self.dropout(fused)
        return self.resnet18(fused)


class PlainResNet18(nn.Module):
    """纯 torchvision ResNet18 (3ch RGB 输入), 仅换分类头。

    用于 ResNet18+CE / ResNet18+Focal / PLUD 基线(C0/C1/C2), 不含任何小波/注意力/物理初始化。
    暴露 .resnet18 以兼容取 fc_layer 与 layer4 特征提取的既有代码。
    """
    def __init__(self, num_classes):
        super().__init__()
        self.resnet18 = ptm.resnet18(weights=ResNet18_Weights.DEFAULT)
        self.resnet18.fc = nn.Linear(self.resnet18.fc.in_features, num_classes)

    def forward(self, x):
        return self.resnet18(x)


class DWTCNNResNet18(nn.Module):
    """纯 DWT + ResNet18 baseline（Sym4-DWT-ResNet18 / Wavelet-CNN Baseline）。

    输入：RGB sonar image → grayscale → single-level 2D DWT(默认 sym4)
        → LL/LH/HL/HH（复用 MyDataset.wavelet_transform 的标准化+resize）
        → concat 为 [B,4,224,224] → ResNet18(conv1=4ch, fc=8) → logits。

    严格不含 proposed 组件：无 WaveletCNN/LWFEB、无 RGBCNN、无 ChannelAttention、
    无 DSAF/AttentionFusion、无 BAF/SBW、无 multi-basis fusion、无 Gaussian/Sobel
    频率先验初始化、无渐进解冻。仅"纯 DWT + 预训练 ResNet18"。

    conv1 初始化（标准、简单、可解释的 4-channel pretrained init，非频率先验）：
        W        = 预训练 ResNet18 conv1.weight           [64,3,7,7]
        W_mean   = mean(W, dim=1, keepdim=True)           [64,1,7,7]
        W_4ch    = W_mean.repeat(1, wave_channels,1,1)     [64,K,7,7]
        conv1.weight = W_4ch * (3.0 / wave_channels)        # 幅值缩放, 4ch 求和≈原 3ch 求和
    """
    def __init__(self, num_classes, wave_channels=4):
        super().__init__()
        self.resnet18 = ptm.resnet18(weights=ResNet18_Weights.DEFAULT)
        original_conv1 = self.resnet18.conv1
        self.resnet18.conv1 = nn.Conv2d(
            in_channels=wave_channels,
            out_channels=original_conv1.out_channels,
            kernel_size=original_conv1.kernel_size,
            stride=original_conv1.stride,
            padding=original_conv1.padding,
            bias=False,
        )
        with torch.no_grad():
            W = original_conv1.weight.data                      # [64,3,7,7]
            W_mean = W.mean(dim=1, keepdim=True)                # [64,1,7,7]
            W_rep = W_mean.repeat(1, wave_channels, 1, 1)       # [64,4,7,7]
            # 幅值缩放：N 通道求和 ≈ 原 3 通道求和的期望幅值
            self.resnet18.conv1.weight.data.copy_(W_rep * (3.0 / wave_channels))
        self.resnet18.fc = nn.Linear(self.resnet18.fc.in_features, num_classes)

    def forward(self, x):
        return self.resnet18(x)


class FC(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(FC, self).__init__()
        self.fc = nn.Linear(num_inputs, num_outputs, bias=True)
    def forward(self, x):
        y = self.fc(x.view(x.shape[0], -1))
        return y

'''
Get fully connected layers
    @ num_in : number of input channels
    @ num_out : number of output channels
    @ num_fcs : number of fully connected layers
'''
def get_fcs(num_in, num_out, num_fcs, device=None):
    fcs = []
    for i in range(num_fcs):
        if device:
            fcs.append(FC(num_in, num_out).to(device))
        else:
            fcs.append(FC(num_in, num_out))
    return fcs

'''
Fix the output class number.
    @ model : pretrained model
    @ nb_classes : output class number
'''
def model_fc_fix(model, train_class_num, device=None):
    if hasattr(model, 'resnet18') and hasattr(model.resnet18, 'fc'):
        fc_layer_name = 'fc'
        fc_layer = getattr(model.resnet18, fc_layer_name)
        new_fc_layer = nn.Linear(fc_layer.in_features, train_class_num)
        setattr(model.resnet18, fc_layer_name, new_fc_layer)
    else:
        raise ValueError(f"模型 {model.__class__.__name__} 不包含ResNet18分类层,无法调整类别数")

    if device:
        model = model.to(device)

    return model
'''
Get k-fold image indexes.
    @ p : index of 10-trails
    @ k : index of 5-folds
    @ dataset : name of dataset, can be FLSMDD or NKSID
    @ data_dir: directory of dataset
    @ sample_type: sampling strategy, can be uniform or balance
'''
def get_kfold_img_idx(p, k, dataset, data_dir, sample_type=None):
    '''
    balance_num: set the balance number of each class in dataset
    class_slices: index range of every class for the dataset, like class 0: [0, 43], class 1: [43, 313]
    '''
    if sample_type == 'balance':
        if 'FLSMDD' in dataset:
            balance_num = 115
            class_slices = [0, 449, 816, 1042, 1391, 1524, 1661, 1760, 1825, 2156, 2364]
        elif 'NKSID' in dataset:
            balance_num = 108
            class_slices = [0, 203, 491, 511, 1462, 1574, 1668, 1783, 2617]
        else:
            print(f'ERROR! Dataset {dataset} is not exist!')
            return None

    kfold_train_path = os.path.join(data_dir, 'kfold_train.txt')
    kfold_val_path = os.path.join(data_dir, 'kfold_val.txt')
    f_train = open(kfold_train_path, 'r')
    f_val = open(kfold_val_path, 'r')
    train_lines = f_train.readlines()
    val_lines = f_val.readlines()
    train_line = None
    val_line = None
    for i in range(len(train_lines)):
        if train_lines[i][0] == '#':
            str_cuts = train_lines[i].rstrip().split('-')
            p_read = int(str_cuts[0][2:])
            k_read = int(str_cuts[1][1:])
            if p_read == p and k_read == k:
                train_line = train_lines[i+1]
                break
    for i in range(len(val_lines)):
        if val_lines[i][0] == '#':
            str_cuts = val_lines[i].rstrip().split('-')
            p_read = int(str_cuts[0][2:])
            k_read = int(str_cuts[1][1:])
            if p_read == p and k_read == k:
                val_line = val_lines[i+1]
                break
    train_line = [int(str) for str in train_line.rstrip().split(' ')]
    val_line = [int(str) for str in val_line.rstrip().split(' ')]
    if sample_type:
        if sample_type == 'uniform':
            train_line = [random.sample(train_line, 1)[0] for n in range(len(train_line))]
        elif sample_type == 'balance':
            balance_train_line = []
            train_line.sort()
            for i in range(len(class_slices) - 1):
                tmp_arr = [tr for tr in train_line if tr >= class_slices[i] and tr < class_slices[i+1]]
                balance_train_line = balance_train_line + [random.sample(tmp_arr, 1)[0] for n in range(balance_num)]
            train_line = balance_train_line
        else:
            print(f'ERROR! sample_type {sample_type} is not exist!')
            return None
    return train_line, val_line

'''
Get k-fold image data iters.
    @ batch_size : batch size
    @ data_dir : dataset directory
    @ train_idx : index list(s) of train images you want, like [[1,2,3],[4,5,6],...] or [1,2,3,...]
    @ val_idx : index list(s) of validation images you want
    @ mean : the mean of the ImageNet dataset
    @ std : the variance of the ImageNet dataset
    @ old_new_lbl_map_train : dictionary of training labels mapping infomation. {'old_label': new_label}, e.g., {'0': 3}
    @ old_new_lbl_map_test : dictionary of test labels mapping infomation
    @ sampler : if using class-balanced sampler, set this item as 'balance'
'''
def get_kfold_img_iters(batch_size, data_dir, train_idx, val_idx, mean, std, old_new_lbl_map=None, sampler=None, seed=0, worker_init_fn=None, generator=None, split_prefix='', aug_mode='none', groups=None):
    if sampler is not None:
        raise ValueError('Class-aware sampling is outside the paper configurations in this package.')
    normalize = transforms.Normalize(mean=mean,
                                     std=std)
    # ---- 增强模式 ----
    # 'none' : 默认(与原始 MWSFNet 完全一致)：仅 Resize+RandomHorizontalFlip+无效果RandomCrop+轻微ColorJitter。
    # 'weak' : 针对相似类对(大小螺旋桨/浮筒-软管)的"尺度+旋转"强增强。用 RandomResizedCrop 做尺度抖动(教模型区分
    #          大小螺旋桨尺寸)，配上随机旋转(声呐目标朝向任意)+翻转+颜色抖动，并通过 pil_transform 作用于"整张图"，
    #          让 RGB 与小波两分支共享同一张增强图，保证空间对齐。默认行为保持 None 不变。
    if aug_mode == 'weak':
        train_pil_transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.65, 1.0), ratio=(0.8, 1.25)),
            transforms.RandomRotation(degrees=20),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        ])
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])
    else:
        train_pil_transform = None
        train_transform = transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.RandomHorizontalFlip(p=0.5),
            # transforms.RandomRotation(90),
            transforms.RandomCrop(224),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            normalize
        ])
    val_transform = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        normalize
    ])

    abs_path = os.path.join(data_dir, f'train_abs{split_prefix}.txt')
    if isinstance(train_idx[0], list):
        train_iter = []
        for idx in train_idx:
            train_imgs = MyDataset(abs_path, idx,
                                transform=train_transform, pil_transform=train_pil_transform, old_new_lbl_map=old_new_lbl_map, input_size=(224, 224), groups=groups)
            samp = None
            shuf = True
            train_iter.append(torch.utils.data.DataLoader(train_imgs,
                                                batch_size=batch_size,
                                                sampler=samp,
                                                shuffle=shuf,
                                                num_workers=7,
                                                worker_init_fn=worker_init_fn,
                                                generator=generator
                                                ))
    elif isinstance(train_idx[0], int):
        train_imgs = MyDataset(abs_path, train_idx,
                                transform=train_transform, pil_transform=train_pil_transform, old_new_lbl_map=old_new_lbl_map, input_size=(224, 224), groups=groups)
        samp = None
        shuf = True
        train_iter = torch.utils.data.DataLoader(train_imgs,
                                            batch_size=batch_size,
                                            sampler=samp,
                                            shuffle=shuf,
                                            num_workers=7,
                                            worker_init_fn=worker_init_fn
                                            )

    val_imgs = MyDataset(abs_path, val_idx,
                                transform=val_transform, old_new_lbl_map=old_new_lbl_map, input_size=(224, 224), groups=groups)
    val_iter = torch.utils.data.DataLoader(val_imgs,
                                            batch_size=batch_size,
                                            num_workers=7,
                                            generator=generator)
    return train_iter, val_iter
'''
Evaluate models via accuracy metric.
    @ data_iter : validation data iter
    @ net : model to be evaluate
    @ device : inference device
'''
def evaluate_accuracy(data_iter, net, device=None):
    if device is None and isinstance(net, torch.nn.Module):
        device = list(net.parameters())[0].device
    acc_sum, n = 0.0, 0
    with torch.no_grad():
        for *X_parts, y in data_iter:
            X = tuple(part.to(device) for part in X_parts)
            y = y.to(device)
            net.eval()
            acc_sum += (net(*X).argmax(dim=1) == y).float().sum().cpu().item()
            net.train()  # 恢复训练模式 (若后续有需要)
            n += y.shape[0]
    return acc_sum / n
'''
Regular train function.
    @ train_iter : train data iter
    @ test_iter : test data iter
    @ net : model to be trained
    @ loss : loss function
    @ optimizer : training optimizer
    @ device : training device
    @ num_epochs : training epochs
'''
def train_plain(train_iter, test_iter, net, loss, optimizer, scheduler, device, num_epochs):
    net = net.to(device)
    print("training on ", device)
    batch_count = 0
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, start = 0.0, 0.0, 0, time.time()
        for *X_parts, y in train_iter:  # 接收3种特征+标签
            X = tuple(part.to(device) for part in X_parts)
            y = y.to(device)
            y_hat = net(*X) # 传入3种特征
            l = loss(y_hat, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_l_sum += l.cpu().item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]
            batch_count += 1
        scheduler.step()
        print('epoch %d, loss %.4f, lr %.4f train acc %.3f, time %.1f sec'
              % (epoch + 1, train_l_sum / batch_count, optimizer.state_dict()['param_groups'][0]['lr'], train_acc_sum / n, time.time() - start))

'''
Balanced muti-branch training function.
    @ train_iter : train data iter
    @ backbone : pretrained backbone of the transfer learning phase
    @ fc : fully connected layer
    @ loss : loss function
    @ optimizer : training optimizer
    @ device : training device
    @ num_epochs : training epochs
'''
def train_mul(train_iter, backbone, fc, loss, optimizer, scheduler, device, num_epochs):
    backbone = backbone.to(device)
    fc = fc.to(device)
    print("training on ", device)
    batch_count = 0
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, start = 0.0, 0.0, 0, time.time()
        backbone.eval()
        for *X_parts, y in train_iter:  # 接收多特征
            X = tuple(part.to(device) for part in X_parts)
            y = y.to(device)
            features = backbone(*X)  # 传入多特征
            y_hat = fc(features)
            l = loss(y_hat, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_l_sum += l.cpu().item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]
            batch_count += 1
        scheduler.step()
        print('epoch %d, loss %.4f, train acc %.3f, time %.1f sec'
              % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n, time.time() - start))
'''
Calculate y_hat for one batch data
    @ fcs : a list of fully connected layers to be evaluated
    @ features : features output by the backbone model
    @ method : fusion strategy, can be normal, hard_voting, soft_voting or weight_averaging
    @ device : inference device
    @ fusion_net : an abandoned function, you can overlook this parameter
    @ if_get_logits : if you want to get logits output, default is false
'''
def cal_y_hat(fcs, features, method, device, fusion_net = None, if_get_logits=False):
    y_hat = torch.tensor([]).to(device)
    if method == 'normal':
        logits = []
        for i in range(0,len(fcs)):
            if isinstance(fcs[i], torch.nn.Module):
                fcs[i].eval()
                logits.append(fcs[i](features).argmax(dim=1).tolist())
        y_hat = torch.tensor(logits).to(device)
    elif method == 'hard_voting':
        labels_sum = torch.tensor([0]).to(device)
        for i in range(0,len(fcs)):
            if isinstance(fcs[i], torch.nn.Module):
                fcs[i].eval()
                logits = fcs[i](features)
                labels_sum = labels_sum + F.one_hot(logits.argmax(dim=1), logits.shape[1])
        y_hat = labels_sum.argmax(dim=1)
    elif method == 'soft_voting':
        logits_sum = torch.tensor([0]).to(device)
        for i in range(0,len(fcs)):
            if isinstance(fcs[i], torch.nn.Module):
                fcs[i].eval()
                logits_sum = logits_sum + fcs[i](features)
        y_hat = logits_sum.argmax(dim=1)
    elif method == 'stacking':
        if fusion_net == None:
            print('ERROR! FUSION NET SHOULD NOT BE NONE!')
            return None
        logits = torch.tensor([]).to(device)
        for fc in fcs:
            fc.eval()  # open evaluate mode
            logits = torch.cat((logits, fc(features)), 1)
        fusion_net.eval() # open evaluate mode
        y_hat = fusion_net(logits).argmax(dim=1)
    elif method == 'weight_averaging':
        fus_param = copy.deepcopy(fcs[0].state_dict()) # fusion params of fcs
        for i in range(1, len(fcs)):
            fus_param['fc.weight'] += fcs[i].state_dict()['fc.weight']
            fus_param['fc.bias'] += fcs[i].state_dict()['fc.bias']
        fus_param['fc.weight'] /= len(fcs)
        fus_param['fc.bias'] /= len(fcs)
        fus_fc = copy.deepcopy(fcs[0]) # use deepcopy to incase fcs[0] be changed
        fus_fc.load_state_dict(fus_param)
        y_hat = fus_fc(features).argmax(dim=1)
        if if_get_logits:
            logits = fus_fc(features)
            return y_hat, logits
    else:
        print(f'ERROR! Method {method} is not exist!')
        return None
    return y_hat

'''
Calculate Gmean
    @ y_true : The true label vector, which should be shaped as : [0, 1, 2, 1...]
    @ y_pred : The predictive label vector, which should be shaped as : [0, 2, 1, 1...]
'''
def cal_gmean(y_true, y_pred):
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()
    conf_matrix = confusion_matrix(y_true, y_pred)
    diag = np.diagonal(conf_matrix) # right classifying number per class
    n_per_class = np.sum(conf_matrix, axis=1) # original number per class

    valid_mask = n_per_class != 0
    valid_diag = diag[valid_mask]
    valid_n = n_per_class[valid_mask]

    if len(valid_n) == 0:
        print("警告：所有类别在验证集中均无样本")
        return 0.0

    recall_per_class = valid_diag / valid_n
    gm = gmean(recall_per_class) if np.all(recall_per_class >= 0) else 0.0
    return gm

'''
The function is using for evaluate the gmean result of the baseline (deep transfer learning) method
    @ data_iter: iter of test/evaluate dataset.
    @ net : model to be evaluated.
    @ device : inference device
    @ if_get_y : if you want to get y_hat and y_true output, default is false
    @ if_get_logits : if you want to get logits output, default is false
'''
def evaluate_gmean(data_iter, net, device=None, if_get_y=False, if_get_logits=False):
    if device is None and isinstance(net, torch.nn.Module):
        device = list(net.parameters())[0].device
    y_true = torch.tensor([]).to(device)
    y_hat = torch.tensor([]).to(device)
    logits = torch.tensor([]).to(device)
    with torch.no_grad():
        for *X_parts,  y in data_iter:
            X = tuple(part.to(device) for part in X_parts)
            y = y.to(device)
            y_true = torch.cat((y_true, y), 0)
            if isinstance(net, torch.nn.Module):
                net.eval()
                output = net(*X)
                logits = torch.cat((logits, output), 0)
                y_hat = torch.cat((y_hat, output.argmax(dim=1)), 0)
        gmean_fc = cal_gmean(y_true, y_hat)
    if if_get_logits:
        return gmean_fc, y_hat, y_true, logits
    if if_get_y:
        return gmean_fc, y_hat, y_true
    return gmean_fc

'''
The function is using for evaluate the gmean result of BETL method
'''
def evaluate_gmean_optional(data_iter, backbone, fcs, method, fusion_net=None, device=None, if_get_y=False, if_get_logits=False):
    if device is None and isinstance(backbone, torch.nn.Module):
        device = list(backbone.parameters())[0].device
    y_hat = torch.tensor([]).to(device)
    y_true = torch.tensor([]).to(device)
    logits = torch.tensor([]).to(device)
    cat_dim = 0
    with torch.no_grad():
        for *X_parts, y in data_iter:  # 接收多特征
            X = tuple(part.to(device) for part in X_parts)
            y = y.to(device)
            y_true = torch.cat((y_true, y), 0)
            backbone.eval()
            features = backbone(*X)
            y_hat = torch.cat((y_hat, cal_y_hat(fcs, features, method, device, fusion_net)), cat_dim)
            if method == 'weight_averaging' and if_get_logits:
                _, lo = cal_y_hat(fcs, features, method, device, fusion_net, if_get_logits=True)
                logits = torch.cat((logits, lo), cat_dim)
        if method == 'normal':
            gmean = [cal_gmean(y_true, y_hat[i]) for i in range(len(y_hat))]
        else:
            gmean = cal_gmean(y_true, y_hat)
    if method == 'weight_averaging' and if_get_logits:
        return gmean, y_hat, y_true, logits
    if if_get_y:
        return gmean, y_hat, y_true
    return gmean

'''
The function is using for evaluate the overall accuracy result of BETL method
'''
def evaluate_acc_optional(data_iter, backbone, fcs, method, fusion_net=None, device=None):
    if device is None and isinstance(backbone, torch.nn.Module):
        device = list(backbone.parameters())[0].device
    y_hat = torch.tensor([]).to(device)
    y_true = torch.tensor([]).to(device)
    cat_dim = 0
    with torch.no_grad():
        for *X_parts, y in data_iter:  # 接收多特征
            X = tuple(part.to(device) for part in X_parts)
            y = y.to(device)
            y_true = torch.cat((y_true, y), 0)
            backbone.eval()
            features = backbone(*X)
            y_hat = torch.cat((y_hat, cal_y_hat(fcs, features, method, device, fusion_net)), cat_dim)
        acc_sum = (y_hat == y_true.to(device)).float().sum().cpu().item()
        acc_ave = acc_sum / len(y_true)
    return acc_ave

'''
The function is using for ensemble pruning in the third phase of BETL
    @ train_iter: iter of training dataset.
    @ backbone: model without the last linear.
    @ fcs: a list of full connected layers.
    @ _p: pruning proportion
    @ use_single_metric: using single metric to calculate the order of fcs, could be 'gmean' or 'acc'.
'''

'''
Get a fusion fully connected layer by weight averaging from a list of full connected layers
    @ fcs: a list of full connected layers.
'''
def get_fusion_fc(fcs):
    fus_param = copy.deepcopy(fcs[0].state_dict()) # fusion params of fcs
    for i in range(1, len(fcs)):
        fus_param['fc.weight'] += fcs[i].state_dict()['fc.weight']
        fus_param['fc.bias'] += fcs[i].state_dict()['fc.bias']
    fus_param['fc.weight'] /= len(fcs)
    fus_param['fc.bias'] /= len(fcs)
    fus_fc = copy.deepcopy(fcs[0]) # use deepcopy to incase fcs[0] be changed
    fus_fc.load_state_dict(fus_param)
    fus_fc.eval()
    return fus_fc

'''
Write y_hat, y_true, logits results to file
    @ base_dir: base directory for save folder (../output/result/dataset/method/backbone)
    @ data_type: type of data, 'train' or 'test'
    @ y_hat: the predict labels of classification model
    @ y_true: the ground truth labels of validation data
    @ logits: the predict scores of classification model
    @ close_y_hat: the predict scores of closed-set evaluation
    @ p : index of 10-trails
    @ k : index of 5-folds
'''
def write_result_to_file(base_dir, data_type, y_hat, y_true, logits=None, p=0, k=0, seed=0):
    file_dir = os.path.join(base_dir,data_type)
    os.makedirs(file_dir, exist_ok=True)

    y_hat_path = os.path.join(file_dir, f'y_hat_p{p}_k{k}_seed{seed}.txt')
    y_true_path = os.path.join(file_dir, f'y_true_p{p}_k{k}_seed{seed}.txt')
    f_y_hat = open(y_hat_path, 'w')
    f_y_true = open(y_true_path, 'w')
    if isinstance(y_hat, torch.Tensor):
        y_hat = y_hat.cpu().numpy()
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()

    for y_h in y_hat:
        f_y_hat.write(str(y_h) + ' ')
    f_y_hat.write('\n')
    f_y_hat.close()

    for y_t in y_true:
        f_y_true.write(str(y_t) + ' ')
    f_y_true.write('\n')
    f_y_true.close()

    if logits is not None and len(logits):
        logits_path = os.path.join(file_dir, f'logits_p{p}_k{k}_seed{seed}.txt')
        f_logits = open(logits_path, 'w')
        if isinstance(logits, torch.Tensor):
            logits = logits.cpu().numpy()
        logits = np.around(logits, 2)
        for lo in logits:
            for l in lo:
                f_logits.write(str(l) + ' ')
            f_logits.write(',')
        f_logits.write('\n')
        f_logits.close()

    print(f'结果已保存至：{file_dir} (seed={seed})')

'''
Read y_hat, y_true, logits results from file
    @ base_dir: base directory of save folder (../output/result/dataset/method/backbone)
    @ data_type: type of data to read, 'train' or 'test'
    @ p : index of 5-trails
    @ k : index of 5-folds
    @ if_get_logits : if you want to read logits, default is false
    @ if_get_close_y_hat : if you want to read closed-set preds, default is false
'''
def read_result_from_file(base_dir, data_type, p, k, seed, if_get_logits=False, if_get_close_y_hat=False):
    file_dir = os.path.join(base_dir, data_type)
    y_hat_path = os.path.join(file_dir, f'y_hat_p{p}_k{k}_seed{seed}.txt')
    y_true_path = os.path.join(file_dir, f'y_true_p{p}_k{k}_seed{seed}.txt')

    if not os.path.exists(y_hat_path):
        raise FileNotFoundError(f"未找到y_hat文件:{y_hat_path}")
    if not os.path.exists(y_true_path):
        raise FileNotFoundError(f"未找到y_true文件:{y_true_path}")

    with open(y_hat_path, 'r', encoding='utf-8') as f:
        y_hat_line = f.readline().strip()
    with open(y_true_path, 'r', encoding='utf-8') as f:
        y_true_line = f.readline().strip()

    y_hat = [int(x) for x in y_hat_line.split() if x]
    y_true = [int(x) for x in y_true_line.split() if x]

    if if_get_logits:
        logits_path = os.path.join(file_dir, f'logits_p{p}_k{k}_seed{seed}.txt')
        if not os.path.exists(logits_path):
            raise FileNotFoundError(f"未找到logits文件:{logits_path}")
        with open(logits_path, 'r', encoding='utf-8') as f:
            logits_line = f.readline().strip()
        logits = []
        for part in logits_line.split(','):
            if part.strip():
                logits.append([float(x) for x in part.strip().split() if x])
        return y_hat, y_true, logits

    if if_get_close_y_hat:
        close_y_hat_path = os.path.join(file_dir, f'close_y_hat_p{p}_k{k}_seed{seed}.txt')
        if not os.path.exists(close_y_hat_path):
            raise FileNotFoundError(f"未找到close_y_hat文件:{close_y_hat_path}")
        with open(close_y_hat_path, 'r', encoding='utf-8') as f:
            close_y_hat_line = f.readline().strip()
        close_y_hat = [int(x) for x in close_y_hat_line.split() if x]
        return y_hat, y_true, close_y_hat

    return y_hat, y_true


'''
Save a tensor as a image
    @ img_tensor: tensor type data
    @ save_dir : image saving directory
'''
def save_tensor_as_img(img_tensor, save_dir):
    from  torchvision import utils as vutils
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for i in range(img_tensor.shape[0]):
        img_dir = save_dir + f'/img_{i}.jpg'
        vutils.save_image(img_tensor[i], img_dir, normalize=True)

'''
Write data to a pikle file
    @ file_dir: file save directory
    @ data : data you want to save
    @ p : index of 5-trails
    @ k : index of 5-folds
'''
def write_data_to_pkl(file_dir, data, p=0, k=0):
    assert os.path.isdir(file_dir)
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    file_path = os.path.join(file_dir, f'p{p}-k{k}.pkl')
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)
    print(f'Write data to file {file_path} finished!')

'''
Read data from a pikle file
    @ file_dir: pikle file directory
    @ p : index of 5-trails
    @ k : index of 5-folds
'''
def read_data_from_pkl(file_dir, p=0, k=0):
    assert os.path.isdir(file_dir)
    file_path = os.path.join(file_dir, f'p{p}-k{k}.pkl')
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    print(f'Load data from file {file_path} finished!')
    return data


def get_new_kfold_img_idx(p, k, data_dir, split_prefix=''):
    """
    从新生成的 kfold_train.txt/kfold_val.txt 中读取指定 trail 和 fold 的索引
    p: 第p次实验(0~9)
    k: 第k折(0~4)
    data_dir: 存储 kfold 文件的目录(如 ../data/FLSMDD)
    split_prefix: 划分文件前缀(如 ''=默认固定789, '_htpl'=HTPL-JB 2094/523 协议)
    """
    kfold_train_path = os.path.join(data_dir, f'kfold_train{split_prefix}.txt')
    kfold_val_path = os.path.join(data_dir, f'kfold_val{split_prefix}.txt')

    # 读取训练集索引
    with open(kfold_train_path, 'r', encoding='utf-8') as f:
        train_lines = f.readlines()
    train_idx = None
    for i in range(len(train_lines)):
        if train_lines[i].startswith(f'#p{p}-k{k}'):
            train_idx = [int(x) for x in train_lines[i+1].strip().split()]
            break

    # 读取验证集索引
    with open(kfold_val_path, 'r', encoding='utf-8') as f:
        val_lines = f.readlines()
    val_idx = None
    for i in range(len(val_lines)):
        if val_lines[i].startswith(f'#p{p}-k{k}'):
            val_idx = [int(x) for x in val_lines[i+1].strip().split()]
            break

    if train_idx is None or val_idx is None:
        raise ValueError(f"未找到 p{p}-k{k} 的索引,请检查文件 {kfold_train_path}")
    return train_idx, val_idx

# 加载独立测试集
def get_test_iter(data_dir, batch_size, mean, std, old_new_lbl_map=None,  worker_init_fn=None, generator=None, split_prefix='', groups=None):
    """从 test_abs.txt 加载测试集(含已知+未知类)。split_prefix: '_htpl' 则读 test_abs_htpl.txt"""
    test_transform = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    test_abs_path = os.path.join(data_dir, f'test_abs{split_prefix}.txt')
    with open(test_abs_path, 'r', encoding='utf-8') as f:
        test_indices = list(range(len(f.readlines())))

    test_imgs = MyDataset(
        test_abs_path,
        test_indices,
        transform=test_transform,
        old_new_lbl_map=old_new_lbl_map,
        groups=groups
    )
    test_iter = torch.utils.data.DataLoader(
        test_imgs,
        batch_size=batch_size,
        shuffle=False,
        num_workers=7,
        worker_init_fn=worker_init_fn,
        generator=generator
    )
    return test_iter

'''
Focal Loss
'''

class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean', num_classes=None):
        """
        多分类 Focal Loss
        Args:
            alpha: 类别平衡系数 (默认1.0,无平衡；若类别不平衡,可传入与类别数一致的列表)
            gamma: 难样本聚焦系数 (默认2.0,gamma越大,越聚焦难样本)
            reduction: 损失聚合方式 ('mean'/'sum',默认'mean',返回平均损失)
            num_classes: 总类别数 (若alpha为列表,需确保长度等于num_classes)
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

        # 处理 alpha (若为列表,转为张量便于计算)
        if isinstance(alpha, list) and num_classes is not None:
            assert len(alpha) == num_classes, "alpha列表长度需与类别数一致"
            self.alpha = torch.tensor(alpha, dtype=torch.float32)
        elif not isinstance(alpha, (int, float, torch.Tensor)):
            raise ValueError("alpha需为int/float/list/torch.Tensor")

    def forward(self, logits, targets):
        """
        Args:
            logits: 模型输出(未经过softmax,shape=[B, num_classes])
            targets: 真实标签(整数标签,shape=[B])
        Returns:
            focal_loss: 计算后的Focal Loss
        """
        # 1. 计算 softmax 概率 (获取每个类别的预测概率)
        softmax_probs = F.softmax(logits, dim=1)  # shape=[B, num_classes]
        # 2. 提取真实类别的概率 p_t (shape=[B])
        p_t = softmax_probs.gather(1, targets.unsqueeze(1)).squeeze(1)  # 按真实标签索引取概率

        # 3. 计算 Focal Loss 核心项：-alpha * (1-p_t)^gamma * log(p_t)
        # 处理 alpha (若为张量,需与targets设备一致)
        if isinstance(self.alpha, torch.Tensor):
            self.alpha = self.alpha.to(targets.device)
            alpha_t = self.alpha[targets]  # 每个样本对应类别的alpha
        else:
            alpha_t = self.alpha  # 所有样本用相同的alpha

        # 计算损失 (加1e-12避免log(0)导致NaN)
        loss = -alpha_t * (1 - p_t + 1e-12) ** self.gamma * torch.log(p_t + 1e-12)

        # 4. 按需求聚合损失
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

# ---------------------- 长尾损失扩展（CB-Focal / LDAM，可插拔） ----------------------

class CBFocalLoss(nn.Module):
    """
    Class-Balanced Focal Loss（类别平衡焦点损失），Cui et al., CVPR 2019.

    用"有效样本数"(effective number) 代替频率来做类别平衡，避免尾部类别权重过大导致过拟合。
    Args:
        cls_num_list: 每类训练样本数列表
        beta: 有效样本数衰减系数（默认0.999，越大越强调稀有类）
        gamma: 难样本聚焦系数
        reduction: 'mean'/'sum'/None
    """
    def __init__(self, cls_num_list, beta=0.999, gamma=2.0, reduction='mean'):
        super(CBFocalLoss, self).__init__()
        cls_num_list = np.asarray(cls_num_list, dtype=np.float64)
        cls_num_list = np.maximum(cls_num_list, 1.0)  # 防止0样本类权重异常
        effective_num = 1.0 - np.power(beta, cls_num_list)
        per_cls_weights = (1.0 - beta) / (effective_num + 1e-6)
        # 归一化权重（使权重和为 num_classes，避免整体损失被过度放大）
        per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
        self.cls_weights = torch.tensor(per_cls_weights, dtype=torch.float32)
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        softmax_probs = F.softmax(logits, dim=1)
        p_t = softmax_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        w_t = self.cls_weights.to(logits.device)[targets]
        loss = -w_t * (1 - p_t + 1e-12) ** self.gamma * torch.log(p_t + 1e-12)
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class LDAMLoss(nn.Module):
    """
    LDAM Loss（Label-Distribution-Aware Margin Loss），Cao et al., CVPR 2019.

    为稀有类分配更大的间隔，迫使网络给出更高的真类 logit，从而拉宽决策边界并改善长尾校准。
    负真类 logit（x_m = 真类logit - margin）使正确分类变难，从而学习到更大的类间余量。
    Args:
        cls_num_list: 每类训练样本数列表
        max_m: 最大间隔
        s: logit 缩放因子（softmax 温度）
        weight: 可选的类别权重（用于 LDAM-DRW 的重新加权阶段）
        reduction: 'mean'/'sum'/None
    """
    def __init__(self, cls_num_list, max_m=0.5, s=30, weight=None, reduction='mean'):
        super(LDAMLoss, self).__init__()
        cls_num_list = np.asarray(cls_num_list, dtype=np.float64)
        # 防止个别类样本数为0导致 margin=inf/NaN（交叉验证某折尾部类可能为空）
        cls_num_list = np.maximum(cls_num_list, 1.0)
        m_list = 1.0 / np.power(cls_num_list, 0.25)
        m_list = m_list * (max_m / np.max(m_list))   # 归一化到 [0, max_m]
        self.m_list = torch.tensor(m_list, dtype=torch.float32)
        self.s = s
        self.weight = weight
        self.reduction = reduction

    def forward(self, logits, targets):
        device = logits.device
        m_list = self.m_list.to(device)
        # 标记真实类别位置
        index = torch.zeros_like(logits, dtype=torch.bool)
        index.scatter_(1, targets.unsqueeze(1), True)
        # 每个样本为其真实类别取对应 margin
        batch_m = torch.matmul(m_list[None, :], index.float().transpose(0, 1)).view(-1, 1)
        # 仅从真实类别 logit 减去 margin
        x_m = logits - index.float() * batch_m
        # 缩放并计算交叉熵（可选 weight 实现 DRW 重训练阶段）
        loss = F.cross_entropy(self.s * x_m, targets, weight=self.weight, reduction='none')
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


# ---------------------- 损失注册表 ----------------------
# 通过 "--loss {ce|focal|cb|ldam}" 一键切换分类损失，无需改动架构 / 数据 / 训练流程。
LOSS_REGISTRY = {
    'ce':    lambda num_classes, alpha=None, class_counts=None: nn.CrossEntropyLoss(reduction='mean'),
    'focal': lambda num_classes, alpha=None, class_counts=None: FocalLoss(
                 alpha=alpha if alpha is not None else 1.0, gamma=2.0, reduction='mean', num_classes=num_classes),
    'cb':    lambda num_classes, alpha=None, class_counts=None: CBFocalLoss(
                 cls_num_list=class_counts, beta=0.999, gamma=2.0, reduction='mean'),
    'ldam':  lambda num_classes, alpha=None, class_counts=None: LDAMLoss(
                 cls_num_list=class_counts, max_m=0.5, s=30, reduction='mean'),
}


def build_loss(name, num_classes, alpha=None, class_counts=None):
    """
    根据名称构建分类损失（criterion）。
    Args:
        name: 'ce' / 'focal' / 'cb' / 'ldam'
        num_classes: 类别数
        alpha: Focal Loss 的类别平衡系数（仅 'focal' 使用）
        class_counts: 每类样本数列表（'cb' / 'ldam' 使用）
    Returns:
        nn.Module: 构建好的损失模块
    """
    name = (name or 'focal').lower()
    if name not in LOSS_REGISTRY:
        raise ValueError(f"未知的损失名称: {name}，可选: {list(LOSS_REGISTRY.keys())}")
    return LOSS_REGISTRY[name](num_classes, alpha=alpha, class_counts=class_counts)


'''
遍历数据集 计算focal Loss的alpha参数
'''


def get_multi_dataset_alpha(dataset_list, data_root, old_new_lbl_map_dict, seed=0, p_value=0, k_value=0, split_prefix=''):
    """
    批量统计多数据集的类别样本数,计算每个数据集的Focal Loss alpha参数
    Args:
        dataset_list: 数据集名称列表(如 ["FLSMDD", "NKSID"])
        data_root: 所有数据集的根目录(如 "../data")
        old_new_lbl_map_dict: 每个数据集的标签映射字典(键：数据集名,值：标签映射)
                              示例：{"FLSMDD": {0:0,1:1,...9:9}, "NKSID": {0:0,...7:7}}
        seed: 随机种子(用于交叉验证索引,确保统计的是训练集)
        split_prefix: 划分文件前缀(与 get_new_kfold_img_idx 一致: ''=默认, '_htpl'=HTPL-JB 2094/523)
    Returns:
        alpha_result_dict: 每个数据集的alpha、类别数和类别样本数
        (键：数据集名,值：(alpha列表, 类别数, 类别样本数列表))
    """
    alpha_result_dict = {}  # 存储最终结果

    for dataset_name in dataset_list:
        print(f"正在统计 {dataset_name} 的类别样本数...")
        # 1. 拼接当前数据集的路径(适配你的目录结构)
        dataset_dir = os.path.join(data_root, dataset_name)  # 单个数据集目录
        train_txt_path = os.path.join(dataset_dir, f"train_abs{split_prefix}.txt")  # 训练集路径文件
        if not os.path.exists(train_txt_path):
            raise FileNotFoundError(f"{dataset_name} 的 train_abs.txt 不存在：{train_txt_path}")

        # 2. 获取当前数据集的训练集索引(假设用你的交叉验证函数get_new_kfold_img_idx)
        init_random_seeds(seed)

        train_idx, _ = get_new_kfold_img_idx(p=p_value, k=k_value, data_dir=dataset_dir, split_prefix=split_prefix)
        # 3. 获取当前数据集的标签映射
        old_new_lbl_map = old_new_lbl_map_dict[dataset_name]
        num_classes = len(old_new_lbl_map)  # 该数据集的类别数

        # 4. 统计当前数据集训练集的类别样本数(复用之前的准确统计逻辑)
        count_dict = defaultdict(int)
        # 读取train.txt,获取所有样本的原始标签
        all_original_labels = []
        with open(train_txt_path, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                all_original_labels.append(int(parts[1]))  # 存储原始标签

        # 统计训练集(train_idx)的类别样本数(按新标签)
        for idx in train_idx:
            original_label = all_original_labels[idx]
            new_label = old_new_lbl_map[original_label]
            count_dict[new_label] += 1

        # 按新标签顺序整理样本数(确保与模型输出类别顺序一致)
        class_counts = [count_dict.get(i, 0) for i in range(num_classes)]
        total_samples = sum(class_counts)
        print(f"{dataset_name} 类别样本数：{class_counts},总样本数：{total_samples}")

        # 5. 计算当前数据集的alpha(与类别频率成反比)
        alpha = [total_samples / (num_classes * cnt) if cnt != 0 else 1.0 for cnt in class_counts]
        # 处理极端情况：若某类样本数为0,alpha设为1.0(避免除以0)
        alpha = [1.0 if np.isinf(a) else a for a in alpha]

        # 6. 存储当前数据集的结果
        alpha_result_dict[dataset_name] = (alpha, num_classes, class_counts)
        print(f"{dataset_name} alpha计算完成:{alpha}\n")

    return alpha_result_dict

# ---------------------- 封装通用的通道注意力分组绘制函数 ----------------------
def plot_channel_att_group(
    channel_weights,  # 单个样本的通道权重 [15,]
    group_name,       # 分组名称（如"RGB"、"bior小波"）
    group_channels,   # 该组包含的通道索引（如RGB是[0,1,2]）
    all_channel_names,# 所有15个通道的完整名称列表
    save_path         # 图片保存路径
):
    """
    绘制某一组通道的注意力权重热力图
    :param channel_weights: 单个样本的15个通道权重(numpy数组 [15,])
    :param group_name: 分组名称（用于图片标题）
    :param group_channels: 该组的通道索引列表(如RGB组是[0,1,2])
    :param all_channel_names: 所有15个通道的完整名称
    :param save_path: 图片保存路径
    """
    # 提取该组的权重和对应的通道名称
    group_weights = channel_weights[group_channels]  # 按索引取该组权重
    group_channel_names = [all_channel_names[i] for i in group_channels]  # 该组通道名称

    # 绘制热力图
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure(figsize=(len(group_channel_names)*2, 3))  # 宽度随通道数自适应
    sns.heatmap(
        group_weights.reshape(1, -1),  # 转为[1, N]矩阵（N是该组通道数）
        cmap='YlOrRd',
        annot=True,
        fmt='.3f',
        xticklabels=group_channel_names,
        yticklabels=[f'{group_name} Attention'],
        vmin=0,
        vmax=1  # 权重范围固定0~1，便于不同组对比
    )
    plt.title(f'{group_name} Channel Attention Weights')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

# # ---------------------- 主可视化函数（整合通道分组+空间注意力） ----------------------
# def visualize_attention(data_iter, net, device, save_dir, target_class='B_Propeller', class_label_map=None):
#     """
#     可视化注意力权重（仅输出指定类别【单张】原图+空间注意力图，移除对比图）
#     Args:
#         data_iter: 数据迭代器
#         net: 模型（需返回 output, channel_weights, spatial_weights）
#         device: 设备（cuda:0/cpu）
#         save_dir: 保存目录
#         target_class: 要可视化的目标类别（如'B_Propeller'）
#         class_label_map: 标签值→类别名称的映射（如{6:'B_Propeller'}），需根据数据集配置
#     """
#     net.eval()
#     os.makedirs(save_dir, exist_ok=True)

#     # 1. 定义15个通道的完整名称（和你的特征顺序严格对应）
#     all_channel_names = [
#         'RGB-1', 'RGB-2', 'RGB-3',
#         'bior-LL', 'bior-LH', 'bior-HL', 'bior-HH',
#         'rbio-LL', 'rbio-LH', 'rbio-HL', 'rbio-HH',
#         'sym4-LL', 'sym4-LH', 'sym4-HL', 'sym4-HH'
#     ]

#     # 2. 定义通道分组规则（保留原有逻辑，若不需要通道图可注释）
#     channel_groups = [
#         ('RGB', [0, 1, 2]),
#         ('bior', [3, 4, 5, 6]),
#         ('rbio', [7, 8, 9, 10]),
#         ('sym4', [11, 12, 13, 14]),
#         ('all', list(range(15)))
#     ]

#     # 反归一化参数（还原RGB原图）
#     mean = torch.tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
#     std = torch.tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)

#     # 标记是否找到目标类别样本
#     found_target = False

#     with torch.no_grad():
#         for batch_idx, (*X_parts, y) in enumerate(data_iter):
#             if found_target:
#                 break  # 找到并可视化单张目标图后，直接终止遍历

#             # 遍历当前batch的标签，找到目标类别的第一个样本
#             for sample_idx in range(len(y)):
#                 current_label = y[sample_idx].item()
#                 # 匹配目标类别
#                 if class_label_map.get(current_label) == target_class:
#                     found_target = True

#                     # 仅取该单个样本（避免冗余）
#                     X = tuple(part[sample_idx:sample_idx+1].to(device) for part in X_parts)  # [1, C, H, W]

#                     # 前向传播获取注意力权重
#                     output, channel_weights, spatial_weights = net(*X, is_visualize=True)

#                     # ---------------------- 1. 绘制通道注意力图（可选，不需要可注释） ----------------------
#                     # sample_channel_att = channel_weights[0].squeeze().cpu().numpy()
#                     # for group_name, group_channel_idx in channel_groups:
#                     #     group_save_path = os.path.join(
#                     #         save_dir,
#                     #         f'{target_class}_channel_att_{group_name}.png'
#                     #     )
#                     #     plot_channel_att_group(
#                     #         channel_weights=sample_channel_att,
#                     #         group_name=group_name,
#                     #         group_channels=group_channel_idx,
#                     #         all_channel_names=all_channel_names,
#                     #         save_path=group_save_path
#                     #     )

#                     # ---------------------- 2. 绘制目标类别【单张】原始RGB图 ----------------------
#                     # 取RGB输入并反归一化
#                     raw_rgb = X_parts[0][sample_idx:sample_idx+1]  # [1, 3, H, W]
#                     raw_rgb = raw_rgb * std + mean
#                     raw_rgb = torch.clip(raw_rgb, 0, 1)
#                     raw_rgb = raw_rgb.squeeze().permute(1, 2, 0).cpu().numpy()

#                     # 保存单张原图
#                     plt.figure(figsize=(8, 8))
#                     plt.imshow(raw_rgb)
#                     plt.title(f'Original RGB ({target_class})')
#                     plt.axis('off')
#                     plt.tight_layout()
#                     plt.savefig(os.path.join(save_dir, f'{target_class}_original_rgb.png'), dpi=300)
#                     plt.close()

#                     # ---------------------- 3. 绘制目标类别【单张】空间注意力图 ----------------------
#                     sample_spatial_att = spatial_weights[0].squeeze().cpu().numpy()
#                     plt.figure(figsize=(8, 8))
#                     im = plt.imshow(sample_spatial_att, cmap='jet')
#                     plt.title(f'Spatial Attention ({target_class})')
#                     plt.axis('off')
#                     plt.colorbar(im, label='Attention Weight')
#                     plt.tight_layout()
#                     plt.savefig(os.path.join(save_dir, f'{target_class}_spatial_att.png'), dpi=300)
#                     plt.close()

#                     print(f"\n✅ 已找到并保存{target_class}类别的单张可视化图！")
#                     print(f"📁 保存路径：{save_dir}")
#                     break  # 找到第一个样本后，终止当前batch的遍历

#     if not found_target:
#         print(f"\n⚠️  未在数据迭代器中找到{target_class}类别样本！")

# 保留通道注意力绘制函数（不需要通道图可删除）
# def plot_channel_att_group(channel_weights, group_name, group_channels, all_channel_names, save_path):
#     group_weights = channel_weights[group_channels]
#     group_names = [all_channel_names[i] for i in group_channels]

#     plt.figure(figsize=(len(group_names)*1.2, 4))
#     bars = plt.bar(group_names, group_weights, color='skyblue')
#     plt.title(f'Channel Attention - {group_name} Group')
#     plt.ylabel('Attention Weight')
#     plt.ylim(0, np.max(channel_weights)*1.1)
#     plt.xticks(rotation=45, ha='right')
#     for bar in bars:
#         height = bar.get_height()
#         plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
#                  f'{height:.3f}', ha='center', va='bottom', fontsize=8)
#     plt.tight_layout()
#     plt.savefig(save_path, dpi=300)
#     plt.close()
