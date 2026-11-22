# coding: utf-8
import os
from PIL import Image
import pywt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torch.nn.functional as F
import torchvision.transforms as transforms

# ----------------------------------------- 数据集类 --------------------------------------

# 分组 -> 小波基名映射(与 my_utils.GROUP_WAVELET 保持一致)
GROUP_WAVELET = {'bior': 'bior3.7', 'rbio': 'rbio3.9', 'sym4': 'sym4'}
CANONICAL_GROUPS = ['rgb', 'bior', 'rbio', 'sym4']


class MyDataset(Dataset):
    def __init__(self, txt_path, img_idx, transform=None, target_transform=None, old_new_lbl_map=None, input_size=(224, 224), wavelets=['bior3.7', 'rbio3.9', 'sym4'], mean=None, std=None, pil_transform=None, groups=None):
        self.data = []
        self.transform = transform
        self.pil_transform = pil_transform
        self.input_size = tuple(input_size)
        self.target_transform = target_transform
        self.old_new_lbl_map = old_new_lbl_map
        self.wavelets = wavelets
        self.mean = mean if mean is not None else [0.485, 0.456, 0.406]
        self.std = std if std is not None else [0.229, 0.224, 0.225]
        # groups: 决定返回哪些张量(顺序即返回顺序)。None=默认全组 rgb+bior+rbio+sym4(与旧行为完全一致)。
        # 规范顺序: rgb(有则) -> bior -> rbio -> sym4。
        if groups is None:
            self.groups = list(CANONICAL_GROUPS)
        else:
            gs = [g.strip() for g in groups.split(',')] if isinstance(groups, str) else list(groups)
            self.groups = [g for g in CANONICAL_GROUPS if g in gs]

        if not os.path.exists(txt_path):
            raise FileNotFoundError(f"数据文件不存在：{txt_path}")

        with open(txt_path, 'r', encoding='utf-8') as f:
            all_lines = f.readlines()

        max_valid_idx = len(all_lines) - 1

        for idx in img_idx:
            if idx < 0 or idx > max_valid_idx:
                raise IndexError(f"索引 {idx} 超出范围(有效范围:0~{max_valid_idx})")

            line = all_lines[idx].strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                raise ValueError(f"第 {idx} 行格式错误，应为 '路径 标签'，实际：{line}")
            img_path = parts[0]
            try:
                original_label = int(parts[1])
            except ValueError:
                raise ValueError(f"第 {idx} 行的标签不是整数：{parts[1]}")
            if not os.path.isabs(img_path):
                img_path = img_path.replace('\\', '/')
                img_path = os.path.abspath(img_path)
            if not os.path.exists(img_path):
                raise FileNotFoundError(f"图像不存在：{img_path}（来自第 {idx} 行）")
            self.data.append((img_path, original_label))
        if not self.data:
            raise ValueError(f"没有有效数据！请检查 {txt_path} 和传入的索引 {img_idx}")

    def wavelet_transform(self, image, wavelet):
        img_array = np.array(image.convert('L'))
        coeffs2 = pywt.dwt2(img_array, wavelet)
        LL, (LH, HL, HH) = coeffs2
        def normalize(coeff):
            mean = np.mean(coeff)
            std = np.std(coeff)
            return (coeff - mean) / (std + 1e-8)

        return normalize(LL), normalize(LH), normalize(HL), normalize(HH)
    def __getitem__(self, index):
        img_path, original_label = self.data[index]
        try:
            img_pil = Image.open(img_path).convert('RGB')
        except Exception as e:
            raise IOError(f"读取图像失败：{img_path}，错误：{str(e)}")

        # 增强策略：
        #  - pil_transform 非空：整张图做"几何+颜色"增强(输出224×224 PIL)，RGB 与小波共享受同一张增强图，
        #    保证两分支在空间上对齐（修复原版"RGB增强/小波不增强"导致的对不准）。
        #  - 否则退回原逻辑：transform 只作用于 RGB，小波用原图（保证默认行为与原始 MWSFNet 完全一致）。
        if self.pil_transform is not None:
            aug_pil = self.pil_transform(img_pil)
            rgb_tensor = transforms.Normalize(self.mean, self.std)(transforms.ToTensor()(aug_pil))
        elif self.transform is not None:
            rgb_tensor = self.transform(img_pil)
            aug_pil = img_pil
        else:
            basic_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std)
            ])
            rgb_tensor = basic_transform(img_pil)
            aug_pil = img_pil

        #  小波特征：与 RGB 使用同一张(增强)图像生成，尺寸保证一致。仅计算 self.groups 选中的小波基。
        def resize_coeff(coeff):
            coeff_tensor = torch.tensor(coeff, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
            if coeff_tensor.shape[2:] != self.input_size:
                coeff_tensor = F.interpolate(coeff_tensor, size=self.input_size, mode='bilinear', align_corners=False)
            return coeff_tensor.squeeze(0)

        wav_cache = {}
        def get_wavelet_feat(key):
            if key not in wav_cache:
                LL, LH, HL, HH = self.wavelet_transform(aug_pil, GROUP_WAVELET[key])
                wav_cache[key] = torch.cat([
                    resize_coeff(LL),
                    resize_coeff(LH),
                    resize_coeff(HL),
                    resize_coeff(HH)
                ], dim=0)
            return wav_cache[key]

        outputs = []
        for g in self.groups:
            if g == 'rgb':
                outputs.append(rgb_tensor)
            else:
                outputs.append(get_wavelet_feat(g))

        # ====================== 标签处理 ======================
        if self.old_new_lbl_map is not None:
            new_label = self.old_new_lbl_map[original_label]
        else:
            new_label = original_label

        if self.target_transform:
            new_label = self.target_transform(new_label)

        outputs.append(new_label)
        return tuple(outputs)

    def __len__(self):
        return len(self.data)
