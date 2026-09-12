# MWSFNet-LongTail-Sonar
Official implementation of MWSFNet for closed-set long-tailed forward-looking sonar image classification.

# MWSFNet: Multi-Basis Wavelet Spatio-Frequency Fusion for Long-Tailed Sonar Image Classification

Official implementation of **MWSFNet**, a multi-basis wavelet spatio-frequency classification framework for **closed-set long-tailed forward-looking sonar (FLS) image classification**.

> **Paper status:** Under review  
> **Code status:** The complete implementation and reproducibility package will be released upon paper acceptance.

## Overview

Long-tailed forward-looking sonar image classification is challenging because severe class imbalance is accompanied by limited visual detail and insufficient representation of minority classes.

MWSFNet investigates this problem from a representation and fusion perspective. The framework jointly exploits RGB appearance and multiple single-level wavelet representations generated using different wavelet bases. It combines branch-wise feature refinement, feature-conditioned branch reweighting, channel-spatial recalibration, frequency-aware initialization, and long-tail-oriented optimization.

The overall framework follows:

**RGB / bior3.7 / rbio3.9 / sym4  
→ Branch Feature Refinement  
→ Branch-Aware Fusion  
→ Weighted Branch Concatenation  
→ Dual-Stage Attention Fusion  
→ Modified ResNet18  
→ Classification**

## Main Components

### Multi-Basis Wavelet Spatio-Frequency Representation

MWSFNet jointly uses:

- RGB representation: 3 channels
- `bior3.7` single-level DWT: 4 channels
- `rbio3.9` single-level DWT: 4 channels
- `sym4` single-level DWT: 4 channels

The resulting representation contains 15 input channels.

### Branch Feature Refinement Block

Each representation branch is independently refined before fusion to enhance informative responses while preserving the original representation through a residual connection.

### Branch-Aware Fusion

Branch-Aware Fusion (BAF) performs feature-conditioned branch reweighting before concatenation. Rather than assuming equal contributions from all representations, BAF estimates normalized branch coefficients from branch descriptors and uses them to regulate the contribution of each refined feature map.

### Dual-Stage Attention Fusion

After branch-aware concatenation, the fused representation is sequentially recalibrated along the channel and spatial dimensions.

### Frequency-Aware Prior Initialization

The ImageNet-pretrained ResNet18 first convolution is expanded from 3 to 15 input channels. The original RGB filters are retained, while Gaussian- and Sobel-derived priors are used to initialize the additional wavelet channels.

### Long-Tail-Oriented Objective

Training combines a class-balanced focal term and a logit-margin term with progressively adjusted loss weights.

## Datasets

Experiments are conducted on two forward-looking sonar image datasets:

- **NKSID** — primary benchmark for long-tailed sonar image classification
- **FLSMDD** — supplementary cross-dataset validation

NKSID exhibits a severe class imbalance with an imbalance factor of 47.55, while FLSMDD has a milder imbalance factor of 6.91.

Dataset files are **not redistributed in this repository**. Please obtain the datasets from their original sources and follow their respective licenses and usage requirements.

## Main Results

Under the aligned 80/20 NKSID evaluation setting:

| Model | G-mean (%) | Macro-F1 (%) |
|---|---:|---:|
| MWSFNet, single model | 95.60 ± 1.92 | 94.41 ± 2.28 |
| MWSFNet, 3-model ensemble | **97.02** | **96.13** |

The three-model ensemble achieves a G-mean of **97.02%**. Under the aligned 80/20 evaluation setting, this is 2.25 percentage points higher than the 94.77% G-mean reported for HTPL-JB.

The literature comparison is provided as a reference under an aligned evaluation setting and should not be interpreted as an exact sample-level split or strictly like-for-like ensemble comparison.

## Repository Structure

The complete repository will include:

```text
MWSFNet-LongTail-Sonar/
├── configs/
├── datasets/
├── models/
├── losses/
├── utils/
├── scripts/
├── train.py
├── test.py
├── ensemble.py
├── requirements.txt
└── README.md
