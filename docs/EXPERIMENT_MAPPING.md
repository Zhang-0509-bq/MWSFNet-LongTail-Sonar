# Experiment mapping

The experiment IDs below match Tables 2–6 of the JEI manuscript. Each row uses p0/k0 and seeds 0, 1, and 2. Source paths are relative to the authors' historical results tree and serve as provenance locators. The executable review entry point for every row is `python evaluation/verify_archived.py --experiment ID`.

| ID | Paper result | Historical result directory | Groups and special switches in the local reference |
| --- | --- | --- | --- |
| A0 | Table 4, RGB-only | `results/htpl20/representation/rgb_only` | `rgb` |
| A1 | Table 4, bior3.7 only | `results/htpl20/representation/bior37_only` | `bior` |
| A2 | Table 4, rbio3.9 only | `results/htpl20/representation/rbio39_only` | `rbio` |
| A3 | Tables 4 and 6, sym4 only | `results/htpl20/representation/sym4_only` | `sym4` |
| A4 | Table 4, three wavelets | `results/htpl20/representation/three_wavelets` | `bior,rbio,sym4` |
| A5 | Tables 4 and 5, RGB + three wavelets with naive concatenation | `results/htpl20/representation/full_mwsfnet` | `rgb,bior,rbio,sym4`, both branch gates off |
| A6 | Table 4, RGB + sym4 | `results/htpl20/representation/rgb_sym4` | `rgb,sym4` |
| A7 | Table 4, bior3.7 + sym4 | `results/htpl20/representation/bior_sym4` | `bior,sym4` |
| A8 | Table 4, rbio3.9 + sym4 | `results/htpl20/representation/rbio_sym4` | `rbio,sym4` |
| SBW | Table 5, global scalar weighting | `results/htpl20/sbw/sbw_mwsfnet` | `rgb,bior,rbio,sym4`, `--static_branch_gate` |
| BAF | Tables 2, 5 and 6, MWSFNet with BAF | `results/htpl20/baf/baf_mwsfnet` | `rgb,bior,rbio,sym4`, `--branch_gate` |
| WC | Table 2, plain Wavelet-CNN baseline | `results/htpl20/wavelet_cnn_baseline/sym4_dwt_resnet18` | `sym4`, `--dwt_cnn` |
| FLS | Table 3, independently trained FLSMDD evaluation | `results/flsmdd/cross_dataset/mwsfnet_flsmdd` | `rgb,bior,rbio,sym4`, `--branch_gate`, FLSMDD and empty split suffix |
| A3 ensemble | Table 6, sym4-only soft voting | `results/htpl20/representation/a3_ensemble_3seed` | `A3` members, seeds 0, 1, 2 |
| BAF ensemble | Tables 2 and 6, MWSFNet soft voting | `results/htpl20/baf/baf_ensemble_3seed` | `BAF` members, seeds 0, 1, 2 |

A1–A3 use the single-basis variants of the fusion architecture. WC is the plain four-input-channel DWT-to-ResNet18 baseline, with a different first-layer initialization path. A5 is the canonical representation-run recording. A numerically similar earlier result is outside this package. The entries attributed to HTPL-JB and NAS-MFF in the paper are literature comparisons, and the records in this package cover the authors' experiments only.

`configs/canonical.json` stores the original settings as recorded for each seed, alongside the numeric summary and archive hashes. A switch absent in an older run file remains absent in that record. Shared recorded settings include `es=80`, `bs=32`, `lr=0.01`, `backbone=resnet18`, `loss=focal`, `no_plud=false`, `aug=none`, and `sampler=null`. NKSID uses `_htpl`. FLSMDD uses an empty suffix. The current source performs SGD with momentum 0.9 and weight decay 0.005, `MultiStepLR([35, 75], gamma=0.5)`, and dropout 0.3 in the fusion model.

## Methodology to code

| Manuscript item | Local implementation to inspect |
| --- | --- |
| §3.2 single-level DWT, subbands and z-score, resize to 224 | `source_review/My_Dataset.py`, `wavelet_transform`, `__getitem__`, and the `resize_coeff` helper |
| §3.3.1 branch refinement | `source_review/my_utils.py`, `RGBCNN`, `WaveletCNN`, `ChannelAttention` |
| §3.3.2 adaptive reweighting and concatenation | `BranchAdaptiveGate` and `FusionResNet18.forward` in `source_review/my_utils.py` |
| Table 5, learned global weighting | `StaticBranchGate` in the same file |
| §3.3.3 post-concatenation attention and dropout | `AttentionFusion` and `FusionResNet18` in the same file |
| §3.1 and §3.4.1 modified ResNet18, classifier and priors | `FusionResNet18.__init__`; see the scale-order discrepancy in [reproducibility notes](REPRODUCIBILITY_NOTES.md) |
| Table 2 plain Wavelet-CNN | `DWTCNNResNet18`; the `--dwt_cnn` path selects it |
| §3.4.2 weighted focal loss, logit separation, and schedule | `FocalLoss`, `get_multi_dataset_alpha`, and `build_loss` in `my_utils.py`; `push_logit_loss`, `train_mwsfnet_one_epoch`, and `train_mwsfnet` in `train.py` |
| §4.1 split and label mapping | `scripts/prepare_splits.py`, `get_new_kfold_img_idx` in `my_utils.py`, and the dataset-specific class maps in `train.py` |
| §4.1.3 metrics and ensemble | `cal_gmean` and `softmax` in `evaluation/ensemble_eval.py`; `evaluation/verify_archived.py` for the recorded runs, recall, and sample standard deviation |

All table rows with local predictions have an exercised CPU verification entry point. Each training command remains a reference recipe pending identification of the original server snapshot and environment.
