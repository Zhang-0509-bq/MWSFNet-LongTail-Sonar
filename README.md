# MWSFNet-LongTail-Sonar

**MWSFNet: Multi-Basis Wavelet Spatio-Frequency Fusion for Long-Tailed Forward-Looking Sonar Image Classification**

This review package accompanies the manuscript prepared for the *Journal of Electronic Imaging*. It contains a locally archived implementation for inspection, the fixed experiment splits, recorded run settings, and a CPU-only verifier for the archived predictions.

The prediction verifier was run successfully for **39 single-model recordings and both three-model ensembles**. The exact server-side source revisions, transforms, and software versions that produced the checkpoints remain unverified. The implementation in `source_review/` is therefore a reference for code inspection. See the [reproducibility notes](docs/REPRODUCIBILITY_NOTES.md) for the open source-to-manuscript discrepancy before attempting model-level reproduction.

## Architecture

MWSFNet classifies closed-set forward-looking sonar images with imbalanced class frequencies. It uses a 3-channel RGB representation and three single-level discrete wavelet transforms (`bior3.7`, `rbio3.9`, `sym4`) with four subbands each. Branch Feature Refinement Blocks (BFRB) refine the inputs. Branch-Aware Fusion (BAF) weights the refined features before concatenation. Dual-Stage Attention Fusion (DSAF) applies channel and spatial recalibration. The resulting 15-channel representation is passed through dropout and a modified ResNet18. Training uses inverse-frequency-weighted focal loss with a logit-separation term and an epoch-dependent mixture.

## Verify the recorded results

From this repository root, install the CPU review dependencies and run the verifier.

```sh
python -m pip install numpy scipy scikit-learn
python evaluation/verify_archived.py
python evaluation/verify_archived.py --experiment BAF --per-class
```

The verifier checks test-set order and labels, compares accuracy, G-mean, macro-F1, and class recalls against the recorded values, and checks the sample standard deviation across seeds 0, 1, and 2. For A3 and BAF, it converts each member's logits to probabilities, averages the three probability vectors, and checks the resulting predictions and metrics against the stored ensembles. It reads the supplied files and prints the verification to standard output. Image files and model weights are unnecessary for this check.

| NKSID experiment | G-mean (%) | Macro-F1 (%) |
| --- | ---: | ---: |
| MWSFNet with BAF, three single-model runs | 95.60 ± 1.92 | 94.41 ± 2.28 |
| MWSFNet with BAF, three-model ensemble | 97.02 | 96.13 |
| Sym4-only, three single-model runs | 95.84 ± 1.79 | 95.56 ± 2.31 |
| Sym4-only, three-model ensemble | 96.65 | 96.30 |

## Dataset preparation

- **NKSID.** Use the 2,617 images from the original release. Keep the category folders and extract both archives for the `tire` class. The paper uses 1,677 training, 417 validation, and 523 test images in its own fixed split. The source repository's split files and the paper's split serve different protocols. [NKSID release](https://github.com/Jorwnpay/NK-Sonar-Image-Dataset) · [Dataset paper](https://doi.org/10.1016/j.eswa.2024.123495)
- **FLSMDD.** Use the ten class-crop folders from the original water-tank release. The separately packaged classification-task release has an additional background class and different splits. In the paper, three models were trained and evaluated within FLSMDD on 1,324 training, 326 validation, and 714 test images. [Water-tank release](https://github.com/mvaldenegro/marine-debris-fls-datasets/releases/tag/watertank-v1.0) · [Dataset paper](https://doi.org/10.1109/OCEANS58557.2025.11104623)

Obtain both datasets from their providers and follow the applicable dataset terms. This repository contains the paper's dataset-relative split identifiers and labels, along with numeric model outputs. Dataset images are obtained separately.

The following commands require a local image root for each dataset with the original category folders directly underneath it. Substitute the paths for your installation. The reference loader uses whitespace-delimited manifest entries, so the image paths must contain no whitespace.

```sh
python scripts/prepare_splits.py NKSID /datasets/NKSID
python scripts/prepare_splits.py FLSMDD /datasets/FLSMDD
```

The helper creates git-ignored path lists under `data/` from the supplied p0/k0 split and preserves the original row and fold-index order. Use `--check-only` to check paths without creating files.

## Environment and training recipe

[`requirements.txt`](requirements.txt) lists the imports needed for the training reference and optional plots. The local prediction verifier was tested with Python 3.11, NumPy, SciPy, and scikit-learn. Source imports and synthetic CPU forwards also passed in a local PyTorch environment, with pretrained-weight loading stubbed for the smoke test. Select a PyTorch/torchvision build appropriate for your Linux CUDA runtime if you plan to train. The exact historical Python, PyTorch, CUDA, and pretrained-weight versions are still being reconciled.

The command below reconstructs the CLI settings of the archived NKSID BAF seed-0 configuration for the local training reference. It is provided for inspection and server-side follow-up after source provenance is confirmed. No training was run for this release. The script expects to be launched inside `source_review/`. Use a new run output directory for a future execution.

```sh
python -m pip install -r requirements.txt
cd source_review
python train.py --dataset NKSID --split_prefix _htpl --p_value 0 --k_value 0 --backbone resnet18 --groups rgb,bior,rbio,sym4 --branch_gate --loss focal --aug none --bs 32 --es 80 --lr 0.01 --seed 0 --method_name baf_mwsfnet --exp_out ../runs/BAF/seed0
```

The [experiment mapping](docs/EXPERIMENT_MAPPING.md) lists the remaining groups and switches. For a future training check, compare the new held-out outputs with the verified archives using the same ordered split. The training reference writes artifacts when `--exp_out` is supplied.

## Files

| Directory | Contents |
| --- | --- |
| [`source_review/`](source_review/) | Locally archived loader, model, loss functions, and training reference |
| [`configs/`](configs/) | Filtered run configurations, recorded results, and checksums |
| [`splits/`](splits/) | Fixed dataset-relative p0/k0 splits for both datasets |
| [`artifacts/`](artifacts/) | Compressed archived logits, labels, and predictions |
| [`evaluation/`](evaluation/) | Executed archive verifier and historical ensemble reference |
| [`scripts/`](scripts/) | Fixed-split path preparation |
| [`docs/`](docs/) | Experiment mapping, release modifications, and reproducibility limits |

The code and dataset terms are handled separately. Any licensing statement for this repository is subject to the authors' and source owners' approval.
