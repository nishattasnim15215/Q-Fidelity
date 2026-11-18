# Q-Fidelity: a quantum-simulator fidelity branch for image anomaly detection

Single-file reference implementation (`main.py`). The model scores anomalies from two
complementary patch-level signals and fuses them:

- a **fidelity branch** over each image patch. Patches are projected into a small
  state space and scored by a simulated parameterized circuit (`--n_qubits`, `--n_layers`,
  depth-dependent noise via `--noise_scale`). The circuit is *simulated in PyTorch* -- no
  quantum hardware or SDK is required, and `--branch_type classical_control` swaps in a
  parameter-matched classical MLP as the control condition.
- a **patch VAE** reconstruction error (`PatchVAE`), giving the generative signal.

The two are combined by `--fusion_mode` (`fixed` weighting by `--gen_weight_patch`, or a
`learned_gate`). Image scores aggregate the patch map; evaluation reports ROC-AUC and F1.

## Requirements
Python 3.12.3 (tested) and the pinned packages in `requirements.txt`
(`pip install -r requirements.txt`). `torch`/`torchvision` are the CUDA 12.4 builds. MIT
licensed (see `LICENSE`).

## Data: MVTec AD 2 (not redistributed here)
The code expects **MVTec AD 2**, which is *not* included in this repository -- it is
third-party data under MVTec's own license terms and must be obtained directly from MVTec:
<https://www.mvtec.com/company/research/datasets/mvtec-ad-2>

Unpack it so that `--data_root` (default `Dataset/`) holds one directory per category, in the
layout the download already uses:

```
Dataset/<category>/train/good/                    # normal images only
Dataset/<category>/validation/good/
Dataset/<category>/test_public/good/              # labelled test split
Dataset/<category>/test_public/bad/
Dataset/<category>/test_public/ground_truth/      # per-defect segmentation masks
Dataset/<category>/test_private/                  # unlabelled challenge splits
Dataset/<category>/test_private_mixed/
```

The eight categories used by default: `can`, `fabric`, `fruit_jelly`, `rice`, `sheet_metal`,
`vial`, `wallplugs`, `walnuts`.

## Run
```bash
python main.py                                   # all 8 categories, seed 7, defaults
python main.py --categories can,rice --seeds 7,13,29
python main.py --data_root /path/to/mvtec_ad_2 --results_dir Results --amp
```
Every run writes to `--results_dir` (default `Results/`):

| file | contents |
|---|---|
| `<category>_seed_<s>_summary.csv` | image- and pixel-level metrics for one cell |
| `<category>_seed_<s>_predictions.csv` | per-image scores and labels |
| `<category>_seed_<s>_speed.csv` | throughput / timing |
| `<category>_seed_<s>_ablation.csv` | per-category ablation grid |
| `seed_level_results.csv`, `seed_mean_results.csv`, `seed_category_results.csv` | aggregates across seeds and categories |
| `ablation_aggregate_seed_<s>.csv` | ablation rolled up over categories |

## Ablations
The controls that isolate each component, all runnable from the same entry point:

```bash
python main.py --branch_type classical_control      # classical branch, parameter-matched
python main.py --projection_mode trainable_linear   # vs. the frozen random projection
python main.py --illumination_token_mode none       # or: stats (default), learned
python main.py --noise_mode none                    # or: sinusoidal (default), linear, learned
python main.py --fusion_mode learned_gate           # vs. the fixed fusion weight
python main.py --retrain_ablation                   # retrain each ablation cell rather than reusing weights
```
Ablation cells run for `--ablation_epochs` (default 4) against the main `--epochs` (default 8).

## Reproducibility
`--seeds` sets torch, CUDA, numpy and Python RNGs per run (`set_seed`). Results are
reproducible up to GPU floating-point non-determinism. Defaults: `--img_size 256`,
`--patch_size 32`, `--batch_size 8`, `--lr 2e-3`, `--epochs 8`, `--n_qubits 6`, `--n_layers 6`,
`--noise_scale 0.35`, `--vae_latent_dim 64`, `--gen_weight_patch 0.35`, `--beta_kl 1e-3`.
Pass `--amp` for mixed precision and `--compile` for `torch.compile`.

## Status
Code release. This repository contains the implementation and the commands to reproduce it;
no result tables are committed, because the runs that would fill them are regenerated from
the dataset above.
