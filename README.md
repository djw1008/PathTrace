# PathTrace

Official PyTorch implementation of **PathTrace: A Pathway-Aware Transformer for Cross-Tissue DNA Methylation Age Prediction via Pairwise Learning**.

Repository: https://github.com/djw1008/PathTrace

## Overview

PathTrace aggregates CpG sites by Reactome pathways, encodes each pathway into a latent token with a pathway-specific MLP tokenizer, models global pathway interactions with a Transformer encoder, and predicts chronological age via two alternating objectives:

1. Pairwise age-difference learning (contrastive / pairwise branch).
2. Absolute age regression (prediction branch).

This repository contains the core model architecture, training scripts, and inference/evaluation utilities.

## Repository layout

```
.
├── train.py                          # Main training script for PathTrace
├── pathway_vae_sup_train_npz.py      # Pre-train supervised pathway VAEs (optional tokenizer init)
├── run_external_eval.py              # Inference and per-GSE/per-tissue evaluation on external test sets
├── data/
│   └── pathways/
│       ├── ReactomePathways.gmt  # CpG-to-Reactome pathway annotations
│       └── ReactomePathways.txt  # Reactome ID-to-name mapping
├── examples/
│   └── input_data/
│       ├── example_beta.csv      # Ten GSE72680 samples (CpGs x samples)
│       └── example_meta.csv      # Matching sample metadata and ages
├── models/
│   ├── __init__.py
│   ├── ContrastivePathwayTransformer.py   # PathTrace model
│   ├── PathwayVAE.py                      # Vanilla pathway VAE
│   └── PathwayVAESup.py                   # Supervised pathway VAE
└── utils/
    ├── __init__.py
    ├── dataload_utils.py             # NPZ dataset loaders
    ├── pathway_utils.py              # Reactome GMT → CpG mapping
    └── file_utils.py                 # Small directory helpers
```

## Dependencies

```bash
pip install -r requirements.txt
```

Core packages: PyTorch, NumPy, pandas, scikit-learn, scipy, matplotlib.

## Data preparation

PathTrace expects training data in `.npz` format with the following fields:

- `data`: a dictionary mapping sample IDs to `{ "feature": np.ndarray, "target": float, "additional": dict }`.
- `train_index`: list/array of training sample IDs.
- `val_index`: list/array of validation sample IDs.
- `cpgs`: list/array of CpG probe names.

Pathway annotations should be provided as a Reactome GMT file where gene entries follow `GENENAME_cgXXXXXX`.

### Training data

PathTrace was trained using the preprocessed pan-tissue DNA methylation dataset released with MAPLE (Zhang et al.). Download `epiAge_traindata.npz` from the official MAPLE resources:

- MAPLE repository: https://github.com/Drizzle-Zhang/MAPLE
- Training data: https://drive.google.com/file/d/1krIh0JC8ejN8gnQpJIHrkqnmINdc9Iyg/view

Place the downloaded file at `./data/epiAge_traindata.npz`. The training dataset is maintained and distributed by the MAPLE authors and is not redistributed in this repository.

> **Note:** Pre-trained PathTrace weights will be released separately.

## Training

```bash
python train.py \
  --data_source ./data/epiAge_traindata.npz \
  --gmt_file ./data/pathways/ReactomePathways.gmt \
  --latent_dim 32 \
  --hidden_topo 128,128 \
  --num_layers 3 \
  --batch_size 256 \
  --num_epochs 500 \
  --path_save ./checkpoints/pathtrace
```

For optional supervised pathway-VAE pre-training (used to initialize the tokenizer):

```bash
python pathway_vae_sup_train_npz.py \
  --data_source ./data/epiAge_traindata.npz \
  --gmt_file ./data/pathways/ReactomePathways.gmt \
  --out_dir ./pretrained_vae
```

Then pass `--vae_ckpt_dir ./pretrained_vae/checkpoints` to `train.py`.

## External evaluation / inference

Given a trained checkpoint, external beta matrix, and sample metadata, run the following command from the repository root:

```bash
python run_external_eval.py \
  --checkpoint ./checkpoints/pathtrace/checkpoints/best_model.pt \
  --beta_path ./examples/input_data/example_beta.csv \
  --meta_path ./examples/input_data/example_meta.csv \
  --output_dir ./external_eval_results
```

The beta-value matrix must contain CpGs as rows and samples as columns. The metadata CSV must contain a sample identifier column (`sample_id`, `SampleID`, `ID`, `id`, or `sample`) and an `age` column. Optional columns used for stratified evaluation include `project_id`, `sample_type`, `tissue`, and `platform`.

The bundled example contains 10 publicly available control samples from GSE72680, selected to span ages 18--77 years. It is provided only as a compact inference example; the full GSE72680 cohort was used for the reported dataset-level evaluation.

The script outputs `predictions.csv`, `gse_mae_comparison.csv`, and `tissue_mae_comparison.csv`.

## Citation

If you use this code, please cite the PathTrace paper (link to be added upon publication).
