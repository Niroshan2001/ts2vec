# TS2Vec-MSM: Hybrid Self-Supervised Learning for Time Series

This repository contains the implementation of **TS2Vec-MSM**, a hybrid self-supervised learning framework that enhances TS2Vec with Masked Signal Modeling (MSM) for universal time-series representation learning.

## Overview

TS2Vec-MSM combines:
- **Contrastive Learning** (from TS2Vec): Learns discriminative instance-level representations
- **Masked Signal Modeling (MSM)**: Learns generative temporal continuity and dependencies

## Architecture

```
Input Time Series → TS2Vec Encoder → Representations
                         ↓
                    MSM Decoder → Reconstruction Loss
                         ↓
               Combined Loss = (1-λ) * Contrastive + λ * MSM
```

## Key Features

- **Dual Objectives**: Simultaneous discriminative and generative learning
- **Dynamic λ Scheduling**: Adaptive balance between contrastive and MSM losses
- **Flexible Masking**: Support for random and block masking strategies
- **Universal**: Works across classification, forecasting, and anomaly detection

## Quick Start

### Basic Usage
```bash
# Hybrid model with balanced losses (λ=0.5)
python train_msm.py ETTh1 my_experiment --loader forecast_csv --msm-weight 0.5 --eval

# Contrastive only (original TS2Vec, λ=0)
python train_msm.py ETTh1 contrastive_only --loader forecast_csv --msm-weight 0.0 --eval

# MSM only (λ=1)
python train_msm.py ETTh1 msm_only --loader forecast_csv --msm-weight 1.0 --eval

# Dynamic λ scheduling
python train_msm.py ETTh1 dynamic_hybrid --loader forecast_csv --msm-weight 0.5 --dynamic-lambda --eval
```

### New Command Line Arguments

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--msm-weight` | Weight for MSM loss (λ parameter, 0=contrastive only, 1=MSM only) | 0.5 |
| `--msm-mask-rate` | Percentage of timestamps to mask for MSM | 0.15 |
| `--msm-decoder-depth` | Number of layers in the MSM decoder | 3 |
| `--dynamic-lambda` | Enable dynamic λ scheduling during training | False |

## Experimental Workflows

### 1. Ablation Study
Run the complete ablation study to compare all configurations:
```bash
bash experiments/ablation_study.sh
```

### 2. Dataset-Specific Experiments
```bash
# ETT Forecasting
bash experiments/ett_experiments.sh

# KPI Anomaly Detection
bash experiments/kpi_experiments.sh
```

### 3. Analysis and Visualization
```python
from experiments.analyze_results import TS2VecMSMAnalyzer

analyzer = TS2VecMSMAnalyzer()
analyzer.load_results()
analyzer.generate_paper_table()
analyzer.compare_configurations('ETTh1', 'forecasting')
```

## Implementation Details

### MSM Decoder
- Lightweight architecture following MAE design principles
- 3-layer MLP with ReLU activations and dropout
- Reconstructs original signal values from encoded representations

### Loss Function
```
L_total = (1 - λ) * L_contrastive + λ * L_MSM
```

Where:
- `L_contrastive`: Hierarchical contrastive loss from TS2Vec
- `L_MSM`: Mean squared error on masked positions
- `λ`: Balance parameter (0 ≤ λ ≤ 1)

### Dynamic λ Scheduling
```python
λ(t) = 0.1 + 0.4 * (1 + cos(π * t/T)) / 2
```

Starts with more contrastive learning, gradually increases MSM contribution.

## Research Validation

### Hypothesis
The hybrid approach should outperform single-paradigm methods by:
1. **Contrastive-only**: Good instance discrimination, limited temporal modeling
2. **MSM-only**: Strong temporal continuity, weak instance separation  
3. **TS2Vec-MSM**: Best of both worlds - captures "what" (identity) and "how" (evolution)

### Expected Results
- **Classification**: Improved accuracy on UCR/UEA datasets
- **Forecasting**: Better MSE/MAE on ETT datasets
- **Anomaly Detection**: Higher AUROC on KPI/Yahoo datasets

## File Structure

```
models/
├── msm_decoder.py          # MSM decoder and loss implementation
ts2vec_msm.py              # Main hybrid model class
train_msm.py               # Training script for TS2Vec-MSM
experiments/
├── ablation_study.sh      # Complete ablation study
├── ett_experiments.sh     # ETT forecasting experiments  
├── kpi_experiments.sh     # KPI anomaly detection experiments
└── analyze_results.py     # Results analysis and visualization
```

## Citation

If you use this code in your research, please cite:

```bibtex
@article{ts2vec_msm_2025,
  title={Hybrid Self-Supervised Learning for Time-Series: Enhancing TS2Vec with Masked Signal Modeling},
  author={Niroshan G.},
  journal={CS4681 Advanced Machine Learning},
  year={2025},
  note={Project ID: TS007}
}
```

## Original TS2Vec Citation

```bibtex
@inproceedings{yue2022ts2vec,
  title={TS2Vec: Towards Universal Representation of Time Series},
  author={Yue, Zhihan and Wang, Yujing and Duan, Jiahui and Yang, Tianmeng and Huang, Chaohe and Tong, Yunhai and Xu, Bingxing},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={36},
  pages={8980--8987},
  year={2022}
}
```
