# DA-BiGRU-CNN-LOB

**Domain-Aware Dual-Branch Recurrent Networks for Limit Order Book Mid-Price Prediction**

> Solovev Sergei — Faculty of Computer Science, HSE University

Paper: [`paper/main.pdf`](paper/main.pdf)

## Overview

This repository contains the source code and experiments for a study on deep learning approaches to LOB mid-price prediction. We present three contributions:

1. **DA-BiGRU-CNN** — a dual-branch architecture that separates LOB features into *price* and *volume* channels, processes them through dedicated BiGRU encoders with shared microstructure features, and fuses representations via multi-scale Conv1d (kernels 3, 5, 7).

2. **Feature Sufficiency Hypothesis** — empirical evidence that a GRU with 53 basic features matches performance of 219 extensively engineered features (weighted Pearson 0.246 vs 0.248), suggesting recurrent hidden states implicitly learn rolling statistics, EMAs, and lag-differences.

3. **Negative Ensemble Effect** — combining GRU and LightGBM degrades prediction quality (0.2657 vs 0.2662 GRU-only), contradicting the common assumption that model diversity improves ensembles.

## Architecture

```
LOB snapshot (32 raw features)
        |
   Feature Engineering (53 features)
        |
   ┌────┴────┐
   Price     Volume
  (37 dim)  (37 dim)
   │          │
  BiGRU     BiGRU
   │          │
   └────┬─────┘
        │
  Conv1d Fusion (k=3,5,7)
  192 → 96 → 48 channels
        │
     MLP Head
        │
   [pred_t0, pred_t1]
```

## Results

| Model | Features | Weighted Pearson |
|-------|----------|-----------------|
| LightGBM | 219 | 0.168 |
| GRU (53 features) | 53 | 0.246 |
| GRU (219 features) | 219 | 0.248 |
| **GRU v1 (optimized)** | **53** | **0.266** |
| GRU + LGB ensemble | 53 + 219 | 0.266 (degraded) |
| DA-BiGRU-CNN | 53 | In development |

## Project Structure

```
.
├── paper/
│   ├── main.tex              # Full LaTeX paper
│   ├── main.pdf              # Compiled PDF
│   ├── generate_figures.py   # Publication-quality figure generation
│   └── figures/              # PDF & PNG figures
│
├── train_gru.py              # GRU v1 training (53 features)
├── train_gru_v2.py           # GRU v2 training (219 features)
├── train_dual_bigru_cnn.py   # DA-BiGRU-CNN training
├── train_lgbm.py             # LightGBM baseline
├── train_catboost.py         # CatBoost baseline
│
├── solution_gru.py           # GRU inference (incremental)
├── solution_dual_bigru_cnn.py # DA-BiGRU-CNN inference (batch)
├── solution_ensemble_3model.py # 3-model ensemble
│
├── utils.py                  # Shared utilities
├── lob_competition_full.py   # Full experiment pipeline
└── requirements.txt          # Python dependencies
```

## Dataset

The experiments use a large-scale LOB dataset comprising 12,165 sequences of limit order book snapshots (12.1M timesteps total). The data is available via the [Wunder Fund Predictorium](https://wundernn.io/predictorium) platform.

Each timestep contains 32 raw features:
- **Prices**: 6 bid + 6 ask price levels
- **Volumes**: 6 bid + 6 ask volume levels
- **Trade data**: last trade price, volume, aggressor side, time delta

**Targets**: mid-price return at two horizons (t0, t1).

## Quick Start

```bash
# Clone
git clone https://github.com/SergeySolovyev/DA-BiGRU-CNN-LOB.git
cd DA-BiGRU-CNN-LOB

# Install dependencies
pip install -r requirements.txt

# Train GRU baseline
python train_gru.py

# Generate paper figures
cd paper && python generate_figures.py
```

## Evaluation Metric

Weighted Pearson correlation with weights = |target|:

$$\rho_w = \frac{\sum w_i (y_i - \bar{y}_w)(\hat{y}_i - \bar{\hat{y}}_w)}{\sqrt{\sum w_i (y_i - \bar{y}_w)^2} \sqrt{\sum w_i (\hat{y}_i - \bar{\hat{y}}_w)^2}}$$

## Citation

```bibtex
@article{solovev2026lob,
  title={When Less Is More: Domain-Aware Dual-Branch Recurrent Networks for Limit Order Book Mid-Price Prediction},
  author={Solovev, Sergei},
  year={2026},
  url={https://github.com/SergeySolovyev/DA-BiGRU-CNN-LOB}
}
```

## License

MIT License. See [LICENSE](LICENSE).

## Contact

- Email: sesesolovev@edu.hse.ru
- Web: [sergeisolovev.com](https://sergeisolovev.com)
- GitHub: [SergeySolovyev](https://github.com/SergeySolovyev)
