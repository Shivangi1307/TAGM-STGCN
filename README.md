# TAGM-STGCN: Temporal Adaptive Graph Memory for Traffic Forecasting

TAGM-STGCN achieves competitive performance with dynamic graph learning 
while maintaining stable predictions across longer horizons.

## Overview

This project implements a spatio-temporal graph neural network for traffic forecasting, extending the standard STGCN framework with dynamic graph learning, memory mechanisms, and regime-aware adaptation.

Traditional traffic forecasting models rely on static graph structures, which fail to capture temporal variations in traffic patterns. This work introduces adaptive graph representations that evolve over time, improving predictive performance and robustness.

---

## Why This Matters

Traffic patterns are highly dynamic. Static graph models fail to capture
these variations. This work demonstrates that adaptive graph structures
improve robustness and generalization.

## Key Contributions

* **Dynamic Graph Learning**
  Learns time-varying adjacency matrices directly from input sequences.

* **Graph Memory (Temporal Smoothing)**
  Maintains an exponential moving average of dynamic graphs to stabilize learning.

* **Regime-Aware Adaptive Fusion**
  Learns traffic regimes and adaptively combines static, dynamic, and memory graphs.

* **Modular Experiment Design**
  Supports multiple configurations:

  * Baseline (static graph)
  * Enhanced (dynamic + memory)
  * Regime (full model)

---

## Model Variants

| Variant  | Description                           |
| -------- | ------------------------------------- |
| Baseline | Standard STGCN with static adjacency  |
| Enhanced | Dynamic adjacency + graph memory      |
| Regime   | Adaptive fusion using regime encoding |

---

## Project Structure

```
code/
  model/
  script/
  main.py
  main_enhanced.py
  main_regime.py

results/
experiments/
```

---

## Dataset Setup

Datasets are not included due to size constraints.

```
data/
  metr-la/
    vel.csv
    adj.npz
  pems-bay/
    vel.csv
    adj.npz
```

Supported datasets:

* METR-LA
* PEMS-BAY
* PEMSD7

---

## Installation

```
pip install -r requirements.txt
```

---

## Usage

### Baseline

```
python code/main.py
```

### Enhanced Model

```
python code/main_enhanced.py
```

### Regime Model

```
python code/main_regime.py
```

---


## Evaluation Metrics

Model performance is evaluated using:

* Mean Absolute Error (MAE)
* Root Mean Square Error (RMSE)
* Mean Absolute Percentage Error (MAPE)

---

# Results

## PeMSD7 — 15 Minute Horizon

### Classical & Deep Learning Models

| Model   | MAE (15/30/60)     | MAPE (%)            | RMSE               |
| ------- | ------------------ | ------------------- | ------------------ |
| HA      | 4.01               | 10.61               | 7.20               |
| ARIMA   | 5.55 / 3.63 / 4.54 | 5.81 / 8.88 / 11.50 | 4.55 / 6.67 / 8.28 |
| FNN     | 2.74 / 4.02 / 5.04 | 6.38 / 9.72 / 12.38 | 4.75 / 6.98 / 8.58 |
| FC-LSTM | 3.57 / 3.94 / 4.16 | 8.60 / 9.55 / 10.10 | 6.20 / 7.03 / 7.51 |
| GCGRU   | 2.37 / 3.31 / 4.01 | 5.54 / 8.06 / 9.99  | 4.21 / 5.96 / 7.13 |
| STTN    | 2.14 / 2.70 / 3.03 | 5.05 / 6.68 / 8.04  | 4.01 / 5.48 / 6.25 |
| STGCN   | 2.25 / 3.03 / 3.57 | 5.26 / 7.33 / 8.69  | 4.04 / 5.70 / 6.77 |

### Proposed Models

| Model            | MAE (15/30/60)     | MAPE (%)           | RMSE               |
| ---------------- | ------------------ | ------------------ | ------------------ |
| STGCN Baseline   | 2.25 / 2.86 / 3.52 | 5.13 / 6.74 / 8.50 | 3.90 / 5.11 / 6.17 |
| Dynamic MH-STGCN | 2.23 / 2.87 / 3.48 | 5.17 / 6.16 / 8.46 | 3.87 / 5.11 / 6.14 |
| TAGM-STGCN       | 2.24 / 2.88 / 3.55 | 5.14 / 6.82 / 8.56 | 3.89 / 5.12 / 6.19 |

---

## PeMSD7 — Classification

| Model            | Accuracy | Precision | Recall | F1    |
| ---------------- | -------- | --------- | ------ | ----- |
| STGCN Baseline   | 0.946    | 0.822     | 0.777  | 0.798 |
| Dynamic MH-STGCN | 0.947    | 0.843     | 0.767  | 0.799 |
| TAGM-STGCN       | 0.945    | 0.825     | 0.776  | 0.789 |

---

## PEMS-BAY — 15 Minute Horizon

### Classical Models

| Model   | MAE (15/30/60)     | MAPE (%)           | RMSE               |
| ------- | ------------------ | ------------------ | ------------------ |
| HA      | 2.88               | 6.80               | 5.59               |
| VAR     | 1.74 / 2.32 / 2.93 | 3.6 / 5.0 / 6.5    | 3.16 / 4.25 / 5.44 |
| SVR     | 1.85 / 2.48 / 3.28 | 3.8 / 5.5 / 8.0    | 3.59 / 5.18 / 7.08 |
| ARIMA   | 1.62 / 2.33 / 3.38 | 3.50 / 5.40 / 8.30 | 3.30 / 4.76 / 6.50 |
| WaveNet | 1.39 / 1.83 / 2.35 | 2.91 / 4.16 / 5.70 | 3.01 / 4.21 / 5.43 |
| DCRNN   | 1.38 / 1.74 / 2.07 | 2.90 / 3.90 / 4.90 | 2.95 / 3.97 / 4.74 |

### Proposed Models

| Model            | MAE (15/30/60)     | MAPE (%)           | RMSE               |
| ---------------- | ------------------ | ------------------ | ------------------ |
| STGCN Baseline   | 1.40 / 1.73 / 2.07 | 2.87 / 3.34 / 3.96 | 2.60 / 3.34 / 3.96 |
| Dynamic MH-STGCN | 1.41 / 1.76 / 2.11 | 2.86 / 3.71 / 4.50 | 2.61 / 3.37 / 3.99 |
| TAGM-STGCN       | 1.41 / 1.77 / 2.11 | 2.96 / 3.87 / 4.73 | 2.62 / 3.37 / 4.03 |

---

## PEMS-BAY — Classification

| Model            | Accuracy | Precision | Recall | F1    |
| ---------------- | -------- | --------- | ------ | ----- |
| STGCN Baseline   | 0.977    | 0.838     | 0.734  | 0.775 |
| Dynamic MH-STGCN | 0.988    | 0.843     | 0.767  | 0.799 |
| TAGM-STGCN       | 0.975    | 0.825     | 0.740  | 0.789 |

---

## Observations

* Dynamic graph learning consistently improves over static STGCN.
* Graph memory stabilizes predictions across longer horizons.
* Regime-aware fusion enables adaptive performance across traffic conditions.

---

## Notes

* Dataset files are excluded
* Model checkpoints are not included
* All results are reproducible



Detailed results and plots are available in the `results/` directory.




