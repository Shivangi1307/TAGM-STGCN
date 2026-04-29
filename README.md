# TAGM-STGCN: Temporal Adaptive Graph Memory for Traffic Forecasting

## Abstract

Accurate traffic forecasting is a critical component of intelligent transportation systems. This project presents an implementation of **TAGM-STGCN**, a spatio-temporal graph convolutional network enhanced with dynamic graph learning, temporal graph memory, and regime-aware fusion. The model addresses limitations of static graph assumptions by adapting spatial dependencies over time while maintaining stability through memory mechanisms.

---

## Paper

The full manuscript is available in the repository:

```
paper/TAGM_STGCN_Paper.pdf
```

---

## Method Overview

The proposed architecture extends the standard STGCN framework with three key components:

1. **Dynamic Adjacency Learning**
   Learns time-varying spatial relationships using multi-head attention.

2. **Temporal Graph Memory**
   Stabilizes dynamic graphs via exponential moving average (EMA).

3. **Regime-Aware Fusion**
   Combines static, dynamic, and memory graphs using learned weights based on traffic conditions.

---

## Repository Structure

```
TAGM-STGCN/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── paper/
│   └── TAGM_STGCN_Paper.pdf
│
├── code/
│   ├── model/
│   │   ├── dynamic_adj.py
│   │   ├── graph_memory.py
│   │   ├── layers.py
│   │   ├── models.py
│   │   └── regime_encoder.py
│   │
│   ├── script/
│   │   ├── dataloader.py
│   │   ├── utility.py
│   │   └── earlystopping.py
│   │
│   ├── main.py
│   └── main_enhanced.py
│
├── experiments/
│   ├── plots/
│   └── results.json
│
└── data/  (excluded from repository)
```

---

## Installation

Create a virtual environment and install dependencies:

```
pip install -r requirements.txt
```

---

## Usage

Run baseline model:

```
python code/main.py
```

Run enhanced TAGM-STGCN:

```
python code/main_enhanced.py
```

---

## Datasets

The model is evaluated on publicly available traffic datasets:

* PeMS-D7
* PeMS-BAY

Dataset files are not included in the repository.

---

## Evaluation Metrics

Model performance is evaluated using:

* Mean Absolute Error (MAE)
* Root Mean Square Error (RMSE)
* Mean Absolute Percentage Error (MAPE)

---

## Results Summary

The proposed model demonstrates:

* Improved adaptability through dynamic graph learning
* Increased stability via temporal graph memory
* Competitive performance across multiple prediction horizons

Detailed results and plots are available in the `experiments/` directory.


---


