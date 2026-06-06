# ⚔️ LoL Win Predictor: DSC 148 Course Project

> **Beyond the Snapshot: Trajectory-Aware Features for Win and Comeback Prediction in Professional League of Legends**

UC San Diego | DSC 148: Introduction to Data Mining

---

## Overview

This project predicts match outcomes in professional League of Legends from early-game state, using the Oracle Elixir 2024 dataset. The main idea is a **trajectory-aware feature set**: momentum terms that capture how gold, XP, and kill differentials are *changing* between 10 and 15 minutes, not just their snapshot values. We test it across four model families.

There are two tasks:
1. **Win Prediction**: given game state through 15 minutes, predict the match winner.
2. **Comeback Classification**: given that a team trails by 500+ gold at 10 minutes, predict whether it recovers and wins.

### Headline finding (reported without spin)

After feature engineering, a **regularized Logistic Regression is the strongest model on both tasks**, and gradient boosting (LightGBM) does *not* beat it. The honest read is that once you compute the trajectory and interaction features by hand, what is left is close to linear, so the extra model complexity buys nothing here. The ablation backs this up: **objective control, not trajectory, is the dominant signal**. That negative result is the point of the project, not a footnote.

---

## Results

### Task 1: Win Prediction (5-fold stratified CV)

| Model | Accuracy | F1 | AUC-ROC |
|---|---|---|---|
| **Logistic Regression (best)** | **0.711** | **0.720** | **0.785** |
| Naive Bayes | 0.687 | 0.693 | 0.754 |
| Random Forest | 0.694 | 0.703 | 0.770 |
| LightGBM | 0.700 | 0.710 | 0.771 |

### Task 2: Comeback Prediction (5-fold stratified CV)

| Model | F1 | AUC-ROC | Precision | Recall |
|---|---|---|---|---|
| **Logistic Regression (best)** | **0.551** | **0.734** | 0.470 | **0.667** |
| Naive Bayes | 0.546 | 0.717 | 0.472 | 0.649 |
| Random Forest | 0.516 | 0.712 | **0.523** | 0.510 |
| LightGBM | 0.406 | 0.702 | 0.439 | 0.384 |

LightGBM's AUC is competitive, but its default threshold is too conservative on the minority class, which gives it the weakest F1. Calibrating the threshold (Platt scaling, say) is the obvious next step.

### Key findings from analysis

- **Ablation:** Dropping objective-control features causes the biggest AUC drop (0.023). Dropping trajectory features costs only 0.002, small but consistent across folds. Dropping the 15-minute snapshot group slightly *raises* AUC, which points to redundancy with the trajectory and early features.
- **Importance vs. ablation:** Gold momentum is the **#1 feature by LightGBM split count**, but removing it barely changes AUC. That is because momentum is just a deterministic function of the 10m and 15m differentials. Split count measures how handy a feature is for the trees, not how much unique signal it adds.
- **Patch sensitivity:** Training on patches 14.1 through 14.11 and testing on 14.12 through 14.16 gives AUC 0.761, close to the CV value of 0.771, so the feature set holds up across patches.

---

## Dataset

**Oracle Elixir 2024 Professional Match Data** ([oracleselixir.com](https://oracleselixir.com/tools/downloads))
- About 5,000 professional matches across 10 leagues (LPL, LCK, LEC, LCS, and 6 regional circuits)
- About 60,000 team- and player-level rows; modeling uses 5,000 Blue-side team-games
- Game state at 10 and 15 minutes, objective counts, first objectives, patch, league

Place the CSV at `lol_2024.csv` in the project root.

---

## Feature Engineering (21 features)

| Group | Features |
|---|---|
| Early (10m) | GD, XPD, CSD, KD at 10 minutes |
| Mid (15m) | GD, XPD, CSD, KD at 15 minutes |
| **Trajectory** | ΔGD, ΔXPD, ΔKD (10→15m momentum) |
| Objectives | Dragon/baron/herald/tower differentials, turret plates, first-obj sum |
| Interactions | GD15 × baron_diff, GD15 × dragon_diff |
| Context | Patch version (numeric), league avg win rate |

---

## Project Structure

```
lol_project/
├── 01_eda.py            # Exploratory data analysis (figures 1 to 5)
├── 02_modeling.py       # Feature engineering + modeling (figures 6 to 10)
├── dashboard.py         # Streamlit interactive demo
├── report/
│   └── report.tex       # ACM-format paper (LaTeX)
├── outputs/             # Generated figures and result CSVs
├── requirements.txt
└── README.md
```

---

## Setup & Run

```bash
pip install -r requirements.txt

python 01_eda.py            # generates EDA figures
python 02_modeling.py       # trains models, generates result figures + CSVs
streamlit run dashboard.py  # launches interactive win-probability demo
```

---

## Live Demo

The dashboard takes a game state through sliders (gold, XP, and kill diffs, plus objectives) and returns a live win-probability estimate with a gold-trajectory plot and an objective radar.

```bash
streamlit run dashboard.py
```

---

## Authors

- Evan Ngo ([github.com/mrevanngo](https://github.com/mrevanngo))
- Jimbo Cai ([github.com/jmbc29](https://github.com/jmbc29))

## Citation

Data: Oracle Elixir (oracleselixir.com)
