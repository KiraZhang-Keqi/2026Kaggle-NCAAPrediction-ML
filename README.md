# NCAA March Madness 2026 — End-to-End ML Prediction Pipeline

**An end-to-end machine learning system for predicting 132,000+ NCAA tournament matchups (Men's & Women's), achieving a Brier Score of 0.1679 through a multi-stage ensemble of XGBoost, GLM, Elo ratings, and logistic calibration.**

Built for [Kaggle March Machine Learning Mania 2026](https://kaggle.com/competitions/march-machine-learning-mania-2026).

## Table of Contents

- [Project Overview](#project-overview)
- [Results](#results)
- [Technical Architecture](#technical-architecture)
- [Feature Engineering](#feature-engineering)
- [Modeling Strategy](#modeling-strategy)
- [Getting Started](#getting-started)

## Project Overview

### Problem

Given historical NCAA basketball data (2003–2025), predict the win probability for every possible matchup in the 2026 NCAA tournament — 132,133 team pairs across both Men's and Women's brackets. Submissions are evaluated on **Brier Score** (lower is better), which penalizes both inaccurate and poorly calibrated probabilities.

### Approach

Instead of treating this as a binary classification problem, I designed a **two-stage regression-to-probability pipeline**: XGBoost first predicts the expected point margin, then a calibrated logistic layer converts margins into well-calibrated probabilities. This preserves ordinal information — a predicted 15-point margin carries different signal than a 2-point margin, even though both are "wins."

### Scale

| Dimension | Value |
|-----------|-------|
| Training data | 208,280 regular-season games + 2,410 tournament games |
| Prediction targets | 132,133 matchups (66,430 Men's + 65,703 Women's) |
| Engineered features | 53 per matchup |
| Cross-validation folds | 22 seasons (Leave-One-Season-Out) |
| Feature families | 7 independent signal sources |

## Results

### Overall Performance

| Metric | Value |
|--------|-------|
| **Brier Score (OOF)** | **0.1679** |
| MAE on point margin | 7.68 – 10.55 across seasons |
| Best logistic calibration | C=0.001, L2 penalty, SAGA solver |
| Prediction range | [0.034, 0.974] |

### Per-Season Brier Scores (Leave-One-Season-Out)

```
Season  Brier   Games       Season  Brier   Games
──────────────────────      ──────────────────────
2003    0.1836   128        2013    0.1804   260
2004    0.1703   128        2014    0.1748   260
2005    0.1663   128        2015    0.1429   260  ← Best
2006    0.1925   128        2016    0.1874   260
2007    0.1440   128        2017    0.1566   260
2008    0.1510   128        2018    0.1901   260
2009    0.1602   128        2019    0.1672   260
2010    0.1690   254        2021    0.1726   260
2011    0.1798   260        2022    0.1815   260
2012    0.1563   260        ...
```

Consistent performance across 22 seasons with no catastrophic failures — demonstrates robustness to year-over-year distribution shift in tournament dynamics.

## Technical Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        DATA LAYER                               │
│  Men's + Women's Regular Season (208K games) + Tournament (2.4K)│
│  Overtime normalization · Symmetric T1/T2 doubling · Seed merge │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                   FEATURE ENGINEERING (×7)                       │
│                                                                 │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────────┐   │
│  │ Elo      │ │ GLM      │ │ Box-Score│ │ SOS-Adjusted     │   │
│  │ Ratings  │ │ Quality  │ │ Rate     │ │ Win Rate         │   │
│  │ (K=80)   │ │ (Gaussian│ │ (eFG%,   │ │ (2-pass opponent │   │
│  │          │ │  GLM)    │ │  ORtg,   │ │  weighting)      │   │
│  └──────────┘ └──────────┘ │  DRtg..) │ └──────────────────┘   │
│  ┌──────────┐ ┌──────────┐ └──────────┘ ┌──────────────────┐   │
│  │ Head-to- │ │ Away Win │               │ Last-14-Day      │   │
│  │ Head     │ │ Ratio    │               │ Momentum         │   │
│  │ (Laplace)│ │          │               │                  │   │
│  └──────────┘ └──────────┘               └──────────────────┘   │
│                                                                 │
│  → 53 features: T1 absolute + T2 absolute + T1−T2 differentials│
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                      MODELING PIPELINE                           │
│                                                                 │
│  [Stage 1] XGBoost (reg:squarederror, hist, lossguide)          │
│            └─ Predicts point margin (continuous)                 │
│            └─ 22× LOSO models, all retained for ensemble        │
│                                                                 │
│  [Stage 2] Logistic Calibration                                 │
│            └─ Margin → P(win) via LogisticRegression             │
│            └─ Grid search: 13 C values × 3 solvers × 2 penalties│
│                                                                 │
│  [Stage 3] Probability Adjustment                               │
│            └─ Fine-tune tails (near 0 and 1) for Brier          │
│            └─ Grid search: 5×16×5×16 = 6,400 configurations    │
│                                                                 │
│  [Inference] Ensemble all 22 LOSO models → mean prediction      │
└─────────────────────────────────────────────────────────────────┘
```

## Feature Engineering

### 3.1 Advanced Box-Score Rate Stats

Aggregated per team per season from detailed regular-season box scores, with overtime normalization (scaled to 40-minute equivalent):

| Feature | Formula | Signal |
|---------|---------|--------|
| eFG% | (FGM + 0.5·FGM3) / FGA | Shooting efficiency accounting for 3pt value |
| OffRtg | 100 · Score / Poss | Points produced per 100 possessions |
| DefRtg | 100 · Opp_Score / Poss | Points allowed per 100 possessions |
| TOR | TO / Poss | Ball security under pressure |
| ORR | OR / (OR + Opp_DR) | Second-chance opportunity creation |
| DRR | DR / (DR + Opp_OR) | Defensive rebounding dominance |
| AstR | Ast / FGM | Ball movement and team play |
| FTR | FTA / FGA | Ability to draw fouls |

**Possessions** estimated as: `FGA − OR + TO + 0.44·FTA`

### 3.2 Elo Ratings

Custom Elo system (K=80, base=1000, width=400) computed over the regular season, resetting each year. Captures team strength trajectory without carry-over bias.

### 3.3 GLM Team Quality

Per-season Gaussian GLM: `PointDiff ~ T1_TeamID + T2_TeamID` on games involving tournament-caliber teams. The T1 coefficient isolates each team's quality after controlling for opponent. Most compute-intensive feature (~9 min for 22 seasons).

### 3.4 Strength-of-Schedule Adjusted Win Rate

Two-pass algorithm:
1. Compute raw win rate per team
2. Re-weight each game outcome by opponent's raw win rate

This ensures a team that goes 20-10 against top-50 opponents is rated higher than one going 25-5 against bottom-100.

### 3.5 Contextual Features

- **Head-to-Head Matchup Probability**: Laplace-smoothed (prior=[1,1]) win probability from regular-season encounters between each specific pair of teams
- **Last-14-Day Win Ratio**: Late-season momentum signal (games after DayNum > 118)
- **Away Win Ratio**: Road game performance — proxy for neutral-site readiness

### 3.6 Feature Composition

Final feature vector per matchup (53 dimensions):
- 17 T1 features (seed, elo, quality, 14 box-score rates)
- 17 T2 features (mirror)
- 17 T1−T2 differential features
- 1 head-to-head matchup probability
- 1 gender flag (men/women)

## Modeling Strategy

### Why Two-Stage (Margin → Probability)?

| Approach | Limitation |
|----------|------------|
| Direct classification (P(win)) | Treats a 1-point win and a 30-point blowout identically |
| **Margin regression → calibration** | **Preserves signal strength, then calibrates separately** |

The point margin encodes how dominant a win was. The logistic layer then learns the non-linear mapping from margin to probability, producing better-calibrated outputs than direct classification.

### Cross-Validation Design

**Leave-One-Season-Out (LOSO)**: Each of 22 seasons (2003–2025) is held out once. This is stricter than k-fold because it prevents any temporal leakage — the model never sees future seasons during training.

All 22 models are retained and ensembled at inference: `P(win) = mean(P₁, P₂, ..., P₂₂)`.

### Hyperparameter Optimization

Three independent grid searches, each optimized for downstream Brier Score:

| Stage | Search Space | Optimal |
|-------|-------------|---------|
| XGBoost | tree depth, learning rate, subsample, colsample, regularization | Pre-tuned (see config) |
| Logistic | 13 C values × 3 solvers × 2 penalties × 2 class weights | C=0.001, L2, SAGA |
| Probability Adjustment | 5 upper thresholds × 16 upper deltas × 5 lower × 16 lower = 6,400 | upper: (0.8, +0.01), lower: (0.2, +0.12) |

## Getting Started

### Prerequisites

```
Python 3.8+
```

### Installation

```bash
git clone https://github.com/<your-username>/ncaa-march-madness-2026.git
cd ncaa-march-madness-2026

pip install numpy pandas xgboost scikit-learn statsmodels matplotlib seaborn tqdm
```

### Data

Download from the [Kaggle competition page](https://kaggle.com/competitions/march-machine-learning-mania-2026):

```bash
kaggle competitions download -c march-machine-learning-mania-2026 -p data/
```

### Run

```bash
jupyter notebook 2026NCAA_ML.ipynb
```

### Project Structure

```
├── 2026NCAA_ML.ipynb          # Complete pipeline (single notebook)
├── data/
│   ├── M*.csv                 # Men's regular season, tournament, seeds
│   ├── W*.csv                 # Women's regular season, tournament, seeds
│   └── SampleSubmissionStage2.csv
├── predictions.csv            # Final submission (132,133 matchups)
└── README.md
```

## License

This project is for educational and portfolio purposes. Competition data is provided by Kaggle under its [competition rules](https://www.kaggle.com/competitions/march-machine-learning-mania-2026/rules).
