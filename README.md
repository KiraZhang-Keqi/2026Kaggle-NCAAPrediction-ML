# NCAA March Madness 2026: An End-to-End ML Prediction Pipeline

An end-to-end machine learning system for predicting 132,000+ NCAA tournament matchups (Men's & Women's), achieving a **Brier Score of 0.1679** through a multi-stage ensemble of XGBoost, GLM, Elo ratings, and logistic calibration.

Built for [Kaggle March Machine Learning Mania 2026](https://www.kaggle.com/competitions/march-machine-learning-mania-2026)

---

## Project Overview

This project predicts win probabilities for every possible matchup in the 2026 NCAA tournament using 22 seasons of historical data (2003–2025). The pipeline includes:

- **Two-stage prediction**: XGBoost margin regression → logistic probability calibration
- **53-dimensional feature engineering** across 7 independent signal families
- **Leave-One-Season-Out (LOSO)** cross-validation preventing temporal leakage
- **Ensemble inference** averaging 22 season-held-out models for robust predictions
- **Probability tail adjustment** via 6,400-configuration grid search optimized for Brier Score

---

## Key Results

### Overall Performance

| Metric | Value |
|---|---|
| Brier Score (OOF) | **0.1679** |
| MAE on Point Margin | 7.68 – 10.55 across seasons |
| Prediction Range | [0.034, 0.974] |
| Best Logistic Calibration | C=0.001, L2 penalty, SAGA solver |

### Per-Season Brier Scores (Leave-One-Season-Out)

| Season | Brier | Games | | Season | Brier | Games |
|---|---|---|---|---|---|---|
| 2003 | 0.1836 | 128 | | 2013 | 0.1804 | 260 |
| 2004 | 0.1703 | 128 | | 2014 | 0.1748 | 260 |
| 2005 | 0.1663 | 128 | | 2015 | **0.1429** ← Best | 260 |
| 2006 | 0.1925 | 128 | | 2016 | 0.1874 | 260 |
| 2007 | 0.1440 | 128 | | 2017 | 0.1566 | 260 |
| 2008 | 0.1510 | 128 | | 2018 | 0.1901 | 260 |
| 2009 | 0.1602 | 128 | | 2019 | 0.1672 | 260 |
| 2010 | 0.1690 | 254 | | 2021 | 0.1726 | 260 |
| 2011 | 0.1798 | 260 | | 2022 | 0.1815 | 260 |
| 2012 | 0.1563 | 260 | | | | |

Consistent performance across 22 seasons with no catastrophic failures — demonstrates robustness to year-over-year distribution shift in tournament dynamics.

### Why Two-Stage (Margin → Probability)?

| Approach | Limitation |
|---|---|
| Direct classification P(win) | Treats a 1-point win and a 30-point blowout identically |
| Margin regression → calibration | Preserves signal strength, then calibrates separately |

The point margin encodes *how dominant* a win was. The logistic layer then learns the non-linear mapping from margin to probability, producing better-calibrated outputs than direct classification.

---

## Pipeline Architecture

```
Data Loading → Overtime Normalization → Symmetric T1/T2 Doubling → Seed Merge
                                                                      ↓
                                    ┌─────────────────────────────────────────────┐
                                    │         FEATURE ENGINEERING (×7)            │
                                    │                                             │
                                    │  Elo Ratings · GLM Quality · Box-Score Rate │
                                    │  SOS-Adjusted Win Rate · Head-to-Head       │
                                    │  Away Win Ratio · Last-14-Day Momentum      │
                                    │  → 53 features per matchup                  │
                                    └─────────────────────┬───────────────────────┘
                                                          ↓
                                    ┌─────────────────────────────────────────────┐
                                    │            MODELING PIPELINE                │
                                    │                                             │
                                    │  [Stage 1] XGBoost → Point Margin           │
                                    │  [Stage 2] Logistic Calibration → P(win)    │
                                    │  [Stage 3] Probability Tail Adjustment      │
                                    │  [Inference] Ensemble 22 LOSO Models        │
                                    └─────────────────────────────────────────────┘
```

---

## Feature Engineering

### 7 Feature Families (53 Dimensions)

| Family | Key Features | Signal |
|---|---|---|
| Box-Score Rate Stats | eFG%, OffRtg, DefRtg, TOR, ORR, DRR, AstR, FTR | Efficiency per 100 possessions (OT-normalized) |
| Elo Ratings | Custom Elo (K=80, base=1000, width=400) | Team strength trajectory, reset each year |
| GLM Team Quality | Gaussian GLM: PointDiff ~ T1 + T2 | Opponent-controlled team quality coefficient |
| SOS-Adjusted Win Rate | Two-pass opponent-weighted win rate | Schedule difficulty normalization |
| Head-to-Head | Laplace-smoothed (prior=[1,1]) matchup probability | Direct historical matchup signal |
| Momentum | Last-14-day win ratio (DayNum > 118) | Late-season form |
| Away Win Ratio | Road game performance | Neutral-site readiness proxy |

### Feature Composition

Final feature vector per matchup (53 dimensions):

| Component | Count |
|---|---|
| T1 absolute features (seed, elo, quality, 14 box-score rates) | 17 |
| T2 absolute features (mirror) | 17 |
| T1−T2 differential features | 17 |
| Head-to-head matchup probability | 1 |
| Gender flag (men/women) | 1 |

---

## Modeling Strategy

### Cross-Validation Design

**Leave-One-Season-Out (LOSO)**: Each of 22 seasons (2003–2025) is held out once. This is stricter than k-fold because it prevents any temporal leakage — the model never sees future seasons during training. All 22 models are retained and ensembled at inference: `P(win) = mean(P₁, P₂, ..., P₂₂)`.

### Hyperparameter Optimization

Three independent grid searches, each optimized for downstream Brier Score:

| Stage | Search Space | Optimal |
|---|---|---|
| XGBoost | tree depth, learning rate, subsample, colsample, regularization | Pre-tuned (see config) |
| Logistic Calibration | 13 C values × 3 solvers × 2 penalties × 2 class weights | C=0.001, L2, SAGA |
| Probability Adjustment | 5 × 16 × 5 × 16 = 6,400 configurations | upper: (0.8, +0.01), lower: (0.2, +0.12) |

---

## Scale

| Dimension | Value |
|---|---|
| Training data | 208,280 regular-season games + 2,410 tournament games |
| Prediction targets | 132,133 matchups (66,430 Men's + 65,703 Women's) |
| Engineered features | 53 per matchup |
| Cross-validation folds | 22 seasons (Leave-One-Season-Out) |
| Feature families | 7 independent signal sources |

---

## Tech Stack

**Core ML:**
- XGBoost (gradient-boosted margin regression)
- scikit-learn (Logistic Regression, grid search, evaluation)
- statsmodels (Gaussian GLM for team quality)

**Data & Visualization:**
- pandas, numpy
- matplotlib, seaborn
- tqdm (progress tracking)

---

## Environment

```bash
git clone https://github.com/<your-username>/ncaa-march-madness-2026.git
cd ncaa-march-madness-2026

pip install numpy pandas xgboost scikit-learn statsmodels matplotlib seaborn tqdm
```

Data download from Kaggle:
```bash
kaggle competitions download -c march-machine-learning-mania-2026 -p data/
```

---

## Files

| File | Description |
|---|---|
| `2026NCAA_ML.ipynb` | Complete Pipeline (single notebook) |
| `data/M*.csv` | Men's regular season, tournament, seeds |
| `data/W*.csv` | Women's regular season, tournament, seeds |
| `predictions.csv` | Final Submission (132,133 matchups) |

---

## Acknowledgments

Built for [Kaggle March Machine Learning Mania 2026](https://www.kaggle.com/competitions/march-machine-learning-mania-2026). Competition data provided by Kaggle under its competition rules.
