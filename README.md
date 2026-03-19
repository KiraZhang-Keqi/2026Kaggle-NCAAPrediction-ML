# 🏀 March Machine Learning Mania 2026 — NCAA Tournament Prediction

Predicting outcomes of the 2026 NCAA Men's and Women's Basketball Tournaments for the [Kaggle March Machine Learning Mania 2026](https://www.kaggle.com/competitions/march-machine-learning-mania-2026) competition.

## Project Overview

This project builds a two-stage machine learning pipeline to forecast win probabilities for every possible matchup in the 2026 NCAA March Madness tournaments (both men's and women's brackets). The model combines rich feature engineering from historical game data with an XGBoost regression model and a calibrated logistic layer to produce well-tuned probability estimates, optimized for the Brier score metric.

## Data

All data is sourced from the Kaggle competition dataset, which includes:

- **Game results** — Regular season and tournament box scores (points, field goals, rebounds, assists, turnovers, etc.) dating back to 2003 (men) and 2010 (women).
- **Tournament seeds** — Official NCAA seedings since 1985.
- **Massey Ordinals** — Weekly rankings from dozens of rating systems (Pomeroy, Sagarin, RPI, etc.).
- **Supplementary data** — Coaches, conference affiliations, game locations, and more.

We filter to seasons ≥ 2003 to ensure detailed box-score statistics are available for all training data.

## Feature Engineering

Features are constructed at the team-season level and then joined for each possible matchup. The pipeline progressively builds features from simple to complex:

### Basic Features
- **Tournament seed** and **seed difference** between teams.
- **Gender indicator** (`men_women`) to jointly model both tournaments.

### Performance Rates (Season Averages)
- **Effective FG%** (`eFG`) — Accounts for the extra value of three-pointers.
- **3-Point Rate** (`FGR3`) — Share of field goal attempts from three.
- **Free Throw Rate** (`FTR`) — Free throws attempted per field goal attempt.
- **Assist Rate** (`AstR`), **Turnover Rate** (`TOR`).
- **Offensive / Defensive Rebound Rate** (`ORR`, `DRR`).
- **Offensive & Defensive Rating** (`OffRtg`, `DefRtg`) — Points per possession.
- **Average score** and **average point differential**.

### Situational & Momentum Features
- **Win ratio in last 14 days** of the regular season — Captures late-season form.
- **Away win probability** — How well a team performs on the road.
- **Head-to-head matchup history** — Laplace-smoothed win probability from regular season meetings.

### Advanced Ratings
- **Elo ratings** — Custom Elo system (K=80, width=400) updated game-by-game through the regular season.
- **GLM quality metric** — A strength-of-schedule-adjusted quality score derived via a generalized linear model (statsmodels). Only games involving tournament-caliber teams are used to fit the model, producing a single quality number per team-season.

### Difference Features
For each rate/metric above, we also compute the **T1 − T2 difference**, giving the model direct access to pairwise comparisons.

## Model Architecture

The prediction pipeline has two stages:

### Stage 1: XGBoost Point Margin Regression
- **Target**: Point differential (continuous).
- **Validation**: Leave-one-season-out cross-validation across all historical tournament games.
- **Key hyperparameters**: `num_parallel_tree=10`, `max_depth=3`, `eta=0.01`, `subsample=0.35`, 500 boosting rounds.

### Stage 2: Logistic Calibration + Probability Adjustment
1. **Logistic Regression** — Maps the predicted margin to a win probability. Grid-searched over C, solver, penalty, and class weights; optimized for Brier score.
2. **Threshold-based probability adjustment** — A post-processing step that nudges extreme probabilities (very high or very low) to improve calibration. Parameters are grid-searched on the OOF predictions.

### Ensemble
Final predictions average the probability outputs across all leave-one-season-out models, producing a robust ensemble prediction for each matchup.

## Repo Structure

```
.
├── NCAA_version7.ipynb   # Main notebook: data loading → feature engineering → training → submission
├── predictions.csv       # Final submission file (132K matchup probabilities)
└── README.md
```

## How to Run

1. Download the competition data from [Kaggle](https://www.kaggle.com/competitions/march-machine-learning-mania-2026/data) and place CSV files in the working directory.
2. Install dependencies:
   ```bash
   pip install numpy pandas scikit-learn xgboost catboost statsmodels matplotlib seaborn tqdm
   ```
3. Run `NCAA_version7.ipynb` end-to-end. The final cell writes `predictions.csv`.

## Results

The model is evaluated using **Brier score** (lower is better) via leave-one-season-out cross-validation on all historical tournament games from 2003–2025. The combination of margin regression + logistic calibration + probability adjustment consistently outperforms seed-only baselines and single-model approaches.

## Key Takeaways

- Predicting the **point margin first** and then converting to probability (via logistic regression) outperforms direct probability classification — it provides a richer training signal.
- The **GLM quality metric** adds meaningful information beyond Elo and seeds, especially for mid-seed matchups where traditional rankings are noisy.
- **Late-season momentum** (14-day win ratio) and **away performance** capture team dynamics that season-long averages miss.
- **Probability calibration** via threshold adjustment is a simple but effective post-processing trick for Brier score optimization.

## Acknowledgments

- [Kenneth Massey](https://masseyratings.com/) for the historical ranking data.
- [Jeff Sonas / Sonas Consulting](https://www.sonasconsulting.com/) for dataset assembly.
- The Kaggle community — in particular, the public notebooks and solutions from prior years that inspired our feature engineering approach.

## License

This project uses data provided under the [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) license. Code is provided as-is for educational and competition purposes.
