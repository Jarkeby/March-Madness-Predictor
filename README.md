# March Madness Predictor

A machine learning model that predicts NCAA men's tournament games and simulates the bracket.

## Results

| Model | Dev 2010–2019 | Holdout 2021–2025 |
|---|---|---|
| **Margin GBM (final model)** | 70.2% · Brier 0.188 | **72.2% · Brier 0.196** |
| Tournament-only logistic regression (L1) | 70.5% · Brier 0.189 | 70.3% · Brier 0.198 |
| Always pick the better seed | 69.1% · Brier 0.201 | 69.8% · Brier 0.202 |

## Bracket backtest: the last 10 tournaments

Models prediction for the last 10 tournaments compared to the actual results.

| Year | Set | Picked champion | Actual champion | Model's odds for actual champion | Final Four correct | Bracket games correct |
|---|---|---|---|---|---|---|
| 2015 | dev | Kentucky | Duke | 6.2% | 1/4 | 47/63 |
| 2016 | dev | Kansas | Villanova | 6.9% | 1/4 | 40/63 |
| 2017 | dev | Villanova | North Carolina | 12.8% | 2/4 | 43/63 |
| 2018 | dev | Virginia | Villanova | 18.2% | 1/4 | 38/63 |
| 2019 | dev | **Virginia** | Virginia | 21.9% | 1/4 | 44/63 |
| 2021 | holdout | Gonzaga | Baylor | 12.6% | 2/4 | 37/62 |
| 2022 | holdout | Gonzaga | Kansas | 6.3% | 1/4 | 33/63 |
| 2023 | holdout | Houston | Connecticut | 4.9% | 0/4 | 37/63 |
| 2024 | holdout | **Connecticut** | Connecticut | 23.7% | 2/4 | 40/63 |
| 2025 | holdout | Duke | Florida | 15.2% | **4/4** | 51/63 |

**Compared to always picking the better seed:**

| | Champions correct | Final Four teams correct | Bracket games correct | Avg. ESPN score (max 1920) | Avg. odds given to actual champion |
|---|---|---|---|---|---|
| **This model** | **2 of 10** | 15 of 40 | **410 of 629 (65.2%)** | **899** | **12.9%** |
| Always pick the better seed | 2 of 10 | **18 of 40** | 397 of 629 (63.1%) | 885 | 8.3% |

## Data

From Kaggle's [March Machine Learning Mania](https://www.kaggle.com/competitions) competition. Place the files in `data/raw/`:

| File | In repo? |
|---|---|
| `MRegularSeasonDetailedResults.csv`, `MNCAATourneyDetailedResults.csv`, `MNCAATourneySeeds.csv`, `MNCAATourneySlots.csv`, `MSeasons.csv`, `MTeams.csv` | Yes |
| `MMasseyOrdinals.csv` | **No.** Too big for GitHub |

## Setup

```
pip install -r requirements.txt
```

## Running

Run the scripts from the project folder, in this order:

```
python src/feature_engineering.py   # team season stats
python src/ratings.py               # Elo ratings
python src/massey.py                # Massey consensus ratings
python src/build_matchups.py        # tournament training table
python src/game_features.py         # point-in-time regular-season features
python src/train.py                 # compare models on dev seasons
python src/simulate.py              # simulate the bracket
python src/backtest.py              # grade past brackets (last 10 tournaments)
```

- `python src/train.py --holdout` scores the holdout seasons. It's meant to be run once, after all model choices are final.
- To simulate a different year, change `SEASON` in `simulate.py`.
- To use a different model, set `FINAL_MODEL` to any name in `train.py`'s `MODELS`.
