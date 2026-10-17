# March Madness Predictor

A machine learning model that predicts NCAA men's tournament games and simulates the bracket.

It predicts the point margin of any matchup from regular-season data, turns that into a win probability, and plays the bracket 10,000 times to estimate each team's chance of reaching every round.

## Results

Every model is tested season by season: to predict a season's tournament, it's trained only on data from before that tournament.

| Model | Dev 2010–2019 (667 games) | Holdout 2021–2025 (334 games) |
|---|---|---|
| **Margin GBM (final model)** | 70.2% · Brier 0.188 | **72.2% · Brier 0.196** |
| Tournament-only logistic regression (L1) | 70.5% · Brier 0.189 | 70.3% · Brier 0.198 |
| Always pick the better seed | 69.1% · Brier 0.201 | 69.8% · Brier 0.202 |

- **Dev seasons** were used to compare models and pick the final one.
- **Holdout seasons** were scored once, after the final model was chosen, to get an unbiased result.
- Single games are close to unpredictable. Even well-regarded rating systems and betting markets top out around 73–75% accuracy, so beating seeds by 1–3 points is a meaningful result. With only 334 holdout games, the 72.2% has a margin of error of roughly ±2.5 points.

## Bracket backtest: the last 10 tournaments

For each tournament from 2015 to 2025 (none in 2020), the model filled out a bracket using only data from before that tournament. It picked the more likely winner of every game, the way a person fills out a bracket. Each bracket was then graded against what actually happened.

| Year | Set | Picked champion | Actual champion | Model's odds for actual champion | Final Four correct | Bracket games correct |
|---|---|---|---|---|---|---|
| 2015 | dev | Kentucky | Duke | 6.2% | 1/4 | 47/63 |
| 2016 | dev | Kansas | Villanova | 6.9% | 1/4 | 40/63 |
| 2017 | dev | Villanova | North Carolina | 12.8% | 2/4 | 43/63 |
| 2018 | dev | Virginia | Villanova | 18.2% | 1/4 | 38/63 |
| 2019 | dev | **Virginia** | Virginia | 21.9% | 1/4 | 44/63 |
| 2021 | holdout | Gonzaga | Baylor | 12.6% | 2/4 | 37/62* |
| 2022 | holdout | Gonzaga | Kansas | 6.3% | 1/4 | 33/63 |
| 2023 | holdout | Houston | Connecticut | 4.9% | 0/4 | 37/63 |
| 2024 | holdout | **Connecticut** | Connecticut | 23.7% | 2/4 | 40/63 |
| 2025 | holdout | Duke | Florida | 15.2% | **4/4** | 51/63 |

\*One 2021 game (Oregon vs. VCU) was a COVID no-contest and isn't counted.

**Totals compared with other ways of filling out a bracket:**

| | Champions correct | Final Four teams correct | Bracket games correct | Avg. ESPN score (max 1920) | Avg. odds given to actual champion |
|---|---|---|---|---|---|
| **This model** | **2 of 10** | 15 of 40 | **410 of 629 (65.2%)** | **899** | **12.9%** |
| Always pick the better seed | 2 of 10† | **18 of 40** | 397 of 629 (63.1%) | 885 | 8.3% |
| Random picks (expected) | 0.2 of 10 | 2.5 of 40 | ~34% | — | 1.6% |

†When two teams have the same seed (for example, 1-seeds meeting in the Final Four), picking by seed is a coin flip, so its champion picks are partly luck.

**What this shows:**

- **The model picked the champion in 2 of the last 10 years** (2019 Virginia and 2024 Connecticut). It got all four 2025 Final Four teams right.
- **It got more bracket games right than picking by seed** (65.2% vs 63.1%) and scored slightly higher under ESPN scoring.
- **Its biggest advantage is in probabilities.** On average it gave the eventual champion a 12.9% chance before the tournament, versus 8.3% for seed-only, so its odds were closer to what happened.
- **Picking by seed got more Final Four teams right** (18 vs 15). Taking the four 1-seeds is hard to beat for the Final Four alone. Over just 10 years, the difference could easily be luck either way.
- **Bracket games (65%) score lower than single-game accuracy (71%)** because mistakes cascade. Once a pick loses, every later game it was picked to win is also wrong.
- 2015–2019 are dev seasons, which were used to choose the model, so those years may be slightly optimistic. 2021–2025 are the untouched holdout.

## How it works

1. **Team stats** (`feature_engineering.py`): season stats for every team from regular-season box scores. Includes per-possession efficiency, the Four Factors (eFG%, turnover rate, rebound rate, free-throw rate) and recent form over the last 10 games.
2. **Elo ratings** (`ratings.py`): a rating updated after every game, adjusted for home court and margin of victory. Settings were tuned on regular-season games only.
3. **Massey consensus** (`massey.py`): the average of five computer ranking systems (KenPom, Moore, Dolphin, Colley, Whitlock), using only rankings published before the tournament.
4. **Point-in-time game features** (`game_features.py`): for about 125,000 regular-season games, each team's stats, Elo and Massey rating **using only games played before that day**. This gives far more training data than the roughly 1,400 tournament games.
5. **Training and validation** (`train.py`): compares 11 models, from a seed-only baseline to gradient boosting.
   - The final model is a gradient-boosted regressor that predicts point margin.
   - The margin becomes a win probability using a normal distribution. The model's typical miss is about 11 points.
6. **Bracket simulation** (`simulate.py`): predicts every possible matchup, then plays the bracket (including the First Four) 10,000 times.

## Key findings

- **Elo and the Massey consensus do most of the work.** On its own, either one matches a 21-feature model built from box-score stats.
- **Predicting point margin works better than predicting win/loss.** Training on regular-season margins beat every model trained only on tournament games.
- **Seed adds nothing once the ratings are good.** Adding it to the margin models made them slightly worse.
- **More complex isn't automatically better.** On tournament data alone, gradient boosting never beat logistic regression.

## Preventing data leakage

The scripts check for leakage automatically:

- Team features use only regular-season games, never tournament results.
- Point-in-time features are verified: a team's Nth game sees exactly N−1 earlier games.
- A season's Elo is verified to be unchanged when that season's tournament games are removed.
- Massey rankings are limited to day 133 or earlier (the tournament starts on day 134).
- Teams are ordered by TeamID, not by winner, so the row order can't reveal the result.

## Data

From Kaggle's [March Machine Learning Mania](https://www.kaggle.com/competitions) competition. Place the files in `data/raw/`:

| File | In repo? |
|---|---|
| `MRegularSeasonDetailedResults.csv`, `MNCAATourneyDetailedResults.csv`, `MNCAATourneySeeds.csv`, `MNCAATourneySlots.csv`, `MSeasons.csv`, `MTeams.csv` | Yes |
| `MMasseyOrdinals.csv` | **No.** Too big for GitHub (129 MB), so download it from Kaggle. |

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

`simulate.py` prints each team's odds of reaching every round and the most likely bracket. It saves `bracket_odds_<season>.csv` and `bracket_picks_<season>.csv` to `data/processed/`.

## Project structure

```
data/
  raw/                     Kaggle data
  processed/               generated by the scripts
src/
  data_collection.py       quick look at the raw files
  feature_engineering.py   team season stats
  ratings.py               Elo ratings
  massey.py                Massey consensus ratings
  build_matchups.py        tournament games -> training rows
  game_features.py         point-in-time features for regular-season games
  train.py                 models + season-by-season validation
  simulate.py              Monte Carlo bracket simulation
  backtest.py              grades past brackets against real results
```
