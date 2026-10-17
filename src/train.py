import argparse
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from build_matchups import (
    OUTPUT_FILE as MATCHUPS_FILE,
    build_matchups,
    diff_feature_names,
    load_team_features,
)
from game_features import (
    build_pair_features,
    diff_feature_names as game_diff_feature_names,
    load_pretourney_features,
    load_regular_game_features,
)

# ============================================================
# Validation seasons
#
# DEV seasons are for comparing and tuning models. HOLDOUT seasons
# are locked away and only scored once, at the end, with
#     python src/train.py --holdout
# If we tuned on the holdout too, its score would be optimistic.
# ============================================================

DEV_SEASONS = list(range(2010, 2020))
HOLDOUT_SEASONS = list(range(2021, 2026))  # no tournament in 2020

# Keeps one confident wrong pick from blowing up log loss
CLIP = 0.025

# Regular-season games need both teams to have played this many
# games, so early-season stats aren't pure noise
MIN_PRIOR_GAMES = 5

# 2003 Elo starts every team at 1500, so skip it for margin models
FIRST_MARGIN_SEASON = 2004

ALL_FEATURES = diff_feature_names()
MARGIN_FEATURES = game_diff_feature_names() + ["Location"]


@dataclass
class TrainingData:
    matchups: pd.DataFrame       # past tournament games (build_matchups.py)
    team_features: pd.DataFrame  # pre-tournament team stats + seeds
    regular: pd.DataFrame        # regular-season games, point-in-time features
    pretourney: pd.DataFrame     # pre-tournament team features (game_features.py)


def load_data():
    regular = load_regular_game_features()
    regular = regular[
        (regular["Season"] >= FIRST_MARGIN_SEASON)
        & (regular["GamesA"] >= MIN_PRIOR_GAMES)
        & (regular["GamesB"] >= MIN_PRIOR_GAMES)
    ].dropna(subset=MARGIN_FEATURES)

    return TrainingData(
        matchups=pd.read_csv(MATCHUPS_FILE),
        team_features=load_team_features(),
        regular=regular,
        pretourney=load_pretourney_features(),
    )


def add_flipped_rows(X, y):
    """
    Add every game a second time from Team B's point of view.

    Diff features change sign and the label flips, so the model sees
    each game both ways and its predictions stay symmetric:
    P(A beats B) = 1 - P(B beats A).
    """
    return (
        pd.concat([X, -X], ignore_index=True),
        pd.concat([y, 1 - y], ignore_index=True),
    )


class TourneyModel:
    """Classifier trained only on past tournament games."""

    def __init__(self, features, make_model):
        self.features = features
        self.make_model = make_model

    def fit(self, data, season):
        """Train on every tournament before `season`."""
        train = data.matchups[data.matchups["Season"] < season]
        X, y = add_flipped_rows(train[self.features], train["A_won"])

        self.model = self.make_model()
        self.model.fit(X, y)

        return self

    def predict(self, data, games):
        """P(TeamA beats TeamB) for a DataFrame with Season, TeamA, TeamB."""
        X = build_matchups(games[["Season", "TeamA", "TeamB"]], data.team_features)[self.features]
        return np.clip(self.model.predict_proba(X)[:, 1], CLIP, 1 - CLIP)


class MarginModel:
    """
    Regressor trained on ~100k regular-season games to predict the
    point margin, which keeps information a win/loss label throws away.

    The predicted margin becomes a win probability in one of two ways:
        "normal":  P = Phi(margin / sigma), where sigma is how far real
                   margins land from predictions (about 11 points)
        "stacked": logistic regression on [predicted margin, seed diff]
                   fit on past tournament games, which also lets seed
                   (unknown for regular-season games) add information
    """

    def __init__(self, make_regressor, convert):
        self.make_regressor = make_regressor
        self.convert = convert

    def fit(self, data, season):
        """
        Train on regular-season games through `season` (all played
        before that season's tournament) and tournaments before it.
        """
        train = data.regular[data.regular["Season"] <= season]
        X, y = train[MARGIN_FEATURES], train["Margin"]

        # Flipped copies: negate every diff, Location and the margin
        self.regressor = self.make_regressor()
        self.regressor.fit(pd.concat([X, -X]), pd.concat([y, -y]))

        if self.convert == "normal":
            self.sigma = np.std(y - self._raw_margin(X))

        elif self.convert == "stacked":
            past = data.matchups[data.matchups["Season"] < season]
            X_stack = self._stack_features(data, past)
            X_stack, y_stack = add_flipped_rows(X_stack, past["A_won"].reset_index(drop=True))

            self.stacker = LogisticRegression()
            self.stacker.fit(X_stack, y_stack)

        return self

    def _raw_margin(self, X):
        """Average f(x) and -f(-x) so swapping teams exactly negates the margin."""
        return (self.regressor.predict(X) - self.regressor.predict(-X)) / 2

    def predict_margin(self, data, games):
        """Predicted TeamA - TeamB margin on a neutral court."""
        pairs = build_pair_features(games[["Season", "TeamA", "TeamB"]], data.pretourney)
        return self._raw_margin(pairs[MARGIN_FEATURES])

    def _stack_features(self, data, games):
        seed_diff = build_matchups(games[["Season", "TeamA", "TeamB"]], data.team_features)["SeedDiff"]

        return pd.DataFrame({
            "PredictedMargin": self.predict_margin(data, games),
            "SeedDiff": seed_diff.to_numpy(),
        })

    def predict(self, data, games):
        """P(TeamA beats TeamB) for a DataFrame with Season, TeamA, TeamB."""
        if self.convert == "normal":
            probs = norm.cdf(self.predict_margin(data, games) / self.sigma)
        else:
            probs = self.stacker.predict_proba(self._stack_features(data, games))[:, 1]

        return np.clip(probs, CLIP, 1 - CLIP)


class EnsembleModel:
    """Average the win probabilities of several models."""

    def __init__(self, makers):
        self.makers = makers

    def fit(self, data, season):
        self.models = [make().fit(data, season) for make in self.makers]
        return self

    def predict(self, data, games):
        return np.mean([model.predict(data, games) for model in self.models], axis=0)


def scaled_lr(**kwargs):
    return lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, **kwargs))


def ridge():
    return make_pipeline(StandardScaler(), Ridge(alpha=1.0))


def margin_gbm():
    return HistGradientBoostingRegressor(
        max_depth=4,
        learning_rate=0.05,
        max_iter=300,
        min_samples_leaf=200,
        random_state=0,
    )


# Each entry builds a fresh, untrained model
MODELS = {
    # Trained on past tournament games only
    "Seed only (LR)": lambda: TourneyModel(["SeedDiff"], scaled_lr()),
    "Elo only (LR)": lambda: TourneyModel(["EloDiff"], scaled_lr()),
    "Massey only (LR)": lambda: TourneyModel(["MasseyRatingDiff"], scaled_lr()),
    "Seed + Elo + Massey (LR)": lambda: TourneyModel(
        ["SeedDiff", "EloDiff", "MasseyRatingDiff"], scaled_lr()
    ),
    "All features (LR)": lambda: TourneyModel(ALL_FEATURES, scaled_lr(C=0.1)),
    # L1 penalty drives weak features' coefficients to exactly 0, so
    # feature selection happens inside each training fold
    "All features (L1 LR)": lambda: TourneyModel(
        ALL_FEATURES, scaled_lr(l1_ratio=1.0, solver="liblinear", C=0.05)
    ),

    # Trained on regular-season games to predict point margin
    "Margin Ridge": lambda: MarginModel(ridge, "normal"),
    "Margin Ridge + Seed": lambda: MarginModel(ridge, "stacked"),
    "Margin GBM": lambda: MarginModel(margin_gbm, "normal"),
    "Margin GBM + Seed": lambda: MarginModel(margin_gbm, "stacked"),

    # A regular-season margin model and a tournament-trained model
    # make different mistakes, so averaging them can help
    "Ensemble (Margin GBM + L1 LR)": lambda: EnsembleModel([
        lambda: MarginModel(margin_gbm, "normal"),
        lambda: TourneyModel(ALL_FEATURES, scaled_lr(l1_ratio=1.0, solver="liblinear", C=0.05)),
    ]),
}


def evaluate(data, seasons):
    """Rolling validation: for each season S, train on what was known before S's tournament."""
    rows = []

    for season in seasons:
        test = data.matchups[data.matchups["Season"] == season]

        for name, make_model in MODELS.items():
            model = make_model().fit(data, season)
            pred = model.predict(data, test)

            rows.append({
                "Season": season,
                "Model": name,
                "LogLoss": log_loss(test["A_won"], pred, labels=[0, 1]),
                "Brier": brier_score_loss(test["A_won"], pred),
                "Accuracy": accuracy_score(test["A_won"], pred > 0.5),
                "Games": len(test),
            })

        print(f"  {season} done")

    return pd.DataFrame(rows)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--holdout",
        action="store_true",
        help="Score the locked holdout seasons. Run once, after all model choices are final.",
    )
    args = parser.parse_args()

    seasons = HOLDOUT_SEASONS if args.holdout else DEV_SEASONS
    label = "HOLDOUT" if args.holdout else "DEV"

    print(f"{'-'*10} Loading Data {'-'*10}")

    data = load_data()

    print(f"Tournament games:     {len(data.matchups):,}")
    print(f"Regular-season games: {len(data.regular):,} (both teams with {MIN_PRIOR_GAMES}+ prior games)")

    if args.holdout:
        print("\n*** Scoring the locked holdout. Don't change models based on these numbers. ***")

    print(f"{'-'*10} Rolling Validation, {label} seasons {seasons[0]}-{seasons[-1]} {'-'*10}")

    results = evaluate(data, seasons)

    summary = (
        results
        .groupby("Model")[["LogLoss", "Brier", "Accuracy"]]
        .mean()
        .sort_values("Brier")
    )

    print(f"\nMean across {label} seasons (lower LogLoss/Brier is better):")
    print(summary.round(4).to_string())

    print(f"\nTest games: {results.groupby('Season')['Games'].first().sum():,}")

    print("\nBrier score by season:")
    print(
        results
        .pivot(index="Season", columns="Model", values="Brier")
        [summary.index]
        .round(4)
        .to_string()
    )

    if not args.holdout:

        print(f"{'-'*10} Margin Ridge Coefficients (trained through {DEV_SEASONS[-1]}) {'-'*10}")

        model = MODELS["Margin Ridge"]().fit(data, DEV_SEASONS[-1])

        coefs = pd.Series(
            model.regressor.named_steps["ridge"].coef_,
            index=MARGIN_FEATURES,
        ).sort_values(key=abs, ascending=False)

        print("Points of margin per 1 standard deviation (positive = helps Team A):")
        print(coefs.round(2).to_string())
        print(f"Typical miss (sigma): {model.sigma:.1f} points")
