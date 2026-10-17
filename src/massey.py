import os
import numpy as np
import pandas as pd

# Paths are resolved from the project root so this runs from any directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MASSEY_FILE = os.path.join(PROJECT_ROOT, "data/raw/MMasseyOrdinals.csv")

OUTPUT_FILE = os.path.join(PROJECT_ROOT, "data/processed/massey_ratings.csv")

# ============================================================
# Ranking systems
#
# The only five systems that rank every tournament team in every
# season 2003-2026. Chosen on coverage alone, NOT on how well they
# predict, so picking them doesn't peek at any test results.
#
#   POM = KenPom, MOR = Moore, DOL = Dolphin,
#   COL = Colley, WLK = Whitlock
# ============================================================

SYSTEMS = ["POM", "MOR", "DOL", "COL", "WLK"]

# Rankings dated day 133 are the final pre-tournament rankings;
# the tournament starts on day 134.
LAST_PRE_TOURNAMENT_DAY = 133


def rank_to_rating(rank):
    """
    Convert an ordinal rank into a rating where gaps mean something.

    The gap between #1 and #10 matters far more than #200 vs #210,
    which a raw rank treats the same. This curve (popular in Kaggle
    March Madness solutions) is steep at the top and flat lower down.
    """
    return 100 - 4 * np.log(rank + 1) - rank / 22


def build_daily_ratings(massey):
    """
    One row per Season / TeamID / day that any chosen system published.

    Each system's most recent ranking is carried forward until it
    publishes again, so every row is "what each system said as of
    that day". MasseyRating averages the systems available that day.
    """
    massey = massey[
        massey["SystemName"].isin(SYSTEMS)
        & (massey["RankingDayNum"] <= LAST_PRE_TOURNAMENT_DAY)
    ]

    daily = (
        massey
        .pivot_table(
            index=["Season", "TeamID", "RankingDayNum"],
            columns="SystemName",
            values="OrdinalRank",
            observed=True,
        )
        .sort_index()
    )

    daily = daily.groupby(level=["Season", "TeamID"]).ffill()

    ratings = rank_to_rating(daily)

    daily = pd.DataFrame({
        "MasseyRating": ratings.mean(axis=1),
        "POMRating": ratings["POM"],
        "MasseySystems": ratings.notna().sum(axis=1),
    }).reset_index()

    return daily


def pre_tournament_ratings(daily):
    """Each team-season's final ratings before the tournament."""
    return (
        daily
        .sort_values("RankingDayNum")
        .groupby(["Season", "TeamID"])
        .tail(1)
        [["Season", "TeamID", "MasseyRating", "POMRating"]]
        .reset_index(drop=True)
    )


def ratings_as_of(games, daily, team_column):
    """
    Attach each team's ratings as of the game day.

    Uses the latest ranking with RankingDayNum <= DayNum. Kaggle defines
    RankingDayNum as the first day a ranking may be used to predict
    games, so a day-d ranking is safe for a day-d game.
    """
    left = games[["Season", "DayNum", team_column]].reset_index()

    merged = pd.merge_asof(
        left.sort_values("DayNum"),
        daily.rename(columns={"TeamID": team_column}).sort_values("RankingDayNum"),
        left_on="DayNum",
        right_on="RankingDayNum",
        by=["Season", team_column],
        direction="backward",
    )

    return merged.set_index("index").loc[games.index, ["MasseyRating", "POMRating"]]


def load_daily_ratings():
    return pd.read_csv(OUTPUT_FILE)


if __name__ == "__main__":

    print(f"{'-'*10} Loading Data {'-'*10}")

    massey = pd.read_csv(
        MASSEY_FILE,
        dtype={"SystemName": "category"},
    )

    print(f"Ranking rows: {len(massey):,}")
    print(f"Systems used: {', '.join(SYSTEMS)}")

    print(f"{'-'*10} Building Daily Ratings {'-'*10}")

    daily = build_daily_ratings(massey)

    print(f"Daily team ratings: {len(daily):,}")

    if daily["RankingDayNum"].max() > LAST_PRE_TOURNAMENT_DAY:
        raise ValueError("DATA LEAKAGE DETECTED: rankings from tournament days included.")

    print(f"✓ No rankings after day {LAST_PRE_TOURNAMENT_DAY} (tournament starts day 134).")

    pre_tournament = pre_tournament_ratings(daily)

    seeds = pd.read_csv(os.path.join(PROJECT_ROOT, "data/raw/MNCAATourneySeeds.csv"))
    seeds = seeds[seeds["Season"] >= daily["Season"].min()]

    covered = seeds.merge(pre_tournament, on=["Season", "TeamID"], how="left")

    if covered["MasseyRating"].isna().any():
        raise ValueError(
            f"{covered['MasseyRating'].isna().sum()} tournament teams have no Massey rating."
        )

    print("✓ Every tournament team has a pre-tournament Massey rating.")

    latest_season = daily["Season"].max()

    print(f"{'-'*10} Top 10 Massey, {latest_season} {'-'*10}")

    teams = pd.read_csv(os.path.join(PROJECT_ROOT, "data/raw/MTeams.csv"))

    print(
        pre_tournament[pre_tournament["Season"] == latest_season]
        .merge(teams[["TeamID", "TeamName"]], on="TeamID")
        .sort_values("MasseyRating", ascending=False)
        .head(10)[["TeamName", "MasseyRating", "POMRating"]]
        .round(1)
        .to_string(index=False)
    )

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    daily.to_csv(OUTPUT_FILE, index=False)

    print(f"\nSaved daily Massey ratings to:\n{OUTPUT_FILE}")
