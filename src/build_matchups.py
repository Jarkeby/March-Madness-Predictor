import os
import pandas as pd

from massey import load_daily_ratings, pre_tournament_ratings

# Paths are resolved from the project root so this runs from any directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TEAM_STATS_FILE = os.path.join(PROJECT_ROOT, "data/processed/team_season_stats.csv")
ELO_FILE = os.path.join(PROJECT_ROOT, "data/processed/elo_ratings.csv")
SEEDS_FILE = os.path.join(PROJECT_ROOT, "data/raw/MNCAATourneySeeds.csv")
TOURNAMENT_RESULTS_FILE = os.path.join(PROJECT_ROOT, "data/raw/MNCAATourneyDetailedResults.csv")

OUTPUT_FILE = os.path.join(PROJECT_ROOT, "data/processed/tourney_matchups.csv")

# Team-season columns turned into A - B difference features.
# Raw totals (PointsFor, FGM, ...) are left out because they depend on
# games played and duplicate the per-game / per-possession versions.
FEATURE_COLUMNS = [
    "Seed",
    "Elo",
    "MasseyRating",
    "POMRating",
    "WinPct",
    "PointDiff",
    "EFGPct",
    "ThreePtPct",
    "FTPct",
    "FTRate",
    "OppEFGPct",
    "OppThreePtPct",
    "OppFTRate",
    "TORate",
    "OppTORate",
    "OffensiveReboundRate",
    "DefensiveReboundRate",
    "PossessionsPerGame",
    "OffensiveEfficiency",
    "DefensiveEfficiency",
    "NetEfficiency",
    "RecentWinPct",
    "RecentNetEfficiency",
]


def load_team_features():
    """Team-season stats joined with Elo, Massey and seeds, one row per Season/TeamID."""
    stats = pd.read_csv(TEAM_STATS_FILE)
    elo = pd.read_csv(ELO_FILE)
    massey = pre_tournament_ratings(load_daily_ratings())
    seeds = pd.read_csv(SEEDS_FILE)

    stats = stats.merge(elo, on=["Season", "TeamID"], how="left")
    stats = stats.merge(massey, on=["Season", "TeamID"], how="left")

    # Turnovers per possession is pace-independent, unlike TOPG
    stats["TORate"] = stats["TOPG"] / stats["PossessionsPerGame"]
    stats["OppTORate"] = stats["OppTOPG"] / stats["OpponentPossessionsPerGame"]

    # Seed strings look like "W01" or "X16a" (a/b = First Four) -> 1, 16
    seeds["Seed"] = seeds["Seed"].str[1:3].astype(int)

    features = stats.merge(
        seeds[["Season", "TeamID", "Seed"]],
        on=["Season", "TeamID"],
        how="inner"
    )

    return features[["Season", "TeamID", "TeamName"] + FEATURE_COLUMNS]


def build_matchups(games, team_features):
    """
    Turn a table of games (Season, TeamA, TeamB) into diff-feature rows.

    TeamA is always the lower TeamID so the winner is not always in the
    same slot, which would let the model "learn" the label from the order.
    """
    matchups = games.copy()

    for side in ["A", "B"]:
        side_features = team_features.rename(
            columns={
                "TeamID": f"Team{side}",
                "TeamName": f"TeamName{side}",
                **{col: f"{col}{side}" for col in FEATURE_COLUMNS},
            }
        )

        matchups = matchups.merge(
            side_features,
            on=["Season", f"Team{side}"],
            how="left"
        )

    for col in FEATURE_COLUMNS:
        matchups[f"{col}Diff"] = matchups[f"{col}A"] - matchups[f"{col}B"]

    return matchups


def diff_feature_names():
    return [f"{col}Diff" for col in FEATURE_COLUMNS]


if __name__ == "__main__":

    print(f"{'-'*10} Loading Data {'-'*10}")

    team_features = load_team_features()
    tournament = pd.read_csv(TOURNAMENT_RESULTS_FILE)

    print(f"Seeded team-seasons: {len(team_features):,}")
    print(f"Tournament games:    {len(tournament):,}")

    print(f"{'-'*10} Building Matchups {'-'*10}")

    games = pd.DataFrame({
        "Season": tournament["Season"],
        "DayNum": tournament["DayNum"],
        "TeamA": tournament[["WTeamID", "LTeamID"]].min(axis=1),
        "TeamB": tournament[["WTeamID", "LTeamID"]].max(axis=1),
    })
    games["A_won"] = (tournament["WTeamID"] == games["TeamA"]).astype(int)

    matchups = build_matchups(games, team_features)

    missing = matchups[diff_feature_names()].isna().any(axis=1)

    if missing.any():
        raise ValueError(
            f"{missing.sum()} tournament games are missing team features "
            "(no seed, Elo, Massey, or regular-season stats):\n"
            f"{matchups.loc[missing, ['Season', 'TeamA', 'TeamB']].head(20)}"
        )

    print("✓ Every tournament game has features for both teams.")

    if len(matchups) != len(tournament):
        raise ValueError(
            f"Expected {len(tournament):,} matchups, got {len(matchups):,}."
        )

    print(f"✓ Built {len(matchups):,} matchups.")
    print(f"A_won rate: {matchups['A_won'].mean():.3f} (should be near 0.5)")

    column_order = (
        ["Season", "DayNum", "TeamA", "TeamNameA", "TeamB", "TeamNameB", "A_won"]
        + [f"{col}A" for col in FEATURE_COLUMNS]
        + [f"{col}B" for col in FEATURE_COLUMNS]
        + diff_feature_names()
    )

    matchups = matchups[column_order]

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    matchups.to_csv(OUTPUT_FILE, index=False)

    print(f"\nSaved matchups to:\n{OUTPUT_FILE}")
    print(f"Final shape: {matchups.shape}")
