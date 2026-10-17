import os
import numpy as np
import pandas as pd

from massey import load_daily_ratings, pre_tournament_ratings, ratings_as_of

# Paths are resolved from the project root so this runs from any directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

REGULAR_SEASON_RESULTS_FILE = os.path.join(PROJECT_ROOT, "data/raw/MRegularSeasonDetailedResults.csv")
ELO_FILE = os.path.join(PROJECT_ROOT, "data/processed/elo_ratings.csv")
ELO_PREGAME_FILE = os.path.join(PROJECT_ROOT, "data/processed/elo_pregame.csv")
TEAM_STATS_FILE = os.path.join(PROJECT_ROOT, "data/processed/team_season_stats.csv")

REGULAR_GAMES_OUTPUT_FILE = os.path.join(PROJECT_ROOT, "data/processed/regular_game_features.csv")
PRETOURNEY_OUTPUT_FILE = os.path.join(PROJECT_ROOT, "data/processed/pretourney_team_features.csv")

# ============================================================
# Point-in-time features
#
# Every regular-season game gets each team's stats, Elo and Massey
# rating using ONLY games played before that day. This turns ~125k
# regular-season games into training data without leaking a game's
# own result into its features.
#
# Tournament games get the same features computed from the full
# regular season (what is known on Selection Sunday).
# ============================================================

RECENT_GAMES = 10

# Season totals summed per team, then turned into rates
SUM_COLUMNS = [
    "Win", "Games",
    "PointsFor", "PointsAgainst",
    "FGM", "FGA", "FGM3", "FTA", "OR", "DR", "TO",
    "Opp_FGM", "Opp_FGA", "Opp_FGM3", "Opp_FTA", "Opp_OR", "Opp_DR", "Opp_TO",
    "Possessions", "Opp_Possessions",
]

TEAM_FEATURES = [
    "Elo",
    "MasseyRating",
    "POMRating",
    "WinPct",
    "NetEfficiency",
    "OffensiveEfficiency",
    "DefensiveEfficiency",
    "EFGPct",
    "OppEFGPct",
    "TORate",
    "OppTORate",
    "OffensiveReboundRate",
    "DefensiveReboundRate",
    "FTRate",
    "OppFTRate",
    "RecentWinPct",
    "RecentNetEfficiency",
]


def diff_feature_names():
    return [f"{col}Diff" for col in TEAM_FEATURES]


def team_game_records(regular):
    """Two rows per game, one from each team's point of view."""
    box_score = ["Score", "FGM", "FGA", "FGM3", "FTA", "OR", "DR", "TO"]

    records = []

    for team, opp, win in [("W", "L", 1), ("L", "W", 0)]:
        side = pd.DataFrame({
            # Row number of the game; (Season, DayNum, TeamID) is not unique
            # because a team occasionally plays twice in one day
            "GameIndex": regular.index,
            "Season": regular["Season"],
            "DayNum": regular["DayNum"],
            "TeamID": regular[f"{team}TeamID"],
            "Win": win,
            "Games": 1,
            "PointsFor": regular[f"{team}Score"],
            "PointsAgainst": regular[f"{opp}Score"],
        })

        for stat in box_score[1:]:
            side[stat] = regular[f"{team}{stat}"]
            side[f"Opp_{stat}"] = regular[f"{opp}{stat}"]

        records.append(side)

    team_games = pd.concat(records, ignore_index=True)

    # Same possession estimate as feature_engineering.py
    team_games["Possessions"] = (
        team_games["FGA"] + 0.475 * team_games["FTA"] - team_games["OR"] + team_games["TO"]
    )
    team_games["Opp_Possessions"] = (
        team_games["Opp_FGA"] + 0.475 * team_games["Opp_FTA"]
        - team_games["Opp_OR"] + team_games["Opp_TO"]
    )

    return team_games.sort_values(["Season", "TeamID", "DayNum", "GameIndex"]).reset_index(drop=True)


def divide(numerator, denominator):
    """Division that gives NaN (not inf) when a team has no games yet."""
    return (numerator / denominator).replace([np.inf, -np.inf], np.nan)


def stats_from_sums(sums):
    """Season rate stats from summed box-score columns."""
    off_eff = divide(sums["PointsFor"], sums["Possessions"]) * 100
    def_eff = divide(sums["PointsAgainst"], sums["Opp_Possessions"]) * 100

    return pd.DataFrame({
        "WinPct": divide(sums["Win"], sums["Games"]),
        "NetEfficiency": off_eff - def_eff,
        "OffensiveEfficiency": off_eff,
        "DefensiveEfficiency": def_eff,
        "EFGPct": divide(sums["FGM"] + 0.5 * sums["FGM3"], sums["FGA"]),
        "OppEFGPct": divide(sums["Opp_FGM"] + 0.5 * sums["Opp_FGM3"], sums["Opp_FGA"]),
        "TORate": divide(sums["TO"], sums["Possessions"]),
        "OppTORate": divide(sums["Opp_TO"], sums["Opp_Possessions"]),
        "OffensiveReboundRate": divide(sums["OR"], sums["OR"] + sums["Opp_DR"]),
        "DefensiveReboundRate": divide(sums["DR"], sums["DR"] + sums["Opp_OR"]),
        "FTRate": divide(sums["FTA"], sums["FGA"]),
        "OppFTRate": divide(sums["Opp_FTA"], sums["Opp_FGA"]),
    }, index=sums.index)


def recent_from_sums(sums):
    """Recent-form stats from box-score columns summed over the last few games."""
    return pd.DataFrame({
        "RecentWinPct": divide(sums["Win"], sums["Games"]),
        "RecentNetEfficiency": (
            divide(sums["PointsFor"], sums["Possessions"]) * 100
            - divide(sums["PointsAgainst"], sums["Opp_Possessions"]) * 100
        ),
    }, index=sums.index)


def pregame_team_features(team_games):
    """
    Each team's stats going INTO every game.

    cumsum - current row = totals over all earlier games this season.
    Subtracting the same total from RECENT_GAMES games back gives the
    totals over just the last RECENT_GAMES games.
    """
    keys = [team_games["Season"], team_games["TeamID"]]

    before = team_games.groupby(keys)[SUM_COLUMNS].cumsum() - team_games[SUM_COLUMNS]
    recent = before - before.groupby(keys).shift(RECENT_GAMES).fillna(0)

    return pd.concat([
        team_games[["GameIndex", "Season", "DayNum", "TeamID"]],
        before["Games"].rename("GamesPlayed"),
        stats_from_sums(before),
        recent_from_sums(recent),
    ], axis=1)


def pretourney_team_features(team_games, elo, massey_daily):
    """Each team's stats after the full regular season, plus Elo and Massey."""
    keys = ["Season", "TeamID"]

    totals = team_games.groupby(keys)[SUM_COLUMNS].sum()
    recent = team_games.groupby(keys).tail(RECENT_GAMES).groupby(keys)[SUM_COLUMNS].sum()

    features = pd.concat([
        totals["Games"].rename("GamesPlayed"),
        stats_from_sums(totals),
        recent_from_sums(recent),
    ], axis=1).reset_index()

    features = features.merge(elo, on=keys, how="left")
    features = features.merge(pre_tournament_ratings(massey_daily), on=keys, how="left")

    return features[keys + ["GamesPlayed"] + TEAM_FEATURES]


def regular_game_features(regular, team_games, elo_pregame, massey_daily):
    """One row per regular-season game with A - B pre-game feature diffs."""
    games = pd.DataFrame({
        "GameIndex": regular.index,
        "Season": regular["Season"],
        "DayNum": regular["DayNum"],
        "TeamA": regular[["WTeamID", "LTeamID"]].min(axis=1),
        "TeamB": regular[["WTeamID", "LTeamID"]].max(axis=1),
    })

    a_won = regular["WTeamID"] == games["TeamA"]
    games["A_won"] = a_won.astype(int)
    games["Margin"] = np.where(a_won, 1, -1) * (regular["WScore"] - regular["LScore"])

    # Home court from Team A's point of view: +1 home, -1 away, 0 neutral
    winner_location = regular["WLoc"].map({"H": 1, "A": -1, "N": 0})
    games["Location"] = np.where(a_won, winner_location, -winner_location)

    pregame = pregame_team_features(team_games).drop(columns=["Season", "DayNum"])

    elo = elo_pregame.assign(
        EloA=np.where(elo_pregame["WTeamID"] < elo_pregame["LTeamID"], elo_pregame["WElo"], elo_pregame["LElo"]),
        EloB=np.where(elo_pregame["WTeamID"] < elo_pregame["LTeamID"], elo_pregame["LElo"], elo_pregame["WElo"]),
        TeamA=elo_pregame[["WTeamID", "LTeamID"]].min(axis=1),
        TeamB=elo_pregame[["WTeamID", "LTeamID"]].max(axis=1),
    )[["Season", "DayNum", "TeamA", "TeamB", "EloA", "EloB"]]

    games = games.merge(elo, on=["Season", "DayNum", "TeamA", "TeamB"], how="left")

    for side in ["A", "B"]:
        side_features = pregame.rename(columns={
            "TeamID": f"Team{side}",
            "GamesPlayed": f"Games{side}",
            **{col: f"{col}{side}" for col in TEAM_FEATURES if col in pregame.columns},
        })

        games = games.merge(side_features, on=["GameIndex", f"Team{side}"], how="left")

        massey = ratings_as_of(games, massey_daily, f"Team{side}")
        games[f"MasseyRating{side}"] = massey["MasseyRating"].to_numpy()
        games[f"POMRating{side}"] = massey["POMRating"].to_numpy()

    for col in TEAM_FEATURES:
        games[f"{col}Diff"] = games[f"{col}A"] - games[f"{col}B"]

    return games[
        ["Season", "DayNum", "TeamA", "TeamB", "A_won", "Margin", "Location", "GamesA", "GamesB"]
        + diff_feature_names()
    ]


def build_pair_features(games, pretourney):
    """
    Diff features for neutral-site games (tournament or simulated)
    from pre-tournament team features. `games` needs Season, TeamA, TeamB.
    """
    pairs = games.copy()

    for side in ["A", "B"]:
        pairs = pairs.merge(
            pretourney.rename(columns={
                "TeamID": f"Team{side}",
                "GamesPlayed": f"Games{side}",
                **{col: f"{col}{side}" for col in TEAM_FEATURES},
            }),
            on=["Season", f"Team{side}"],
            how="left",
        )

    for col in TEAM_FEATURES:
        pairs[f"{col}Diff"] = pairs[f"{col}A"] - pairs[f"{col}B"]

    pairs["Location"] = 0

    return pairs


def load_regular_game_features():
    return pd.read_csv(REGULAR_GAMES_OUTPUT_FILE)


def load_pretourney_features():
    return pd.read_csv(PRETOURNEY_OUTPUT_FILE)


if __name__ == "__main__":

    print(f"{'-'*10} Loading Data {'-'*10}")

    regular = pd.read_csv(REGULAR_SEASON_RESULTS_FILE)
    elo = pd.read_csv(ELO_FILE)
    elo_pregame = pd.read_csv(ELO_PREGAME_FILE)
    massey_daily = load_daily_ratings()

    print(f"Regular season games: {len(regular):,}")

    team_games = team_game_records(regular)

    print(f"{'-'*10} Building Pre-Game Features {'-'*10}")

    games = regular_game_features(regular, team_games, elo_pregame, massey_daily)

    if len(games) != len(regular):
        raise ValueError(f"Expected {len(regular):,} games, got {len(games):,}.")

    print(f"✓ {len(games):,} regular-season games.")

    print(f"{'-'*10} Data Leakage Checks {'-'*10}")

    # A team's Nth game must see exactly N-1 earlier games
    pregame = pregame_team_features(team_games)
    game_number = team_games.groupby(["Season", "TeamID"]).cumcount()

    if not (pregame["GamesPlayed"] == game_number).all():
        raise ValueError("DATA LEAKAGE DETECTED: pre-game stats include the current game.")

    print("✓ Pre-game stats only use earlier games.")

    first_games = pregame["GamesPlayed"] == 0

    if pregame.loc[first_games, "NetEfficiency"].notna().any():
        raise ValueError("DATA LEAKAGE DETECTED: teams have stats before their first game.")

    print("✓ Teams have no stats before their first game.")

    if games["EloDiff"].isna().any():
        raise ValueError("Some games are missing pre-game Elo.")

    print("✓ Every game has pre-game Elo.")

    print(f"{'-'*10} Building Pre-Tournament Features {'-'*10}")

    pretourney = pretourney_team_features(team_games, elo, massey_daily)

    # Full-season numbers must match feature_engineering.py exactly
    team_stats = pd.read_csv(TEAM_STATS_FILE)
    compare = pretourney.merge(team_stats, on=["Season", "TeamID"], suffixes=("", "_check"))

    for col in ["NetEfficiency", "EFGPct", "FTRate", "RecentNetEfficiency"]:
        if not np.allclose(compare[col], compare[f"{col}_check"]):
            raise ValueError(f"{col} does not match team_season_stats.csv.")

    print("✓ Full-season stats match team_season_stats.csv.")
    print(f"✓ {len(pretourney):,} team-seasons.")

    usable = games.dropna()
    print(f"\nGames with every feature available: {len(usable):,} of {len(games):,}")
    print("(Missing ones are mostly each team's first game or two of a season.)")

    games.to_csv(REGULAR_GAMES_OUTPUT_FILE, index=False)
    pretourney.to_csv(PRETOURNEY_OUTPUT_FILE, index=False)

    print(f"\nSaved regular-season game features to:\n{REGULAR_GAMES_OUTPUT_FILE}")
    print(f"Saved pre-tournament team features to:\n{PRETOURNEY_OUTPUT_FILE}")
