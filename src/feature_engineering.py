import os
import pandas as pd
import numpy as np

REGULAR_SEASON_RESULTS_FILE = "data/raw/MRegularSeasonDetailedResults.csv"
TOURNAMENT_RESULTS_FILE = "data/raw/MNCAATourneyDetailedResults.csv"
TEAMS_FILE = "data/raw/MTeams.csv"

OUTPUT_FILE = "data/processed/team_season_stats.csv"


def safe_divide(numerator, denominator):
    """Safely divide two pandas Series, returning 0 if the denominator is 0."""
    return np.where(
        denominator != 0,
        numerator / denominator,
        0
    )

def check_required_columns(df, required_columns, name):
    """Check if the DataFrame contains all required columns."""
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"{name} is missing required columns: {missing_columns}")



print(f"{'-'*10} Loading Data {'-'*10}")

regular = pd.read_csv(REGULAR_SEASON_RESULTS_FILE)
tournament = pd.read_csv(TOURNAMENT_RESULTS_FILE)
teams = pd.read_csv(TEAMS_FILE)

print(f"Regular season rows: {len(regular):,}")
print(f"Tournament rows:     {len(tournament):,}")
print(f"Teams:               {len(teams):,}")


# validate input columns
required_regular_columns = [
    "Season",
    "DayNum",
    "WTeamID",
    "WScore",
    "LTeamID",
    "LScore",
    "WLoc",
    "NumOT",

    "WFGM",
    "WFGA",
    "WFGM3",
    "WFGA3",
    "WFTM",
    "WFTA",
    "WOR",
    "WDR",
    "WAst",
    "WTO",
    "WStl",
    "WBlk",
    "WPF",

    "LFGM",
    "LFGA",
    "LFGM3",
    "LFGA3",
    "LFTM",
    "LFTA",
    "LOR",
    "LDR",
    "LAst",
    "LTO",
    "LStl",
    "LBlk",
    "LPF",
]

required_tournament_columns = [
    "Season",
    "DayNum",
    "WTeamID",
    "LTeamID",
]

required_team_columns = [
    "TeamID",
    "TeamName",
]

check_required_columns(
    regular,
    required_regular_columns,
    "MRegularSeasonDetailedResults.csv"
)

check_required_columns(
    tournament,
    required_tournament_columns,
    "MNCAATourneyDetailedResults.csv"
)

check_required_columns(
    teams,
    required_team_columns,
    "MTeams.csv"
)



print(f"{'-'*10} Input Validation {'-'*10}")

if regular.isnull().any().any():
    print("WARNING: Regular season data contains missing values.")
else:
    print("Regular season missing values: 0")

if tournament.isnull().any().any():
    print("WARNING: Tournament data contains missing values.")
else:
    print("Tournament missing values: 0")

if regular.duplicated().any():
    print("WARNING: Duplicate regular season rows detected.")
else:
    print("Duplicate regular season rows: 0")



print(f"{'-'*10} Data Leakage Check {'-'*10}")

regular_game_keys = set(
    zip(
        regular["Season"],
        regular["DayNum"],
        regular["WTeamID"],
        regular["LTeamID"]
    )
)

tournament_game_keys = set(
    zip(
        tournament["Season"],
        tournament["DayNum"],
        tournament["WTeamID"],
        tournament["LTeamID"]
    )
)

overlap = regular_game_keys.intersection(tournament_game_keys)

if overlap:
    raise ValueError(
        f"DATA LEAKAGE DETECTED: "
        f"{len(overlap)} games appear in both datasets."
    )

print("✓ No games overlap between regular season and tournament data.")

print("\nChecking regular-season/tournament date boundaries...")

common_seasons = sorted(
    set(regular["Season"]).intersection(
        set(tournament["Season"])
    )
)

boundary_problems = []

for season in common_seasons:

    regular_max_day = regular.loc[
        regular["Season"] == season,
        "DayNum"
    ].max()

    tournament_min_day = tournament.loc[
        tournament["Season"] == season,
        "DayNum"
    ].min()

    if regular_max_day >= tournament_min_day:
        boundary_problems.append(
            (
                season,
                regular_max_day,
                tournament_min_day
            )
        )

if boundary_problems:

    print("WARNING: Possible date leakage detected:")

    for season, regular_day, tournament_day in boundary_problems:
        print(
            f"  Season {season}: "
            f"regular max DayNum={regular_day}, "
            f"tournament min DayNum={tournament_day}"
        )

else:
    print(
        "✓ Regular-season games occur before "
        "tournament games for all common seasons."
    )


feature_source = regular.copy()

if len(feature_source) != len(regular):
    raise ValueError(
        "Feature source does not match regular-season dataset."
    )

print(
    "✓ Feature calculations will use ONLY "
    "MRegularSeasonDetailedResults.csv."
)



print(f"{'-'*10} Creating Team-Game Records {'-'*10}")

winner_records = pd.DataFrame({
    "Season": regular["Season"],
    "DayNum": regular["DayNum"],
    "TeamID": regular["WTeamID"],
    "OpponentID": regular["LTeamID"],

    "Win": 1,

    "PointsFor": regular["WScore"],
    "PointsAgainst": regular["LScore"],

    "FGM": regular["WFGM"],
    "FGA": regular["WFGA"],

    "FGM3": regular["WFGM3"],
    "FGA3": regular["WFGA3"],

    "FTM": regular["WFTM"],
    "FTA": regular["WFTA"],

    "OR": regular["WOR"],
    "DR": regular["WDR"],

    "AST": regular["WAst"],
    "TO": regular["WTO"],
    "STL": regular["WStl"],
    "BLK": regular["WBlk"],
    "PF": regular["WPF"],

    "Opponent_FGM": regular["LFGM"],
    "Opponent_FGA": regular["LFGA"],

    "Opponent_FGM3": regular["LFGM3"],
    "Opponent_FGA3": regular["LFGA3"],

    "Opponent_FTM": regular["LFTM"],
    "Opponent_FTA": regular["LFTA"],

    "Opponent_OR": regular["LOR"],
    "Opponent_DR": regular["LDR"],

    "Opponent_AST": regular["LAst"],
    "Opponent_TO": regular["LTO"],
    "Opponent_STL": regular["LStl"],
    "Opponent_BLK": regular["LBlk"],
    "Opponent_PF": regular["LPF"],

    "Location": regular["WLoc"],
    "NumOT": regular["NumOT"],
})

loser_records = pd.DataFrame({
    "Season": regular["Season"],
    "DayNum": regular["DayNum"],
    "TeamID": regular["LTeamID"],
    "OpponentID": regular["WTeamID"],

    "Win": 0,

    "PointsFor": regular["LScore"],
    "PointsAgainst": regular["WScore"],

    "FGM": regular["LFGM"],
    "FGA": regular["LFGA"],

    "FGM3": regular["LFGM3"],
    "FGA3": regular["LFGA3"],

    "FTM": regular["LFTM"],
    "FTA": regular["LFTA"],

    "OR": regular["LOR"],
    "DR": regular["LDR"],

    "AST": regular["LAst"],
    "TO": regular["LTO"],
    "STL": regular["LStl"],
    "BLK": regular["LBlk"],
    "PF": regular["LPF"],

    "Opponent_FGM": regular["WFGM"],
    "Opponent_FGA": regular["WFGA"],

    "Opponent_FGM3": regular["WFGM3"],
    "Opponent_FGA3": regular["WFGA3"],

    "Opponent_FTM": regular["WFTM"],
    "Opponent_FTA": regular["WFTA"],

    "Opponent_OR": regular["WOR"],
    "Opponent_DR": regular["WDR"],

    "Opponent_AST": regular["WAst"],
    "Opponent_TO": regular["WTO"],
    "Opponent_STL": regular["WStl"],
    "Opponent_BLK": regular["WBlk"],
    "Opponent_PF": regular["WPF"],

    "Location": regular["WLoc"],
    "NumOT": regular["NumOT"],
})

# combine winner and loser records
team_games = pd.concat(
    [
        winner_records,
        loser_records
    ],
    ignore_index=True
)

team_games = team_games.sort_values(
    ["Season", "DayNum"]
).reset_index(drop=True)

print(
    f"Created {len(team_games):,} team-game records."
)

expected_records = len(regular) * 2

if len(team_games) != expected_records:
    raise ValueError(
        f"Expected {expected_records:,} team-game records, "
        f"but got {len(team_games):,}."
    )

print(
    f"✓ Every game produced exactly two team records."
)



# ============================================================
# Calculate possessions
# ============================================================
#
# NCAA estimated possession formula:
#
# Possessions =
#     FGA
#   + 0.475 * FTA
#   - Offensive Rebounds
#   + Turnovers
#
# This is an estimate rather than an exact possession count.
# ============================================================

team_games["Possessions"] = (
    team_games["FGA"]
    + 0.475 * team_games["FTA"]
    - team_games["OR"]
    + team_games["TO"]
)

team_games["Opponent_Possessions"] = (
    team_games["Opponent_FGA"]
    + 0.475 * team_games["Opponent_FTA"]
    - team_games["Opponent_OR"]
    + team_games["Opponent_TO"]
)


if (team_games["Possessions"] <= 0).any():

    bad_games = team_games[
        team_games["Possessions"] <= 0
    ]

    print(
        "WARNING: Some games have non-positive "
        "estimated possessions."
    )

    print(
        bad_games[
            [
                "Season",
                "DayNum",
                "TeamID",
                "Possessions"
            ]
        ].head()
    )

else:
    print("✓ All estimated possessions are positive.")



print(f"{'-'*10} Aggregating Team-Season Statistics {'-'*10}")

group_columns = ["Season", "TeamID"]

team_stats = (
     team_games
    .groupby(group_columns)
    .agg(
        Games=("Win", "count"),
        Wins=("Win", "sum"),

        PointsFor=("PointsFor", "sum"),
        PointsAgainst=("PointsAgainst", "sum"),

        FGM=("FGM", "sum"),
        FGA=("FGA", "sum"),

        FGM3=("FGM3", "sum"),
        FGA3=("FGA3", "sum"),

        FTM=("FTM", "sum"),
        FTA=("FTA", "sum"),

        OR=("OR", "sum"),
        DR=("DR", "sum"),

        AST=("AST", "sum"),
        TO=("TO", "sum"),
        STL=("STL", "sum"),
        BLK=("BLK", "sum"),
        PF=("PF", "sum"),

        Opponent_FGM=("Opponent_FGM", "sum"),
        Opponent_FGA=("Opponent_FGA", "sum"),

        Opponent_FGM3=("Opponent_FGM3", "sum"),
        Opponent_FGA3=("Opponent_FGA3", "sum"),

        Opponent_FTM=("Opponent_FTM", "sum"),
        Opponent_FTA=("Opponent_FTA", "sum"),

        Opponent_OR=("Opponent_OR", "sum"),
        Opponent_DR=("Opponent_DR", "sum"),

        Opponent_AST=("Opponent_AST", "sum"),
        Opponent_TO=("Opponent_TO", "sum"),
        Opponent_STL=("Opponent_STL", "sum"),
        Opponent_BLK=("Opponent_BLK", "sum"),
        Opponent_PF=("Opponent_PF", "sum"),

        Possessions=("Possessions", "sum"),
        Opponent_Possessions=(
            "Opponent_Possessions",
            "sum"
        ),
    )
    .reset_index()
)


team_stats["Losses"] = (
    team_stats["Games"]
    - team_stats["Wins"]
)

team_stats["WinPct"] = safe_divide(
    team_stats["Wins"],
    team_stats["Games"]
)

team_stats["PPG"] = safe_divide(
    team_stats["PointsFor"],
    team_stats["Games"]
)

team_stats["OppPPG"] = safe_divide(
    team_stats["PointsAgainst"],
    team_stats["Games"]
)

team_stats["PointDiff"] = (
    team_stats["PPG"]
    - team_stats["OppPPG"]
)

team_stats["FGPct"] = safe_divide(
    team_stats["FGM"],
    team_stats["FGA"]
)

team_stats["ThreePtPct"] = safe_divide(
    team_stats["FGM3"],
    team_stats["FGA3"]
)

team_stats["FTPct"] = safe_divide(
    team_stats["FTM"],
    team_stats["FTA"]
)

team_stats["OppFGPct"] = safe_divide(
    team_stats["Opponent_FGM"],
    team_stats["Opponent_FGA"]
)

team_stats["OppThreePtPct"] = safe_divide(
    team_stats["Opponent_FGM3"],
    team_stats["Opponent_FGA3"]
)

team_stats["OppFTPct"] = safe_divide(
    team_stats["Opponent_FTM"],
    team_stats["Opponent_FTA"]
)

team_stats["ORPG"] = safe_divide(
    team_stats["OR"],
    team_stats["Games"]
)

team_stats["DRPG"] = safe_divide(
    team_stats["DR"],
    team_stats["Games"]
)

team_stats["APG"] = safe_divide(
    team_stats["AST"],
    team_stats["Games"]
)

team_stats["TOPG"] = safe_divide(
    team_stats["TO"],
    team_stats["Games"]
)

team_stats["SPG"] = safe_divide(
    team_stats["STL"],
    team_stats["Games"]
)

team_stats["BPG"] = safe_divide(
    team_stats["BLK"],
    team_stats["Games"]
)

team_stats["PFPG"] = safe_divide(
    team_stats["PF"],
    team_stats["Games"]
)

team_stats["OppORPG"] = safe_divide(
    team_stats["Opponent_OR"],
    team_stats["Games"]
)

team_stats["OppDRPG"] = safe_divide(
    team_stats["Opponent_DR"],
    team_stats["Games"]
)

team_stats["OppAPG"] = safe_divide(
    team_stats["Opponent_AST"],
    team_stats["Games"]
)

team_stats["OppTOPG"] = safe_divide(
    team_stats["Opponent_TO"],
    team_stats["Games"]
)


team_stats["OffensiveReboundRate"] = safe_divide(
    team_stats["OR"],
    team_stats["OR"] + team_stats["Opponent_DR"]
)

team_stats["DefensiveReboundRate"] = safe_divide(
    team_stats["DR"],
    team_stats["DR"] + team_stats["Opponent_OR"]
)

team_stats["OffensiveEfficiency"] = (
    safe_divide(
        team_stats["PointsFor"],
        team_stats["Possessions"]
    ) * 100
)

team_stats["DefensiveEfficiency"] = (
    safe_divide(
        team_stats["PointsAgainst"],
        team_stats["Opponent_Possessions"]
    ) * 100
)

team_stats["NetEfficiency"] = (
    team_stats["OffensiveEfficiency"]
    - team_stats["DefensiveEfficiency"]
)

team_stats["PossessionsPerGame"] = safe_divide(
    team_stats["Possessions"],
    team_stats["Games"]
)

team_stats["OpponentPossessionsPerGame"] = safe_divide(
    team_stats["Opponent_Possessions"],
    team_stats["Games"]
)


# ============================================================
# Four Factors shooting stats
#
# eFG% = (FGM + 0.5 * FGM3) / FGA
#     Field goal % that counts a three as 1.5 makes.
#
# FT Rate = FTA / FGA
#     How often a team gets to the free-throw line.
# ============================================================

team_stats["EFGPct"] = safe_divide(
    team_stats["FGM"] + 0.5 * team_stats["FGM3"],
    team_stats["FGA"]
)

team_stats["OppEFGPct"] = safe_divide(
    team_stats["Opponent_FGM"] + 0.5 * team_stats["Opponent_FGM3"],
    team_stats["Opponent_FGA"]
)

team_stats["FTRate"] = safe_divide(
    team_stats["FTA"],
    team_stats["FGA"]
)

team_stats["OppFTRate"] = safe_divide(
    team_stats["Opponent_FTA"],
    team_stats["Opponent_FGA"]
)


# ============================================================
# Recent form
#
# Win % and net efficiency over each team's last
# RECENT_GAMES regular-season games going into the tournament.
# ============================================================

RECENT_GAMES = 10

print(f"{'-'*10} Calculating Recent Form (last {RECENT_GAMES} games) {'-'*10}")

recent_games = (
    team_games
    .sort_values(["Season", "TeamID", "DayNum"])
    .groupby(["Season", "TeamID"])
    .tail(RECENT_GAMES)
)

recent_form = (
    recent_games
    .groupby(["Season", "TeamID"])
    .agg(
        RecentWins=("Win", "sum"),
        RecentGames=("Win", "count"),
        RecentPointsFor=("PointsFor", "sum"),
        RecentPointsAgainst=("PointsAgainst", "sum"),
        RecentPossessions=("Possessions", "sum"),
        RecentOpponentPossessions=("Opponent_Possessions", "sum"),
    )
    .reset_index()
)

recent_form["RecentWinPct"] = safe_divide(
    recent_form["RecentWins"],
    recent_form["RecentGames"]
)

recent_form["RecentNetEfficiency"] = (
    safe_divide(
        recent_form["RecentPointsFor"],
        recent_form["RecentPossessions"]
    ) * 100
    - safe_divide(
        recent_form["RecentPointsAgainst"],
        recent_form["RecentOpponentPossessions"]
    ) * 100
)

team_stats = team_stats.merge(
    recent_form[["Season", "TeamID", "RecentWinPct", "RecentNetEfficiency"]],
    on=["Season", "TeamID"],
    how="left"
)

print(
    f"✓ Recent form calculated for {len(recent_form):,} team-seasons."
)

team_stats = team_stats.merge(
    teams[
        [
            "TeamID",
            "TeamName"
        ]
    ],
    on="TeamID",
    how="left"
)



# check for missing team names
missing_names = team_stats[
    team_stats["TeamName"].isna()
]

if len(missing_names) > 0:

    print(
        "\nWARNING: Could not find team names for "
        f"{len(missing_names)} team-season rows."
    )

    print(
        missing_names[
            [
                "Season",
                "TeamID"
            ]
        ].drop_duplicates().head(20)
    )

else:
    print(
        "✓ Every team has a corresponding team name."
    )



# ============================================================
# DATA LEAKAGE CHECK #4
#
# Every team-season row should only contain regular-season
# statistics.
#
# Verify that each team-season's final DayNum is before the
# tournament's first DayNum for that season.
# ============================================================

print("\nChecking team-season feature boundaries...")

last_regular_days = (
    team_games
    .groupby("Season")["DayNum"]
    .max()
)

first_tournament_days = (
    tournament
    .groupby("Season")["DayNum"]
    .min()
)

boundary_issues = []

for season in last_regular_days.index:

    if season not in first_tournament_days.index:
        continue

    regular_day = last_regular_days.loc[season]
    tournament_day = first_tournament_days.loc[season]

    if regular_day >= tournament_day:

        boundary_issues.append({
            "Season": season,
            "LastRegularDay": regular_day,
            "FirstTournamentDay": tournament_day
        })

if boundary_issues:

    print(
        "WARNING: Some seasons have potentially overlapping "
        "regular-season and tournament DayNums:"
    )

    print(
        pd.DataFrame(boundary_issues)
    )

else:

    print(
        "✓ All team-season features end before "
        "the tournament begins."
    )


# ============================================================
# DATA LEAKAGE CHECK #5
#
# Verify that tournament games are NOT represented in our
# team-game feature data.
# ============================================================

feature_game_keys = set(
    zip(
        team_games["Season"],
        team_games["DayNum"],
        team_games["TeamID"],
        team_games["OpponentID"]
    )
)

tournament_team_game_keys = set()

for _, game in tournament.iterrows():

    tournament_team_game_keys.add(
        (
            game["Season"],
            game["DayNum"],
            game["WTeamID"],
            game["LTeamID"]
        )
    )

    tournament_team_game_keys.add(
        (
            game["Season"],
            game["DayNum"],
            game["LTeamID"],
            game["WTeamID"]
        )
    )

feature_tournament_overlap = (
    feature_game_keys
    .intersection(tournament_team_game_keys)
)

if feature_tournament_overlap:

    raise ValueError(
        "DATA LEAKAGE DETECTED: Tournament games "
        "were found in team feature data."
    )

else:

    print(
        "✓ No tournament games are present in "
        "the feature dataset."
    )



column_order = [
    "Season",
    "TeamID",
    "TeamName",

    "Games",
    "Wins",
    "Losses",
    "WinPct",

    "PointsFor",
    "PointsAgainst",
    "PPG",
    "OppPPG",
    "PointDiff",

    "FGPct",
    "ThreePtPct",
    "FTPct",

    "OppFGPct",
    "OppThreePtPct",
    "OppFTPct",

    "ORPG",
    "DRPG",
    "APG",
    "TOPG",
    "SPG",
    "BPG",
    "PFPG",

    "OppORPG",
    "OppDRPG",
    "OppAPG",
    "OppTOPG",

    "OffensiveReboundRate",
    "DefensiveReboundRate",

    "PossessionsPerGame",
    "OpponentPossessionsPerGame",

    "OffensiveEfficiency",
    "DefensiveEfficiency",
    "NetEfficiency",

    "EFGPct",
    "OppEFGPct",
    "FTRate",
    "OppFTRate",

    "RecentWinPct",
    "RecentNetEfficiency",
]

team_stats = team_stats[column_order]



print(f"{'-'*10} Final Validation {'-'*10}")

print(
    f"Team-season rows: {len(team_stats):,}"
)

print(
    f"Unique seasons:   {team_stats['Season'].nunique()}"
)

print(
    f"Unique teams:     {team_stats['TeamID'].nunique()}"
)

print(
    f"Missing values:   "
    f"{team_stats.isna().sum().sum():,}"
)

# Check duplicate team-season combinations
duplicates = team_stats.duplicated(
    subset=["Season", "TeamID"]
).sum()

if duplicates > 0:

    raise ValueError(
        f"Found {duplicates} duplicate "
        "Season/TeamID combinations."
    )

else:

    print(
        "✓ No duplicate Season/TeamID combinations."
    )


# Check impossible win/loss records
record_check = (
    team_stats["Wins"]
    + team_stats["Losses"]
    == team_stats["Games"]
)

if not record_check.all():

    raise ValueError(
        "Some teams have inconsistent "
        "win/loss/game totals."
    )

else:

    print(
        "✓ Win/loss/game totals are consistent."
    )


# Check efficiency values
efficiency_columns = [
    "OffensiveEfficiency",
    "DefensiveEfficiency",
    "NetEfficiency"
]

if team_stats[efficiency_columns].isna().any().any():

    raise ValueError(
        "Efficiency statistics contain NaN values."
    )

else:

    print(
        "✓ Efficiency statistics contain no NaN values."
    )



print(f"{'-'*10} Sample Team-Season Statistics {'-'*10}")

print(
    team_stats[
        [
            "Season",
            "TeamName",
            "Games",
            "Wins",
            "WinPct",
            "PPG",
            "OppPPG",
            "PointDiff",
            "FGPct",
            "ThreePtPct",
            "FTPct",
            "OffensiveEfficiency",
            "DefensiveEfficiency",
            "NetEfficiency",
            "OffensiveReboundRate",
            "DefensiveReboundRate"
        ]
    ]
    .head(10)
    .to_string(index=False)
)


latest_season = team_stats["Season"].max()

print(f"{'-'*10} Top Teams by Net Efficiency {'-'*10}")

latest = (
    team_stats[
        team_stats["Season"] == latest_season
    ]
    .sort_values(
        "NetEfficiency",
        ascending=False
    )
    .head(10)
)

print(
    latest[
        [
            "TeamName",
            "Games",
            "Wins",
            "Losses",
            "WinPct",
            "PPG",
            "OppPPG",
            "OffensiveEfficiency",
            "DefensiveEfficiency",
            "NetEfficiency",
        ]
    ].to_string(index=False)
)


# ============================================================
# Save processed data
# ============================================================

os.makedirs(
    os.path.dirname(OUTPUT_FILE),
    exist_ok=True
)

team_stats.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n" + "=" * 70)
print("Finished")
print("=" * 70)

print(
    f"Saved team-season statistics to:\n"
    f"{OUTPUT_FILE}"
)

print(
    f"\nFinal shape: {team_stats.shape}"
)