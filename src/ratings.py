import os
import numpy as np
import pandas as pd

# Paths are resolved from the project root so this runs from any directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

REGULAR_SEASON_RESULTS_FILE = os.path.join(PROJECT_ROOT, "data/raw/MRegularSeasonDetailedResults.csv")
TOURNAMENT_RESULTS_FILE = os.path.join(PROJECT_ROOT, "data/raw/MNCAATourneyDetailedResults.csv")

OUTPUT_FILE = os.path.join(PROJECT_ROOT, "data/processed/elo_ratings.csv")
PREGAME_OUTPUT_FILE = os.path.join(PROJECT_ROOT, "data/processed/elo_pregame.csv")

# ============================================================
# Elo parameters
#
# Tuned by grid search on how well PRE-GAME Elo predicts
# regular-season games (2004+), not on tournament games, so
# the tournament validation in train.py stays clean.
# ============================================================

MEAN_RATING = 1500   # every new team starts here
K = 45               # how far one game moves a rating
HOME_ADVANTAGE = 100 # Elo points added to the home team's rating
CARRYOVER = 0.9      # share of last season's (rating - mean) that is kept

# Let PAST tournaments update ratings carried into the next season.
# A season's own tournament is never included in its Elo feature.
USE_TOURNAMENT_GAMES = True


def win_probability(rating_diff):
    """Elo expected score for a team that is rating_diff points better."""
    return 1 / (1 + 10 ** (-rating_diff / 400))


def margin_multiplier(margin, winner_rating_diff):
    """
    Bigger wins move ratings more, but less so when a strong team
    beats a weak one (FiveThirtyEight-style, stops blowouts of bad
    teams from inflating ratings).
    """
    return (margin + 3) ** 0.8 / (7.5 + 0.006 * winner_rating_diff)


def update_ratings(ratings, game, k, home_advantage):
    """
    Apply one game to the ratings dict.

    Returns the winner's and loser's PRE-game ratings and the winner's
    pre-game win probability.
    """
    winner_rating = ratings.get(game.WTeamID, MEAN_RATING)
    loser_rating = ratings.get(game.LTeamID, MEAN_RATING)

    if game.WLoc == "H":
        home_adjustment = home_advantage
    elif game.WLoc == "A":
        home_adjustment = -home_advantage
    else:
        home_adjustment = 0

    rating_diff = winner_rating - loser_rating + home_adjustment
    expected = win_probability(rating_diff)

    change = (
        k
        * margin_multiplier(game.WScore - game.LScore, rating_diff)
        * (1 - expected)
    )

    ratings[game.WTeamID] = winner_rating + change
    ratings[game.LTeamID] = loser_rating - change

    return winner_rating, loser_rating, expected


def run_elo(
    regular,
    tournament=None,
    k=K,
    home_advantage=HOME_ADVANTAGE,
    carryover=CARRYOVER,
):
    """
    Walk through games season by season in date order, updating ratings.

    For each season:
        1. pull every rating part of the way back toward the mean
        2. play the regular season
        3. SNAPSHOT ratings -> this is the season's Elo feature
        4. play that season's tournament games (if given), which only
           affect next season's starting ratings

    Because the snapshot happens before step 4, a season's Elo never
    includes that season's tournament results.

    Returns:
        end_of_season: DataFrame of Season, TeamID, Elo for every team
            that played a regular-season game that season
        pregame: one row per regular-season game with both teams'
            PRE-game ratings (WElo, LElo) and the winner's pre-game win
            probability (WinProb). Used as point-in-time features and to
            measure how predictive the ratings are.
    """
    regular = regular.sort_values(["Season", "DayNum"]).reset_index(drop=True)

    if tournament is None:
        tournament = regular.iloc[0:0]

    tournament = tournament.sort_values(["Season", "DayNum"])

    ratings = {}
    end_of_season = []
    pregame = []

    for season, season_games in regular.groupby("Season", sort=True):

        ratings = {
            team: MEAN_RATING + carryover * (rating - MEAN_RATING)
            for team, rating in ratings.items()
        }

        for game in season_games.itertuples(index=False):
            pregame.append(
                (game.Season, game.DayNum, game.WTeamID, game.LTeamID)
                + update_ratings(ratings, game, k, home_advantage)
            )

        played = set(season_games["WTeamID"]) | set(season_games["LTeamID"])

        end_of_season.extend(
            (season, team, ratings[team]) for team in played
        )

        for game in tournament[tournament["Season"] == season].itertuples(index=False):
            update_ratings(ratings, game, k, home_advantage)

    end_of_season = pd.DataFrame(end_of_season, columns=["Season", "TeamID", "Elo"])

    pregame = pd.DataFrame(
        pregame,
        columns=["Season", "DayNum", "WTeamID", "LTeamID", "WElo", "LElo", "WinProb"],
    )

    return end_of_season, pregame


def pregame_accuracy_metrics(pregame):
    """
    Log loss and accuracy of pre-game Elo on regular-season games.

    The first season starts every team at 1500, so its ratings are
    rough; it is skipped when judging accuracy.
    """
    probs = pregame.loc[pregame["Season"] > pregame["Season"].min(), "WinProb"].to_numpy()

    log_loss = -np.mean(np.log(np.clip(probs, 1e-6, 1)))
    accuracy = np.mean(probs > 0.5)

    return log_loss, accuracy


if __name__ == "__main__":

    print(f"{'-'*10} Loading Data {'-'*10}")

    regular = pd.read_csv(REGULAR_SEASON_RESULTS_FILE)
    tournament = pd.read_csv(TOURNAMENT_RESULTS_FILE)

    print(f"Regular season games: {len(regular):,}")
    print(f"Tournament games:     {len(tournament):,}")

    print(f"{'-'*10} Running Elo {'-'*10}")
    print(f"K={K}, HOME_ADVANTAGE={HOME_ADVANTAGE}, CARRYOVER={CARRYOVER}")

    for label, tourney_games in [
        ("regular season only", None),
        ("+ past tournaments", tournament),
    ]:
        _, pregame = run_elo(regular, tourney_games)
        log_loss, accuracy = pregame_accuracy_metrics(pregame)
        print(f"  {label:<20} pre-game log loss {log_loss:.4f}, accuracy {accuracy:.4f}")

    elo, pregame = run_elo(regular, tournament if USE_TOURNAMENT_GAMES else None)

    print(f"Using: {'+ past tournaments' if USE_TOURNAMENT_GAMES else 'regular season only'}")

    print(f"{'-'*10} Data Leakage Check {'-'*10}")

    # Removing a season's tournament must not change that season's Elo
    check_season = tournament["Season"].max()

    elo_without, _ = run_elo(
        regular,
        tournament[tournament["Season"] != check_season] if USE_TOURNAMENT_GAMES else None,
    )

    with_check = elo[elo["Season"] == check_season].set_index("TeamID")["Elo"]
    without_check = elo_without[elo_without["Season"] == check_season].set_index("TeamID")["Elo"]

    if not np.allclose(with_check.sort_index(), without_check.sort_index()):
        raise ValueError(
            f"DATA LEAKAGE DETECTED: {check_season} tournament games "
            f"changed {check_season} Elo ratings."
        )

    print(f"✓ {check_season} Elo is unchanged when {check_season} tournament games are removed.")

    if elo.duplicated(subset=["Season", "TeamID"]).any():
        raise ValueError("Duplicate Season/TeamID Elo rows.")

    print(f"✓ {len(elo):,} team-season ratings.")

    latest_season = elo["Season"].max()

    print(f"{'-'*10} Top 10 Elo, {latest_season} {'-'*10}")

    teams = pd.read_csv(os.path.join(PROJECT_ROOT, "data/raw/MTeams.csv"))

    print(
        elo[elo["Season"] == latest_season]
        .merge(teams[["TeamID", "TeamName"]], on="TeamID")
        .sort_values("Elo", ascending=False)
        .head(10)[["TeamName", "Elo"]]
        .round(0)
        .to_string(index=False)
    )

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    elo.to_csv(OUTPUT_FILE, index=False)
    pregame.to_csv(PREGAME_OUTPUT_FILE, index=False)

    print(f"\nSaved pre-tournament Elo ratings to:\n{OUTPUT_FILE}")
    print(f"Saved pre-game Elo for every regular-season game to:\n{PREGAME_OUTPUT_FILE}")
