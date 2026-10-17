import os

import numpy as np
import pandas as pd

from build_matchups import PROJECT_ROOT, SEEDS_FILE, TOURNAMENT_RESULTS_FILE
from simulate import (
    FINAL_MODEL,
    RANDOM_SEED,
    SLOTS_FILE,
    most_likely_bracket,
    simulate,
    slot_order,
    win_probability_matrix,
)
from train import DEV_SEASONS, MODELS, load_data

# ============================================================
# Bracket backtest
#
# For each of the last 10 tournaments, fill out the bracket the
# model would have picked using ONLY data from before that
# tournament, then grade it against what actually happened.
#
# Note: 2015-2019 are DEV seasons, which were used to choose the
# final model, so those years are slightly optimistic. 2021-2025
# are the untouched holdout.
# ============================================================

BACKTEST_SEASONS = [2015, 2016, 2017, 2018, 2019, 2021, 2022, 2023, 2024, 2025]

# Seed-only picks the better seed every game. When equal seeds meet
# (e.g. two 1-seeds in the Final Four) it's a coin flip, broken by
# taking the team listed first in the bracket.
COMPARE_MODELS = [FINAL_MODEL, "Seed only (LR)"]

N_SIMULATIONS = 10_000

# Standard ESPN Tournament Challenge scoring: 10, 20, 40, 80, 160, 320
# points per correct pick in rounds 1-6, so a perfect bracket = 1920
ESPN_POINTS = {r: 10 * 2 ** (r - 1) for r in range(1, 7)}


def load_bracket(season):
    seeds = pd.read_csv(SEEDS_FILE)
    seeds = seeds[seeds["Season"] == season]

    slots = pd.read_csv(SLOTS_FILE)
    slots = slots[slots["Season"] == season]
    slots = slots.sort_values("Slot", key=lambda s: s.map(slot_order), kind="stable")

    return dict(zip(seeds["Seed"], seeds["TeamID"])), slots


def actual_bracket(slots, seed_to_team, results):
    """
    Walk the bracket with real results.

    Returns ({slot: TeamID that actually won it}, set of slots with no
    game played). Each slot's real game is found by the two teams that
    occupied it. A missing game (2021 Oregon vs VCU was a COVID
    no-contest) goes to whichever team played later in the tournament.
    """
    games = {
        frozenset((w, l)): w
        for w, l in zip(results["WTeamID"], results["LTeamID"])
    }
    last_day = pd.concat([
        results[["WTeamID", "DayNum"]].set_axis(["TeamID", "DayNum"], axis=1),
        results[["LTeamID", "DayNum"]].set_axis(["TeamID", "DayNum"], axis=1),
    ]).groupby("TeamID")["DayNum"].max()

    occupant = dict(seed_to_team)
    not_played = set()

    for slot in slots.itertuples(index=False):
        strong, weak = occupant[slot.StrongSeed], occupant[slot.WeakSeed]
        pair = frozenset((strong, weak))

        if pair in games:
            occupant[slot.Slot] = games[pair]
        else:
            occupant[slot.Slot] = max((strong, weak), key=lambda t: last_day.get(t, -1))
            not_played.add(slot.Slot)

    return {slot: occupant[slot] for slot in slots["Slot"]}, not_played


def grade(picks, actual, not_played):
    """Score one filled-out bracket against the real results."""
    picks = picks[picks["Round"] >= 1].copy()  # First Four isn't part of a normal bracket
    picks["Correct"] = picks["PickTeamID"] == picks["Slot"].map(actual)

    # A game that was never played can't be scored either way
    picks = picks[~picks["Slot"].isin(not_played)]

    final_four_slots = picks.loc[picks["Round"] == 4, "Slot"]
    predicted_final_four = set(picks.loc[picks["Round"] == 4, "PickTeamID"])
    actual_final_four = {actual[s] for s in final_four_slots}

    return {
        "PickedChampion": picks.loc[picks["Round"] == 6, "PickTeamID"].iloc[0],
        "ActualChampion": actual["R6CH"],
        "ChampionCorrect": bool(picks.loc[picks["Round"] == 6, "Correct"].iloc[0]),
        "FinalFourCorrect": len(predicted_final_four & actual_final_four),
        "GamesCorrect": int(picks["Correct"].sum()),
        "GamesScored": len(picks),
        "ESPNScore": int((picks["Correct"] * picks["Round"].map(ESPN_POINTS)).sum()),
    }


def champion_odds(slots, seed_to_team, P, index, team, rng):
    """Model's pre-tournament probability that `team` wins it all."""
    round_winners = simulate(slots, seed_to_team, P, index, N_SIMULATIONS, rng)
    return np.mean(round_winners[6] == index[team])


if __name__ == "__main__":

    print(f"{'-'*10} Loading Data {'-'*10}")

    data = load_data()
    results = pd.read_csv(TOURNAMENT_RESULTS_FILE)
    teams = pd.read_csv(os.path.join(PROJECT_ROOT, "data/raw/MTeams.csv"))
    names = dict(zip(teams["TeamID"], teams["TeamName"]))

    rng = np.random.default_rng(RANDOM_SEED)
    rows = []

    print(f"{'-'*10} Backtesting {len(BACKTEST_SEASONS)} Tournaments {'-'*10}")

    for season in BACKTEST_SEASONS:
        seed_to_team, slots = load_bracket(season)
        team_ids = sorted(seed_to_team.values())
        actual, not_played = actual_bracket(slots, seed_to_team, results[results["Season"] == season])

        for name in COMPARE_MODELS:
            model = MODELS[name]().fit(data, season)
            P, index = win_probability_matrix(model, data, team_ids, season)

            row = grade(most_likely_bracket(slots, seed_to_team, P, index), actual, not_played)
            row["ActualChampionOdds"] = champion_odds(
                slots, seed_to_team, P, index, row["ActualChampion"], rng
            )
            rows.append({"Season": season, "Model": name, **row})

        print(f"  {season} done")

    results_df = pd.DataFrame(rows)

    for name in COMPARE_MODELS:

        df = results_df[results_df["Model"] == name]

        print(f"{'-'*10} {name} {'-'*10}")

        display = pd.DataFrame({
            "Season": df["Season"],
            "Set": np.where(df["Season"].isin(DEV_SEASONS), "dev", "holdout"),
            "Picked Champion": df["PickedChampion"].map(names),
            "Actual Champion": df["ActualChampion"].map(names),
            "Champ": np.where(df["ChampionCorrect"], "yes", "no"),
            "Champ Odds": (df["ActualChampionOdds"] * 100).map(lambda x: f"{x:.1f}%"),
            "Final Four": df["FinalFourCorrect"].map(lambda x: f"{x}/4"),
            "Games": [f"{c}/{n}" for c, n in zip(df["GamesCorrect"], df["GamesScored"])],
            "ESPN": df["ESPNScore"],
        })

        print(display.to_string(index=False))

        print(f"\nChampions correct:   {df['ChampionCorrect'].sum()} of {len(df)}")
        print(f"Final Four correct:  {df['FinalFourCorrect'].sum()} of {4 * len(df)} "
              f"({df['FinalFourCorrect'].mean():.1f} per year)")
        print(f"Bracket games:       {df['GamesCorrect'].sum()} of {df['GamesScored'].sum()} "
              f"({df['GamesCorrect'].sum() / df['GamesScored'].sum():.1%})")
        print(f"Average ESPN score:  {df['ESPNScore'].mean():.0f} of 1920")
        print("Champ Odds = model's pre-tournament chance for the team that actually won.")
