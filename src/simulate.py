import os
from itertools import combinations

import numpy as np
import pandas as pd

from build_matchups import PROJECT_ROOT, SEEDS_FILE
from train import MODELS, load_data

SLOTS_FILE = os.path.join(PROJECT_ROOT, "data/raw/MNCAATourneySlots.csv")
SEASONS_FILE = os.path.join(PROJECT_ROOT, "data/raw/MSeasons.csv")

ODDS_OUTPUT_FILE = os.path.join(PROJECT_ROOT, "data/processed/bracket_odds_{season}.csv")
PICKS_OUTPUT_FILE = os.path.join(PROJECT_ROOT, "data/processed/bracket_picks_{season}.csv")

SEASON = 2026
FINAL_MODEL = "Margin GBM"  # best log loss / Brier on DEV seasons in train.py
N_SIMULATIONS = 10_000
RANDOM_SEED = 0

# Winning a slot in round N means reaching the next stage
ROUND_NAMES = {
    1: "Round of 32",
    2: "Sweet 16",
    3: "Elite 8",
    4: "Final Four",
    5: "Title Game",
    6: "Champion",
}


def slot_order(slot):
    """First Four slots (e.g. "W16") first, then R1, R2, ... R6."""
    return 0 if not slot.startswith("R") else int(slot[1])


def win_probability_matrix(model, data, team_ids, season=SEASON):
    """
    P[i, j] = probability team_ids[i] beats team_ids[j].

    Every pair is predicted with the lower TeamID as Team A, the same
    orientation used in training, then mirrored so P[j, i] = 1 - P[i, j].
    """
    pairs = pd.DataFrame(
        list(combinations(sorted(team_ids), 2)),
        columns=["TeamA", "TeamB"],
    )
    pairs["Season"] = season

    probs = model.predict(data, pairs)

    index = {team: i for i, team in enumerate(team_ids)}
    P = np.full((len(team_ids), len(team_ids)), 0.5)

    for team_a, team_b, p in zip(pairs["TeamA"], pairs["TeamB"], probs):
        P[index[team_a], index[team_b]] = p
        P[index[team_b], index[team_a]] = 1 - p

    return P, index


def simulate(slots, seed_to_team, P, index, n_simulations, rng):
    """
    Play the bracket n_simulations times at once.

    `winners` maps a bracket label (a seed like "W01", or a slot like
    "R1W1") to an array holding which team occupies it in each simulation.
    A slot's winner is stored under the slot's own name, which is how
    later rounds find their teams.

    Returns {round: array of team indices that won a slot in that round}.
    """
    winners = {
        seed: np.full(n_simulations, index[team])
        for seed, team in seed_to_team.items()
    }
    round_winners = {r: [] for r in [0] + list(ROUND_NAMES)}

    for slot in slots.itertuples(index=False):
        strong = winners[slot.StrongSeed]
        weak = winners[slot.WeakSeed]

        strong_wins = rng.random(n_simulations) < P[strong, weak]
        winners[slot.Slot] = np.where(strong_wins, strong, weak)

        round_winners[slot_order(slot.Slot)].append(winners[slot.Slot])

    return {r: np.concatenate(w) for r, w in round_winners.items() if w}


def most_likely_bracket(slots, seed_to_team, P, index):
    """Fill out one bracket by always picking the team more likely to win each game."""
    occupant = dict(seed_to_team)
    picks = []

    for slot in slots.itertuples(index=False):
        strong = occupant[slot.StrongSeed]
        weak = occupant[slot.WeakSeed]

        p_strong = P[index[strong], index[weak]]
        pick = strong if p_strong >= 0.5 else weak

        occupant[slot.Slot] = pick
        picks.append({
            "Slot": slot.Slot,
            "Round": slot_order(slot.Slot),
            "StrongTeamID": strong,
            "WeakTeamID": weak,
            "PickTeamID": pick,
            "PickWinProb": max(p_strong, 1 - p_strong),
        })

    return pd.DataFrame(picks)


if __name__ == "__main__":

    print(f"{'-'*10} Loading {SEASON} Bracket {'-'*10}")

    seeds = pd.read_csv(SEEDS_FILE)
    seeds = seeds[seeds["Season"] == SEASON]

    slots = pd.read_csv(SLOTS_FILE)
    slots = slots[slots["Season"] == SEASON]
    slots = slots.sort_values("Slot", key=lambda s: s.map(slot_order), kind="stable")

    regions = pd.read_csv(SEASONS_FILE).set_index("Season").loc[SEASON]

    data = load_data()
    team_features = data.team_features[data.team_features["Season"] == SEASON]

    seed_to_team = dict(zip(seeds["Seed"], seeds["TeamID"]))
    team_ids = sorted(seeds["TeamID"])

    print(f"Teams: {len(team_ids)}")
    print(f"Games: {len(slots)}")

    missing = set(team_ids) - set(team_features["TeamID"])

    if missing:
        raise ValueError(f"No {SEASON} features for TeamIDs: {sorted(missing)}")

    print(f"✓ Every {SEASON} tournament team has features.")

    print(f"{'-'*10} Training Final Model {'-'*10}")

    model = MODELS[FINAL_MODEL]().fit(data, SEASON)

    print(f"Model: {FINAL_MODEL}")
    print(f"Trained with everything known before the {SEASON} tournament.")

    P, index = win_probability_matrix(model, data, team_ids)

    print(f"{'-'*10} Simulating {N_SIMULATIONS:,} Tournaments {'-'*10}")

    rng = np.random.default_rng(RANDOM_SEED)
    round_winners = simulate(slots, seed_to_team, P, index, N_SIMULATIONS, rng)

    odds = pd.DataFrame({"TeamID": team_ids})

    # Round of 64: automatic unless the team is in a First Four game
    first_four_teams = {
        seed_to_team[s]
        for s in slots.loc[slots["Slot"].map(slot_order) == 0, ["StrongSeed", "WeakSeed"]].values.ravel()
    }
    first_four_wins = np.bincount(round_winners[0], minlength=len(team_ids)) / N_SIMULATIONS
    odds["Round of 64"] = [
        first_four_wins[index[t]] if t in first_four_teams else 1.0
        for t in team_ids
    ]

    for round_num, name in ROUND_NAMES.items():
        wins = np.bincount(round_winners[round_num], minlength=len(team_ids)) / N_SIMULATIONS
        odds[name] = [wins[index[t]] for t in team_ids]

    # Sanity check: e.g. exactly 4 Final Four teams per simulation
    for round_num, name in ROUND_NAMES.items():
        expected = len(slots[slots["Slot"].map(slot_order) == round_num])
        if not np.isclose(odds[name].sum(), expected):
            raise ValueError(f"{name} probabilities sum to {odds[name].sum()}, expected {expected}.")

    print("✓ Round probabilities sum to the number of teams in each round.")

    team_info = (
        seeds
        .assign(
            Region=lambda d: d["Seed"].str[0].map(lambda r: regions[f"Region{r}"]),
            SeedNum=lambda d: d["Seed"].str[1:3].astype(int),
        )
        .merge(team_features[["TeamID", "TeamName"]], on="TeamID")
        [["TeamID", "TeamName", "Region", "SeedNum"]]
    )

    odds = team_info.merge(odds, on="TeamID").sort_values("Champion", ascending=False)

    print(f"{'-'*10} {SEASON} Championship Odds (top 16) {'-'*10}")

    display = odds.head(16).copy()
    for col in ["Round of 64"] + list(ROUND_NAMES.values()):
        display[col] = (display[col] * 100).map(lambda x: f"{x:5.1f}%")

    print(display.drop(columns="TeamID").to_string(index=False))

    print(f"{'-'*10} Most Likely Bracket {'-'*10}")

    picks = most_likely_bracket(slots, seed_to_team, P, index)

    names = dict(zip(team_info["TeamID"], team_info["TeamName"]))
    seed_nums = dict(zip(team_info["TeamID"], team_info["SeedNum"]))

    for side in ["Strong", "Weak", "Pick"]:
        picks[f"{side}Team"] = picks[f"{side}TeamID"].map(names)
        picks[f"{side}SeedNum"] = picks[f"{side}TeamID"].map(seed_nums)

    # Winners of round 4 slots make the Final Four, round 5 the title game, ...
    for round_num in [4, 5, 6]:
        teams = picks.loc[picks["Round"] == round_num, "PickTeamID"]
        print(f"{ROUND_NAMES[round_num] + ':':<12} "
              + ", ".join(f"({seed_nums[t]}) {names[t]}" for t in teams))

    # An upset = picking the team with the worse (higher) seed number
    picks["Upset"] = picks["PickSeedNum"] > picks[["StrongSeedNum", "WeakSeedNum"]].min(axis=1)
    upsets = picks[(picks["Round"] >= 1) & picks["Upset"]]

    print(f"\nUpsets picked: {len(upsets)}")
    for row in upsets.itertuples(index=False):
        loser = row.WeakTeamID if row.PickTeamID == row.StrongTeamID else row.StrongTeamID
        print(f"  {row.Slot:<6} ({row.PickSeedNum}) {row.PickTeam} over "
              f"({seed_nums[loser]}) {names[loser]}  [{row.PickWinProb:.0%}]")

    os.makedirs(os.path.dirname(ODDS_OUTPUT_FILE), exist_ok=True)

    odds.to_csv(ODDS_OUTPUT_FILE.format(season=SEASON), index=False)
    picks.to_csv(PICKS_OUTPUT_FILE.format(season=SEASON), index=False)

    print(f"\nSaved odds to:\n{ODDS_OUTPUT_FILE.format(season=SEASON)}")
    print(f"Saved bracket picks to:\n{PICKS_OUTPUT_FILE.format(season=SEASON)}")
