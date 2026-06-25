"""
squad_optimizer.py
--------------------
Given per-player expected points (from the forecaster) and the official
FPL squad rules, solves the constrained squad-selection / transfer problem
via Integer Linear Programming (PuLP):

  - 15-man squad: 2 GK, 5 DEF, 5 MID, 3 FWD
  - Total cost <= 100.0 (budget)
  - Max 3 players from any one real-world club
  - Starting XI formation must be valid (>=1 GK, >=3 DEF, >=2 MID, >=1 FWD,
    11 total starters), captain = highest-EV starter (counts double)
"""

import argparse

import pandas as pd
import pulp


POSITION_LIMITS = {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}
STARTING_MIN = {"GK": 1, "DEF": 3, "MID": 2, "FWD": 1}
BUDGET = 100.0
MAX_PER_CLUB = 3
SQUAD_SIZE = 15
STARTERS = 11


def optimize_squad(players: pd.DataFrame, budget: float = BUDGET) -> pd.DataFrame:
    """
    players: DataFrame with columns [player_id, web_name, position, club, price, ev]
    Returns the selected 15-man squad with a 'starting' boolean column.
    """
    prob = pulp.LpProblem("FantasyXI_Squad", pulp.LpMaximize)

    squad_vars = {pid: pulp.LpVariable(f"squad_{pid}", cat="Binary") for pid in players["player_id"]}
    start_vars = {pid: pulp.LpVariable(f"start_{pid}", cat="Binary") for pid in players["player_id"]}
    cap_vars = {pid: pulp.LpVariable(f"cap_{pid}", cat="Binary") for pid in players["player_id"]}

    ev = players.set_index("player_id")["ev"]
    prob += pulp.lpSum(
        start_vars[pid] * ev[pid] + cap_vars[pid] * ev[pid]  # captain doubles their EV
        for pid in players["player_id"]
    )

    # Squad composition
    for pos, limit in POSITION_LIMITS.items():
        ids = players.loc[players.position == pos, "player_id"]
        prob += pulp.lpSum(squad_vars[pid] for pid in ids) == limit

    prob += pulp.lpSum(squad_vars.values()) == SQUAD_SIZE
    prob += pulp.lpSum(squad_vars[pid] * players.set_index("player_id").loc[pid, "price"]
                        for pid in players["player_id"]) <= budget

    for club, grp in players.groupby("club"):
        prob += pulp.lpSum(squad_vars[pid] for pid in grp["player_id"]) <= MAX_PER_CLUB

    # Starting XI must be a subset of the squad
    for pid in players["player_id"]:
        prob += start_vars[pid] <= squad_vars[pid]
        prob += cap_vars[pid] <= start_vars[pid]

    prob += pulp.lpSum(start_vars.values()) == STARTERS
    prob += pulp.lpSum(cap_vars.values()) == 1

    for pos, min_start in STARTING_MIN.items():
        ids = players.loc[players.position == pos, "player_id"]
        prob += pulp.lpSum(start_vars[pid] for pid in ids) >= min_start

    prob.solve(pulp.PULP_CBC_CMD(msg=False))

    players = players.copy()
    players["in_squad"] = players["player_id"].map(lambda pid: pulp.value(squad_vars[pid]) == 1)
    players["starting"] = players["player_id"].map(lambda pid: pulp.value(start_vars[pid]) == 1)
    players["captain"] = players["player_id"].map(lambda pid: pulp.value(cap_vars[pid]) == 1)

    return players[players.in_squad].sort_values(["starting", "position", "ev"], ascending=[False, True, False])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gw", default="next")
    parser.add_argument("--strategy", default="balanced", choices=["balanced", "template", "differential"])
    args = parser.parse_args()

    # Demo synthetic pool — replace with real forecaster output in production
    import numpy as np
    rng = np.random.default_rng(0)
    n = 60
    positions = (["GK"] * 8 + ["DEF"] * 20 + ["MID"] * 22 + ["FWD"] * 10)
    demo = pd.DataFrame({
        "player_id": range(n),
        "web_name": [f"P{i}" for i in range(n)],
        "position": positions,
        "club": rng.integers(1, 21, n),
        "price": rng.uniform(4.0, 13.5, n).round(1),
        "ev": rng.uniform(1, 9, n).round(2),
    })
    squad = optimize_squad(demo)
    print(squad[["web_name", "position", "price", "ev", "starting", "captain"]].to_string(index=False))
