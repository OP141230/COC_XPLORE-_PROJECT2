

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

FEATURE_COLS = [
    "minutes", "total_points", "goals_scored", "assists",
    "expected_goals", "expected_assists", "expected_goal_involvements",
    "expected_goals_conceded", "bonus", "bps", "ict_index", "influence",
    "creativity", "threat", "was_home", "team_h_score", "team_a_score",
    "value", "transfers_balance", "selected",
]


def _latest(path_glob: str) -> str:
    files = sorted(glob.glob(path_glob))
    if not files:
        raise FileNotFoundError(f"No files matching {path_glob}")
    return files[-1]


def load_raw(raw_dir: str):
    bootstrap = json.load(open(_latest(os.path.join(raw_dir, "bootstrap_*.json"))))
    fixtures = json.load(open(_latest(os.path.join(raw_dir, "fixtures_*.json"))))
    histories = json.load(open(_latest(os.path.join(raw_dir, "player_histories_*.json"))))
    return bootstrap, fixtures, histories


def build_player_table(bootstrap: dict, histories: dict) -> pd.DataFrame:
    elements = {p["id"]: p for p in bootstrap.get("elements", [])}
    rows = []
    for pid_str, hist in histories.items():
        pid = int(pid_str)
        meta = elements.get(pid, {})
        for gw in hist.get("history", []):
            row = {col: gw.get(col, np.nan) for col in FEATURE_COLS}
            row["player_id"] = pid
            row["round"] = gw.get("round")
            row["web_name"] = meta.get("web_name")
            row["element_type"] = meta.get("element_type")  # 1 GK 2 DEF 3 MID 4 FWD
            row["team"] = meta.get("team")
            rows.append(row)
    df = pd.DataFrame(rows).sort_values(["player_id", "round"])
    return df


def add_rolling_features(df: pd.DataFrame, windows=(3, 5)) -> pd.DataFrame:
    df = df.copy()
    for w in windows:
        df[f"form_{w}gw"] = (
            df.groupby("player_id")["total_points"]
            .transform(lambda s: s.rolling(w, min_periods=1).mean())
        )
        df[f"xgi_{w}gw"] = (
            df.groupby("player_id")["expected_goal_involvements"]
            .transform(lambda s: s.rolling(w, min_periods=1).mean())
        )
        df[f"minutes_{w}gw"] = (
            df.groupby("player_id")["minutes"]
            .transform(lambda s: s.rolling(w, min_periods=1).mean())
        )
    df["ownership_delta"] = df.groupby("player_id")["selected"].diff().fillna(0)
    df["price_delta"] = df.groupby("player_id")["value"].diff().fillna(0)
    return df


def make_sequences(df: pd.DataFrame, seq_len: int = 5, target_col: str = "total_points"):
    """Return (X, y, meta) sliding-window sequences per player for the BiLSTM."""
    feature_cols = [c for c in df.columns if c not in (
        "player_id", "round", "web_name", "element_type", "team", target_col)]
    X, y, meta = [], [], []
    for pid, g in df.groupby("player_id"):
        g = g.sort_values("round").reset_index(drop=True)
        vals = g[feature_cols].fillna(0).values
        targets = g[target_col].fillna(0).values
        for i in range(len(g) - seq_len):
            X.append(vals[i:i + seq_len])
            y.append(targets[i + seq_len])
            meta.append((pid, g.loc[i + seq_len, "round"]))
    return np.array(X), np.array(y), meta, feature_cols


def main(raw_dir: str, out_dir: str, seq_len: int):
    os.makedirs(out_dir, exist_ok=True)
    bootstrap, fixtures, histories = load_raw(raw_dir)
    df = build_player_table(bootstrap, histories)
    df = add_rolling_features(df)
    df.to_parquet(os.path.join(out_dir, "player_gw_features.parquet"))

    X, y, meta, feature_cols = make_sequences(df, seq_len=seq_len)
    np.save(os.path.join(out_dir, "X_sequences.npy"), X)
    np.save(os.path.join(out_dir, "y_targets.npy"), y)
    json.dump(feature_cols, open(os.path.join(out_dir, "feature_cols.json"), "w"))
    print(f"Saved {len(df)} player-gameweek rows and {len(X)} training sequences to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="raw_dir", default="data/raw")
    parser.add_argument("--out", dest="out_dir", default="data/processed")
    parser.add_argument("--seq_len", type=int, default=5)
    args = parser.parse_args()
    main(args.raw_dir, args.out_dir, args.seq_len)
