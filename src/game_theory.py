

import argparse

import numpy as np
import pandas as pd


def expected_rank_gain(ev: np.ndarray, ownership: np.ndarray, field_ev: float,
                        variance: np.ndarray, risk_lambda: float = 0.5) -> np.ndarray:
    """
    ev: predicted points for each candidate player
    ownership: fraction of managers (0-1) who own the player
    field_ev: average expected points across the whole player pool (proxy for "the field")
    variance: forecast variance / uncertainty for each player
    risk_lambda: 0 = pure rank protection (favor template), 1 = pure differential hunting
    """
    differential_factor = 1 - ownership
    upside = (ev - field_ev) * differential_factor
    risk_penalty = risk_lambda * np.sqrt(variance) * differential_factor
    return upside - risk_penalty


def classify_strategy(df: pd.DataFrame, ev_col="ev", own_col="selected_by_percent") -> pd.DataFrame:
    df = df.copy()
    df[own_col] = df[own_col].astype(float)
    median_own = df[own_col].median()
    median_ev = df[ev_col].median()
    conditions = [
        (df[own_col] >= median_own) & (df[ev_col] >= median_ev),
        (df[own_col] < median_own) & (df[ev_col] >= median_ev),
        (df[own_col] >= median_own) & (df[ev_col] < median_ev),
    ]
    choices = ["template_premium", "differential", "template_safe"]
    df["pick_type"] = np.select(conditions, choices, default="avoid") if hasattr(np, "select") else np.select(conditions, choices, default="avoid")
    return df


def recommend(df: pd.DataFrame, risk_lambda: float = 0.5, top_n: int = 15) -> pd.DataFrame:
    """
    df must have columns: player_id, web_name, position, ev (predicted points),
    selected_by_percent (0-100), variance (forecast variance).
    """
    df = df.copy()
    ownership = df["selected_by_percent"].astype(float) / 100.0
    field_ev = df["ev"].mean()
    df["rank_gain_score"] = expected_rank_gain(
        df["ev"].values, ownership.values, field_ev, df["variance"].values, risk_lambda
    )
    df = classify_strategy(df)
    return df.sort_values("rank_gain_score", ascending=False).head(top_n)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lambda_risk", type=float, default=0.5,
                         help="0 = protect rank (template), 1 = chase rank gain (differentials)")
    parser.add_argument("--top_n", type=int, default=15)
    args = parser.parse_args()

    # Example synthetic usage — replace with real forecaster output in production
    demo = pd.DataFrame({
        "player_id": range(10),
        "web_name": [f"Player_{i}" for i in range(10)],
        "position": ["MID"] * 10,
        "ev": np.random.uniform(2, 9, 10),
        "selected_by_percent": np.random.uniform(0.5, 60, 10),
        "variance": np.random.uniform(0.5, 4, 10),
    })
    print(recommend(demo, risk_lambda=args.lambda_risk, top_n=args.top_n))
