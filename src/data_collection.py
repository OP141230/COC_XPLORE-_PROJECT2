"""
data_collection.py
------------------
Pulls live data from the official Fantasy Premier League (FPL) public API
and stores per-gameweek snapshots so player histories can be reconstructed
as proper time series.

Official endpoints used:
  - https://fantasy.premierleague.com/api/bootstrap-static/   (players, teams, GWs)
  - https://fantasy.premierleague.com/api/fixtures/           (fixtures + FDR)
  - https://fantasy.premierleague.com/api/element-summary/{player_id}/  (per-player history)
"""

import argparse
import json
import os
import time
from datetime import datetime

import requests

BASE_URL = "https://fantasy.premierleague.com/api"


def fetch_json(url: str, retries: int = 3, backoff: float = 1.5) -> dict:
    for attempt in range(retries):
        try:
            resp = requests.get(url, timeout=15, headers={"User-Agent": "FantasyXI/1.0"})
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            if attempt == retries - 1:
                raise
            time.sleep(backoff ** attempt)
    return {}


def fetch_bootstrap_static() -> dict:
    return fetch_json(f"{BASE_URL}/bootstrap-static/")


def fetch_fixtures() -> list:
    return fetch_json(f"{BASE_URL}/fixtures/")


def fetch_player_history(player_id: int) -> dict:
    return fetch_json(f"{BASE_URL}/element-summary/{player_id}/")


def collect_all(out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M")

    print("Fetching bootstrap-static (players, teams, gameweeks)...")
    bootstrap = fetch_bootstrap_static()
    with open(os.path.join(out_dir, f"bootstrap_{stamp}.json"), "w") as f:
        json.dump(bootstrap, f)

    print("Fetching fixtures...")
    fixtures = fetch_fixtures()
    with open(os.path.join(out_dir, f"fixtures_{stamp}.json"), "w") as f:
        json.dump(fixtures, f)

    players = bootstrap.get("elements", [])
    print(f"Fetching per-player gameweek history for {len(players)} players...")
    histories = {}
    for i, p in enumerate(players):
        pid = p["id"]
        try:
            histories[pid] = fetch_player_history(pid)
        except Exception as e:
            print(f"  failed for player {pid}: {e}")
        if i % 50 == 0:
            print(f"  {i}/{len(players)} done")
        time.sleep(0.05)  # be polite to the API

    with open(os.path.join(out_dir, f"player_histories_{stamp}.json"), "w") as f:
        json.dump(histories, f)

    print(f"Done. Raw data saved under {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect FPL data")
    parser.add_argument("--season", type=str, default="current")
    parser.add_argument("--out", type=str, default="data/raw")
    args = parser.parse_args()
    collect_all(args.out)
