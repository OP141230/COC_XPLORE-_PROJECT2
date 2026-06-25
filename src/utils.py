"""utils.py - shared helpers used across the FantasyXI pipeline."""

import json
import os

import numpy as np


def set_seed(seed: int = 42):
    import random
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def save_json(obj, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def load_json(path: str):
    with open(path) as f:
        return json.load(f)


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2) + 1e-9
    return 1 - ss_res / ss_tot


def position_name(element_type: int) -> str:
    return {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}.get(element_type, "UNK")
