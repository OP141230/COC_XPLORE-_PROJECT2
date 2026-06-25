# FantasyXI — Deep Learning, Game Theory & Reinforcement Learning for FPL

FantasyXI is an intelligent decision-support system for Fantasy Premier League (FPL)
that combines **deep learning** (BiLSTM/Attention player forecasting), **game theory**
(differential ownership & risk-reward analysis), and **reinforcement learning**
(sequential transfer & captaincy optimization) to recommend squads, transfers, and
captaincy choices across a full FPL season.

## 1. Project Structure

```
fantasyxi/
├── data/
│   ├── raw/                 # raw FPL API pulls (players, fixtures, gameweeks)
│   └── processed/           # cleaned, feature-engineered time series
├── src/
│   ├── data_collection.py   # pulls data from the official FPL API
│   ├── feature_engineering.py
│   ├── forecasting_model.py # BiLSTM + Attention point predictor
│   ├── game_theory.py       # differential ownership / EV-variance / rank-gain model
│   ├── rl_agent.py          # RL environment + agent for transfers & captaincy
│   ├── squad_optimizer.py   # ILP-based squad/transfer optimizer under budget+rules
│   └── utils.py
├── models/                  # saved model checkpoints (.pt)
├── notebooks/               # EDA & experiment notebooks
├── requirements.txt
└── README.md
```

## 2. Pipeline Overview

1. **Data Collection** (`data_collection.py`) — pulls current season data from the
   official FPL API (`https://fantasy.premierleague.com/api/`): player stats,
   fixtures, gameweek history, ownership %, and prices. Also stores a rolling
   per-gameweek snapshot so the same player has a real time series.
2. **Feature Engineering** (`feature_engineering.py`) — builds per-player time
   series of form, xG, xA, xGI, minutes, fixture difficulty (FDR), ownership
   delta, price change, and constructs sliding windows for sequence modelling.
3. **Forecasting** (`forecasting_model.py`) — a BiLSTM with an attention layer
   over the last N gameweeks predicts each player's expected points for the
   next 1–5 gameweeks.
4. **Game Theory** (`game_theory.py`) — converts point forecasts + ownership
   into an expected-rank-gain score, so the system can recommend high-EV
   "template" picks vs. high-variance "differential" picks depending on the
   user's rank goals.
5. **RL Agent** (`rl_agent.py`) — models transfers/captaincy as a sequential
   decision problem (MDP) across gameweeks with budget, free-transfer, and
   chip constraints, trained with a DQN/PPO-style agent to balance immediate
   vs long-term reward (total season points / rank).
6. **Squad Optimizer** (`squad_optimizer.py`) — given current forecasts, solves
   the constrained squad-selection problem (budget ≤ £100m, max 3 per club,
   formation rules) via integer linear programming (PuLP), used both as a
   stand-alone tool and as the RL agent's action space filter.

## 3. Quick Start

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 1. Pull data
python src/data_collection.py --season 2025-26 --out data/raw

# 2. Build features
python src/feature_engineering.py --in data/raw --out data/processed

# 3. Train forecaster
python src/forecasting_model.py --train --data data/processed --epochs 50

# 4. Run game-theory + squad optimizer for the next gameweek
python src/squad_optimizer.py --gw next --strategy balanced

# 5. Train/evaluate the RL transfer agent
python src/rl_agent.py --train --episodes 2000
```

## 4. Expected Results / Validation

- Forecasting model is evaluated with **R² and MAE per gameweek horizon**,
  benchmarked against the FPL API's own predicted points and against
  `OpenFPL` (an open-source baseline — see References).
- Game-theory module is validated by backtesting differential vs template
  picks over a past completed season and measuring realised rank gain.
- RL agent is validated by simulating a full season (38 GWs) against a
  greedy (points-maximizing, no foresight) baseline and comparing total
  season score and final rank.
