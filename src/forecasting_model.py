"""
forecasting_model.py
---------------------
BiLSTM + Attention network that forecasts a player's expected points
for the next gameweek(s) from a sliding window of past performance
features (form, xG, xA, xGI, minutes, fixture difficulty, ownership, price).
"""

import argparse
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split


class SequenceDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class Attention(nn.Module):
    """Simple additive attention over BiLSTM hidden states."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, lstm_out):
        # lstm_out: (batch, seq_len, hidden_dim)
        scores = self.attn(lstm_out).squeeze(-1)          # (batch, seq_len)
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)  # (batch, seq_len, 1)
        context = (lstm_out * weights).sum(dim=1)          # (batch, hidden_dim)
        return context, weights


class BiLSTMAttentionForecaster(nn.Module):
    def __init__(self, n_features: int, hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features, hidden_size=hidden_dim, num_layers=num_layers,
            batch_first=True, bidirectional=True, dropout=dropout if num_layers > 1 else 0.0,
        )
        self.attention = Attention(hidden_dim * 2)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        context, attn_weights = self.attention(lstm_out)
        pred = self.head(context)
        return pred.squeeze(-1), attn_weights


def train(data_dir: str, model_out: str, epochs: int, batch_size: int = 64, lr: float = 1e-3):
    X = np.load(os.path.join(data_dir, "X_sequences.npy"))
    y = np.load(os.path.join(data_dir, "y_targets.npy"))

    dataset = SequenceDataset(X, y)
    n_val = max(1, int(0.15 * len(dataset)))
    train_ds, val_ds = random_split(dataset, [len(dataset) - n_val, n_val])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = BiLSTMAttentionForecaster(n_features=X.shape[-1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    best_val = float("inf")
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred, _ = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(xb)
        train_loss /= len(train_ds)

        model.eval()
        val_loss, preds, trues = 0.0, [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred, _ = model(xb)
                val_loss += criterion(pred, yb).item() * len(xb)
                preds.append(pred.cpu().numpy())
                trues.append(yb.cpu().numpy())
        val_loss /= len(val_ds)
        preds, trues = np.concatenate(preds), np.concatenate(trues)
        ss_res = np.sum((trues - preds) ** 2)
        ss_tot = np.sum((trues - trues.mean()) ** 2) + 1e-9
        r2 = 1 - ss_res / ss_tot

        print(f"Epoch {epoch+1}/{epochs} | train_mse={train_loss:.3f} val_mse={val_loss:.3f} val_R2={r2:.3f}")

        if val_loss < best_val:
            best_val = val_loss
            os.makedirs(os.path.dirname(model_out), exist_ok=True)
            torch.save(model.state_dict(), model_out)

    print(f"Best model saved to {model_out} (val_mse={best_val:.3f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--data", default="data/processed")
    parser.add_argument("--model_out", default="models/bilstm_attention.pt")
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()
    if args.train:
        train(args.data, args.model_out, args.epochs)
