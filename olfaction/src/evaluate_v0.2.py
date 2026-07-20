#!/usr/bin/env python3
"""Three-level group-aware cross-validation for the neural network classifier.

Three levels of increasing strictness:

  Leave-one-out        -- hold out a single sample (control baseline).
  Leave-one-session-out -- hold out all samples from one recording file,
      ensuring no recording source is shared between train and test.
  Leave-one-date-out   -- hold out all data from an entire calendar date.
      This is the only level whose folds cross a sensor-drift epoch,
      exposing class-date confounding that sub-date groupings cannot detect.

Stable accuracy across all three levels indicates genuine odour
discrimination rather than session-level leakage or drift artifacts.

    python3 evaluate_v0.2.py
"""
import os
import re

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import LeaveOneGroupOut, LeaveOneOut
from sklearn.preprocessing import StandardScaler

torch.set_num_threads(1)
SEEDS = list(range(20))
CLASSES = ["baseline", "fresh_plant", "perfume"]
FEATURES = ["gas_trough_d", "gas_mean_d", "humid_peak_d", "humid_mean_d",
            "temp_d", "gas_trough_pct", "gas_slope", "gas_std"]

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
f = pd.read_csv(os.path.join(DATA, "features_v0.2.csv"))
f = f[f["label"].isin(CLASSES)].reset_index(drop=True)
f["date"] = f["source_file"].map(lambda s: re.search(r"(\d{4}-\d{2}-\d{2})", str(s)).group(1))

c2i = {c: i for i, c in enumerate(CLASSES)}
X = f[FEATURES].to_numpy(np.float32)
y = np.array([c2i[c] for c in f["label"]], np.int64)
counts = np.bincount(y, minlength=len(CLASSES)).astype(np.float32)
class_w = torch.tensor(counts.sum() / (len(CLASSES) * counts))


class MLP(nn.Module):
    """Deployed architecture: 8 -> 16 -> 3 -> 3, dropout 0.3 after each ReLU."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(8, 16), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(16, 3), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(3, 3),
        )

    def forward(self, x):
        return self.net(x)


def one_seed(splitter, split_args, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    y_true, y_pred = [], []
    for tr, te in splitter.split(X, y, *split_args):
        scaler = StandardScaler().fit(X[tr])
        Xtr = torch.tensor(scaler.transform(X[tr]), dtype=torch.float32)
        Xte = torch.tensor(scaler.transform(X[te]), dtype=torch.float32)
        model = MLP()
        opt = torch.optim.Adam(model.parameters(), lr=0.01)
        lossf = nn.CrossEntropyLoss(weight=class_w)
        model.train()
        for _ in range(300):
            opt.zero_grad()
            lossf(model(Xtr), torch.tensor(y[tr])).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            y_pred += model(Xte).argmax(1).numpy().tolist()
        y_true += y[te].tolist()
    return accuracy_score(y_true, y_pred), balanced_accuracy_score(y_true, y_pred)


LEVELS = [
    ("Leave-one-out",         "Individual samples", LeaveOneOut(),      ()),
    ("Leave-one-session-out", "Recording files",    LeaveOneGroupOut(), (f["source_file"].to_numpy(),)),
    ("Leave-one-date-out",    "Calendar dates",     LeaveOneGroupOut(), (f["date"].to_numpy(),)),
]

print(f"n = {len(y)}   " + "  ".join(f"{c}={int((y == i).sum())}" for c, i in c2i.items()))
print(f"seeds = {SEEDS}\n")
print(f"{'Level':<24}{'Groups':>7}{'Accuracy':>18}{'Balanced acc':>20}{'Acc range':>14}")
for name, _, sp, args in LEVELS:
    n_groups = len(np.unique(args[0])) if args else len(y)
    r = [one_seed(sp, args, s) for s in SEEDS]
    acc, bal = np.array([x[0] for x in r]), np.array([x[1] for x in r])
    print(f"{name:<24}{n_groups:>7}"
          f"{acc.mean() * 100:>12.1f}% +/-{acc.std() * 100:>3.1f}"
          f"{bal.mean() * 100:>14.1f}% +/-{bal.std() * 100:>3.1f}"
          f"{acc.min() * 100:>9.0f}-{acc.max() * 100:.0f}%")
