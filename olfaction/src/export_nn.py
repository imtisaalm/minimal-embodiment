#!/usr/bin/env python3
"""Train the final NN on all data and export weights to JSON.

Model: Linear(8,16) -> ReLU -> Linear(16,3) -> ReLU -> Linear(3,3)
Classes: baseline / fresh_plant / perfume (n=46)
Output: models/nn_3class_v0.2.json
Dropout is disabled at inference, so it is not exported.
"""
import json
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
feat = pd.read_csv(os.path.join(DATA, "features_v0.2.csv"))
CLASSES = ["baseline", "fresh_plant", "perfume"]
feat = feat[feat["label"].isin(CLASSES)].reset_index(drop=True)
F = ["gas_trough_d", "gas_mean_d", "humid_peak_d", "humid_mean_d",
     "temp_d", "gas_trough_pct", "gas_slope", "gas_std"]
classes = CLASSES
cls2i = {c: i for i, c in enumerate(classes)}

X = torch.tensor(feat[F].values, dtype=torch.float32)
y = torch.tensor([cls2i[c] for c in feat["label"]], dtype=torch.long)

# standardize on ALL data (for final deployment model)
mean = X.mean(dim=0)
std = X.std(dim=0, unbiased=False)  # ddof=0, matches sklearn StandardScaler
std[std == 0] = 1.0
X_norm = (X - mean) / std

# class weights
counts = np.bincount(y.numpy(), minlength=len(classes))
weights = torch.tensor(
    [len(y) / (len(classes) * max(c, 1)) for c in counts],
    dtype=torch.float32)

nc = len(classes)

# search for the best seed
best_seed = -1
best_acc = 0
best_model = None

for seed in range(50):
    torch.manual_seed(seed)
    m = nn.Sequential(
        nn.Linear(8, 16), nn.ReLU(), nn.Dropout(0.3),
        nn.Linear(16, nc), nn.ReLU(), nn.Dropout(0.3),
        nn.Linear(nc, nc),
    )
    opt = torch.optim.Adam(m.parameters(), lr=0.01)
    m.train()
    for epoch in range(300):
        logits = m(X_norm)
        loss = nn.CrossEntropyLoss(weight=weights)(logits, y)
        loss.backward()
        opt.step()
        opt.zero_grad()
    m.eval()
    with torch.no_grad():
        pred = m(X_norm).argmax(dim=1)
        acc = (pred == y).float().mean().item()
        if acc > best_acc:
            best_acc = acc
            best_seed = seed
            best_model = m

model = best_model
# Training accuracy on all data — used only for seed selection.
# Generalisation metrics come from evaluate_v0.2.py (group-aware CV).
print(f"Best seed={best_seed}, training accuracy: {best_acc:.0%}")

model.eval()
with torch.no_grad():
    logits = model(X_norm)
    pred = logits.argmax(dim=1)
    probs = torch.softmax(logits, dim=1)
    for i in range(len(y)):
        true_label = classes[y[i]]
        pred_label = classes[pred[i]]
        conf = probs[i, pred[i]].item()
        ok = "ok" if pred[i] == y[i] else "MISS"
        print(f"  {true_label:>12} -> {pred_label:>12}  {ok:<4}  {conf:.1%}")

# export weights
# model structure: [Linear, ReLU, Dropout, Linear, ReLU, Dropout, Linear]
# indices:          0       1     2        3       4     5        6
layer_indices = [0, 3, 6]  # the three Linear layers

exported = {
    "classes": classes,
    "feature_names": F,
    "scaler": {
        "mean": mean.tolist(),
        "std": std.tolist(),
    },
    "layers": [],
}

for idx in layer_indices:
    layer = model[idx]
    exported["layers"].append({
        "weight": layer.weight.data.tolist(),
        "bias": layer.bias.data.tolist(),
    })

MODELS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
out_path = os.path.join(MODELS, "nn_3class_v0.2.json")
with open(out_path, "w") as f:
    json.dump(exported, f, indent=2)

# verify: manual forward pass matches model output
def manual_forward(x_raw):
    x = [(x_raw[i] - exported["scaler"]["mean"][i]) / exported["scaler"]["std"][i] for i in range(len(x_raw))]
    for li, layer in enumerate(exported["layers"]):
        W = layer["weight"]
        b = layer["bias"]
        out = [sum(W[j][k] * x[k] for k in range(len(x))) + b[j] for j in range(len(b))]
        if li < len(exported["layers"]) - 1:  # ReLU except last layer
            out = [max(0, v) for v in out]
        x = out
    return x

# check first sample
raw = feat[F].values[0].tolist()
manual = manual_forward(raw)
with torch.no_grad():
    torch_out = model(X_norm[0:1]).squeeze().tolist()

print(f"\nManual forward pass vs PyTorch:")
print(f"  PyTorch: {[f'{v:.4f}' for v in torch_out]}")
print(f"  Manual:  {[f'{v:.4f}' for v in manual]}")
max_diff = max(abs(a - b) for a, b in zip(torch_out, manual))
print(f"  Max diff: {max_diff:.2e} {'OK' if max_diff < 1e-5 else 'MISMATCH'}")

print(f"\nExported to: {os.path.abspath(out_path)} ({os.path.getsize(out_path)} bytes)")
print(f"Parameters: {sum(p.numel() for p in model.parameters())}")
