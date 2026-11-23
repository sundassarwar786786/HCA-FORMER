#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Lun Lug 07 10:51:02 2025

@author: Sundas Sarwar
"""

# %%

import argparse
import copy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    cohen_kappa_score,
    matthews_corrcoef,
)

# 1. HYPERPARAMTERS

parser = argparse.ArgumentParser(description="HCA-Former on the ASHRAE dataset")
parser.add_argument("--preprocessed_dir", type=str, default=".",
                    help="Folder with ashrae_split1_train/val/test.csv")
parser.add_argument("--file_tag", type=str, default="ashrae_split1",
                    help="Use ashrae_split1_noDBSCAN for the data without outlier removal")
parser.add_argument("--seeds", type=str, default="42,1,2,3,4", help="One run per seed (5 runs)")
parser.add_argument("--epochs", type=int, default=100, help="Training epochs per run")
parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
parser.add_argument("--learning_rate", type=float, default=1e-3, help="Adam learning rate")
parser.add_argument("--embed_dim", type=int, default=128, help="Embedding dimension")
parser.add_argument("--num_heads", type=int, default=8, help="Attention heads")
parser.add_argument("--depth", type=int, default=2, help="Number of HCA-Former blocks")
parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
parser.add_argument("--kernel", type=int, default=3, help="Conv1D kernel size for Q, K, V")
args, _ = parser.parse_known_args()     # parse_known_args also works in Jupyter / Kaggle

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

# 2. LOAD THE PREPROCESSED DATASET

data_dir = Path(args.preprocessed_dir)
train_df = pd.read_csv(data_dir / f"{args.file_tag}_train.csv")
val_df = pd.read_csv(data_dir / f"{args.file_tag}_val.csv")
test_df = pd.read_csv(data_dir / f"{args.file_tag}_test.csv")

print("Train:", train_df.shape, "| Validation:", val_df.shape, "| Test:", test_df.shape)

# 3. PREPARE THE INPUTS

hier_cols = ["Building type", "City", "Season", "Koppen climate classification", "Country"]
target_col = "label"                   

sensor_cols = [c for c in train_df.columns if c not in hier_cols + [target_col, "is_synthetic"]]
print("Sensor features:", len(sensor_cols), "| Context features:", len(hier_cols))

mean = train_df[sensor_cols].mean()
std = train_df[sensor_cols].std().replace(0, 1)


def split_columns(df):
    X_sensors = ((df[sensor_cols] - mean) / std).to_numpy(np.float32)
    X_hier = df[hier_cols].to_numpy(np.int64)
    y = df[target_col].to_numpy(np.int64)
    return X_sensors, X_hier, y


X_sensors_train, X_hier_train, y_train = split_columns(train_df)
X_sensors_val, X_hier_val, y_val = split_columns(val_df)
X_sensors_test, X_hier_test, y_test = split_columns(test_df)

num_classes = 5
# number of categories of each context feature (+1 for the "Unknown" code)
num_categories = [int(max(X_hier_train[:, i].max(), X_hier_val[:, i].max(), X_hier_test[:, i].max())) + 1
                  for i in range(len(hier_cols))]

print("Training classes (-2..+2):", np.bincount(y_train).tolist())

# 4. DATASET AND DATALOADERS

class SensorDataset(Dataset):
    def __init__(self, X_sensors, X_hier, y):
        self.X_sensors = torch.tensor(X_sensors, dtype=torch.float32)
        self.X_hier = torch.tensor(X_hier, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X_sensors[idx], self.X_hier[idx], self.y[idx]


val_loader = DataLoader(SensorDataset(X_sensors_val, X_hier_val, y_val),
                        batch_size=1024, shuffle=False)
test_loader = DataLoader(SensorDataset(X_sensors_test, X_hier_test, y_test),
                         batch_size=1024, shuffle=False)

# 5. HCA-FORMER FRAMEWORK


# ---------------------------------------------------------------- Step 1
class InputEmbedding(nn.Module):

    def __init__(self, N, d):
        super().__init__()
        self.w = nn.Parameter(torch.empty(N, d))
        nn.init.xavier_uniform_(self.w)
        self.b = nn.Parameter(torch.zeros(N, d))
        self.x_cls = nn.Parameter(torch.zeros(1, 1, d))

        position = torch.arange(N + 1).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d, 2) * (-np.log(10000.0) / d))
        P = torch.zeros(N + 1, d)
        P[:, 0::2] = torch.sin(position * div_term)
        P[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("P", P)

    def forward(self, X_env):                                   # X_env: (batch, N)
        tokens = X_env.unsqueeze(-1) * self.w + self.b          # Embed(X_env): (batch, N, d)
        x_cls = self.x_cls.expand(len(X_env), -1, -1)           # (batch, 1, d)
        return torch.cat([x_cls, tokens], dim=1) + self.P       # X: (batch, N+1, d)


# ---------------------------------------------------------------- Steps 2 and 3
class HierarchicalContext(nn.Module):

    def __init__(self, num_categories, d):
        super().__init__()
        n_building, n_city, n_season, n_climate, n_country = num_categories
        # Step 2: one learnable embedding per contextual variable
        self.embed_building = nn.Embedding(n_building, d)
        self.embed_city = nn.Embedding(n_city, d)
        self.embed_season = nn.Embedding(n_season, d)
        self.embed_climate = nn.Embedding(n_climate, d)
        self.embed_country = nn.Embedding(n_country, d)
        # Step 3: gates, fusion projections P1..P4 and output projections W1..W4
        self.alpha = nn.Parameter(torch.zeros(1))
        self.beta = nn.Parameter(torch.zeros(1))
        self.gamma = nn.Parameter(torch.zeros(1))
        self.zeta = nn.Parameter(torch.zeros(1))
        self.P1, self.P2, self.P3, self.P4 = (nn.Linear(d, d) for _ in range(4))
        self.W1, self.W2, self.W3, self.W4 = (nn.Linear(d, d) for _ in range(4))

    def forward(self, X_context):          # X_context: (batch, 5) = building, city, season, climate, country
        # Step 2: encode contextual variables
        E_building = self.embed_building(X_context[:, 0])
        E_city = self.embed_city(X_context[:, 1])
        E_season = self.embed_season(X_context[:, 2])
        E_climate = self.embed_climate(X_context[:, 3])
        E_country = self.embed_country(X_context[:, 4])

        # Step 3: sequential gated fusion
        alpha, beta = torch.sigmoid(self.alpha), torch.sigmoid(self.beta)
        gamma, zeta = torch.sigmoid(self.gamma), torch.sigmoid(self.zeta)
        h1 = self.P1(alpha * E_building + (1 - alpha) * E_city)
        h2 = self.P2(beta * h1 + (1 - beta) * E_season)
        h3 = self.P3(gamma * h2 + (1 - gamma) * E_climate)
        h4 = self.P4(zeta * h3 + (1 - zeta) * E_country)

        H_c = torch.stack([self.W1(h1), self.W2(h2), self.W3(h3), self.W4(h4)], dim=1)
        return H_c                                              # (batch, 4, d)

    def gates(self):
        return {name: torch.sigmoid(getattr(self, name)).item()
                for name in ("alpha", "beta", "gamma", "zeta")}


# ---------------------------------------------------------------- Step 4: HCA
class HCA(nn.Module):
    """Cross-attention: feature tokens give the queries, H_c gives the keys and values."""

    def __init__(self, d, num_heads, dropout):
        super().__init__()
        self.attention = nn.MultiheadAttention(d, num_heads, dropout=dropout, batch_first=True)

    def forward(self, X_norm, H_c):
        output, _ = self.attention(query=X_norm, key=H_c, value=H_c)
        return output


# ---------------------------------------------------------------- Step 5: MHSA
class MHSA(nn.Module):
    """Q = Conv1D_Q(Z), K = Conv1D_K(Z), V = Conv1D_V(Z)   with Z = LayerNorm(X_t)
    A = softmax(Q K^T / sqrt(d_h) + B)   (B = 0: no mask)
    Z' = A V for every head;  MHSA = [Z'_1 || ... || Z'_H] W_O"""

    def __init__(self, d, num_heads, kernel, dropout):
        super().__init__()
        self.H = num_heads
        self.d_h = d // num_heads
        self.conv_Q = nn.Conv1d(d, d, kernel, padding=kernel // 2)
        self.conv_K = nn.Conv1d(d, d, kernel, padding=kernel // 2)
        self.conv_V = nn.Conv1d(d, d, kernel, padding=kernel // 2)
        self.W_O = nn.Linear(d, d)
        self.dropout = nn.Dropout(dropout)

    def forward(self, Z, B=0.0):                                # Z: (batch, T, d)
        batch, T, d = Z.shape
        Zc = Z.transpose(1, 2)                                  # Conv1D runs along the token axis

        def split_heads(t):                                     # (batch, d, T) -> (batch, H, T, d_h)
            return t.transpose(1, 2).reshape(batch, T, self.H, self.d_h).transpose(1, 2)

        Q = split_heads(self.conv_Q(Zc))
        K = split_heads(self.conv_K(Zc))
        V = split_heads(self.conv_V(Zc))

        A = torch.softmax(Q @ K.transpose(-2, -1) / np.sqrt(self.d_h) + B, dim=-1)
        Z_prime = self.dropout(A) @ V                           # Z'_m = A V, per head
        Z_concat = Z_prime.transpose(1, 2).reshape(batch, T, d) # [Z'_1 || ... || Z'_H]
        return self.W_O(Z_concat)


# ---------------------------------------------------------------- Steps 4-6: one block
class HCAFormerBlock(nn.Module):
    def __init__(self, d, num_heads, kernel, dropout):
        super().__init__()
        self.LN_hca = nn.LayerNorm(d)
        self.LN_mhsa = nn.LayerNorm(d)
        self.LN_mlp = nn.LayerNorm(d)
        self.hca = HCA(d, num_heads, dropout)
        self.mhsa = MHSA(d, num_heads, kernel, dropout)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(4 * d, d))
        self.dropout = nn.Dropout(dropout)

    def forward(self, X, H_c):
        X_t = self.dropout(self.hca(self.LN_hca(X), H_c)) + X            # Step 4
        X_st = self.dropout(self.mhsa(self.LN_mhsa(X_t))) + X_t          # Step 5
        X_out = self.dropout(self.mlp(self.LN_mlp(X_st))) + X_st         # Step 6  (X')
        return X_out


# ---------------------------------------------------------------- full model
class HCAFormer(nn.Module):
    def __init__(self, num_sensors, num_categories, num_classes,
                 embed_dim=128, num_heads=8, depth=2, kernel=3, dropout=0.1):
        super().__init__()
        d = embed_dim
        self.input_embedding = InputEmbedding(num_sensors, d)                       # Step 1
        self.hierarchical_context = HierarchicalContext(num_categories, d)          # Steps 2-3
        self.blocks = nn.ModuleList([HCAFormerBlock(d, num_heads, kernel, dropout)  # Steps 4-6, L times
                                     for _ in range(depth)])
        # Step 8: decoder  W_p GELU(W_d LayerNorm(x_cls) + b_d) + b_p
        self.decoder_LN = nn.LayerNorm(d)
        self.W_d = nn.Linear(d, d)
        self.W_p = nn.Linear(d, num_classes)

        # Glorot-uniform initialization (Table I)
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv1d)):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, X_env, X_context):
        X = self.input_embedding(X_env)                         # Step 1
        H_c = self.hierarchical_context(X_context)              # Steps 2-3
        for block in self.blocks:                               # Steps 4-6 in each of the L blocks
            X = block(X, H_c)
        x_cls = X[:, 0]                                         # Step 7: classification token
        r = torch.nn.functional.gelu(self.W_d(self.decoder_LN(x_cls)))   # Step 8
        return self.W_p(r)          # class scores; softmax is applied in the loss and in predict_proba

    def predict_proba(self, X_env, X_context):                  # Steps 8-9: y_hat = softmax(...)
        return torch.softmax(self.forward(X_env, X_context), dim=1)

# 6. EVALUATION FUNCTION

def evaluate_model(model, loader, title):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for sensors, hier, labels in loader:
            outputs = model(sensors.to(device), hier.to(device))
            all_preds.extend(outputs.argmax(dim=1).cpu().numpy())
            all_labels.extend(labels.numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="weighted", zero_division=0
    )
    kappa = cohen_kappa_score(all_labels, all_preds)
    mcc = matthews_corrcoef(all_labels, all_preds)

    print(f"\n=== {title} ===")
    print(f"✅ Accuracy: {accuracy:.4f}")
    print(f"✅ Precision: {precision:.4f}")
    print(f"✅ Recall: {recall:.4f}")
    print(f"✅ F1-Score: {f1:.4f}")
    print(f"✅ Cohen’s Kappa: {kappa:.4f}")
    print(f"✅ MCC: {mcc:.4f}")

    return accuracy, precision, recall, f1, kappa, mcc

# 7. TRAINING: 5 RUNS, SAME SPLIT, DIFFERENT SEEDS

seeds = [int(s) for s in args.seeds.split(",")]
run_results = []

for run, seed in enumerate(seeds, start=1):
    print(f"\n==================== Run {run}/{len(seeds)} (seed {seed}) ====================")
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_loader = DataLoader(SensorDataset(X_sensors_train, X_hier_train, y_train),
                              batch_size=args.batch_size, shuffle=True)

    model = HCAFormer(
        num_sensors=len(sensor_cols), num_categories=num_categories, num_classes=num_classes,
        embed_dim=args.embed_dim, num_heads=args.num_heads, depth=args.depth,
        kernel=args.kernel, dropout=args.dropout,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    best_val_loss = float("inf")
    best_model_wts = None
    best_epoch = 0

    for epoch in range(args.epochs):
        # ---- training
        model.train()
        total_loss, correct, total = 0, 0, 0
        for sensors, hier, labels in train_loader:
            sensors, hier, labels = sensors.to(device), hier.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(sensors, hier)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total
        train_loss = total_loss / len(train_loader)

        # ---- validation
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for sensors, hier, labels in val_loader:
                sensors, hier, labels = sensors.to(device), hier.to(device), labels.to(device)
                outputs = model(sensors, hier)
                val_loss += criterion(outputs, labels).item() * labels.size(0)
                val_correct += (outputs.argmax(1) == labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / val_total
        val_loss = val_loss / val_total

        print(f"Epoch {epoch + 1}/{args.epochs} | Train Acc: {train_acc:.4f} | Train Loss: {train_loss:.4f}"
              f" | Validation Loss: {val_loss:.4f} | Validation Acc: {val_acc:.4f}")

        # ---- keep the model of the epoch with the lowest VALIDATION loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            best_model_wts = copy.deepcopy(model.state_dict())  

    model.load_state_dict(best_model_wts)
    print(f"Best epoch: {best_epoch} (validation loss {best_val_loss:.4f})")

    # ---- evaluate this run on the TEST set (never used for training or model selection)
    run_results.append(evaluate_model(model, test_loader, f"Run {run} Results (Test set)"))

    # save the test predictions of this run
    model.eval()
    with torch.no_grad():
        probs = torch.cat([model.predict_proba(s.to(device), h.to(device)).cpu()
                           for s, h, _ in test_loader]).numpy()
    np.savez(f"hca_ashrae_run{run}_test_predictions.npz", y_true=y_test, proba=probs)

# 8. FINAL RESULTS: MEAN ± SD OVER THE 5 RUNS
run_results = np.array(run_results)
mean_results = run_results.mean(axis=0)
sd_results = run_results.std(axis=0, ddof=1) if len(run_results) > 1 else np.zeros(6)

print(f"\n=== Final Results (Test set, mean ± SD over {len(run_results)} runs) ===")
print(f"✅ Accuracy: {mean_results[0]:.4f} ± {sd_results[0]:.4f}")
print(f"✅ Precision: {mean_results[1]:.4f} ± {sd_results[1]:.4f}")
print(f"✅ Recall: {mean_results[2]:.4f} ± {sd_results[2]:.4f}")
print(f"✅ F1-Score: {mean_results[3]:.4f} ± {sd_results[3]:.4f}")
print(f"✅ Cohen’s Kappa: {mean_results[4]:.4f} ± {sd_results[4]:.4f}")
print(f"✅ MCC: {mean_results[5]:.4f} ± {sd_results[5]:.4f}")

