#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Lun Lug 07 11:05:49 2025

@author: Sundas Sarwar
"""

# %%


import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from imblearn.over_sampling import SMOTENC

DATA_PATH = "./dataset/ashrae_db2.01.csv"      # change to the file location
SEEDS = [42, 1, 2, 3, 4]                     # one random split per seed (5 runs)

# LOAD DATASET

try:
    df = pd.read_csv(DATA_PATH, low_memory=False)
except UnicodeDecodeError:
    df = pd.read_csv(DATA_PATH, encoding="ISO-8859-1", low_memory=False)

print("Original shape:", df.shape)

# ------------------------------------------------------------
# 1. DROP DUPLICATE ROWS
# ------------------------------------------------------------

n_before = len(df)
df = df.drop_duplicates().reset_index(drop=True)
print("Duplicate rows removed:", n_before - len(df))

# 2. THERMAL SENSATION: 7 POINTS -> 5 POINTS

# -3, -2 -> -2 ;  -1 -> -1 ;  0 -> 0 ;  +1 -> +1 ;  +2, +3 -> +2
# Votes such as -0.5 or 2.1 are first rounded to the nearest point.
# Rows without a thermal sensation vote are removed.

TARGET = "Thermal sensation"

df = df.dropna(subset=[TARGET]).reset_index(drop=True)
tsv_5 = np.clip(np.round(df[TARGET]), -2, 2)
y = (tsv_5 + 2).astype(int).to_numpy()        
df = df.drop(columns=[TARGET])

print("\nFive-point distribution (0..4 = -2..+2):")
print(pd.Series(y).value_counts().sort_index())

# FEATURE TYPES

# Five contextual features used by HCA.
contextual_features = ["Building type", "City", "Season",
                       "Koppen climate classification", "Country"]

# Text columns are categorical; all other columns are numerical.
categorical_features = [c for c in df.columns
                        if df[c].dtype == object or str(df[c].dtype).startswith("str")
                        or c in contextual_features]
numerical_features = [c for c in df.columns if c not in categorical_features]

print("\nNumerical features:", len(numerical_features))
print("Categorical features:", len(categorical_features))

# 3. IMPUTE MISSING VALUES + STANDARDIZE

# Categorical: most frequent value.
for col in categorical_features:
    df[col] = df[col].map(lambda v: str(v).strip() if pd.notna(v) else np.nan).astype(object)
    df[col] = df[col].fillna(df[col].mode(dropna=True).iloc[0])

# Numerical: KNN imputer (k = 5).
print("\nKNN imputation ...")
imputer = KNNImputer(n_neighbors=5)
imputer.fit(df[numerical_features].to_numpy())
values = df[numerical_features].to_numpy()
# transformed in batches of 1,000 rows to keep memory use low
df[numerical_features] = np.vstack(
    [imputer.transform(values[i:i + 1000]) for i in range(0, len(values), 1000)])

# Standardize numerical features (needed so that DBSCAN distances are comparable).
scaler = StandardScaler()
df[numerical_features] = scaler.fit_transform(df[numerical_features].to_numpy())

# 4. DBSCAN OUTLIER REMOVAL

print("\nDBSCAN (eps = 5, min_samples = 3) ...")
dbscan = DBSCAN(eps=5, min_samples=3, n_jobs=-1)
labels = dbscan.fit_predict(df[numerical_features].to_numpy())

keep = labels != -1
print("DBSCAN outliers removed:", int((~keep).sum()),
      f"of {len(df)} ({100 * (~keep).mean():.2f} %)")

df = df[keep].reset_index(drop=True)
y = y[keep]

# ENCODE CATEGORICAL FEATURES (integer codes for SMOTE-NC and HCA embeddings)

for col in categorical_features:
    df[col] = df[col].astype("category").cat.codes

# Context columns first, then all other features.
columns = contextual_features + [c for c in df.columns if c not in contextual_features]
df = df[columns]
cat_indices = [df.columns.get_loc(c) for c in categorical_features]

# 5. FIVE RANDOM SPLITS (70 / 10 / 20) 

for run, seed in enumerate(SEEDS, start=1):
    print(f"\n========== Run {run} (seed {seed}) ==========")

    # 70 % train, 10 % validation, 20 % test, stratified by class
    idx = np.arange(len(df))
    train_idx, rest_idx = train_test_split(idx, test_size=0.30, stratify=y, random_state=seed)
    val_idx, test_idx = train_test_split(rest_idx, test_size=2 / 3, stratify=y[rest_idx],
                                         random_state=seed)

    train, y_train = df.iloc[train_idx], y[train_idx]
    val, y_val = df.iloc[val_idx], y[val_idx]
    test, y_test = df.iloc[test_idx], y[test_idx]
    print("Train:", train.shape, " Validation:", val.shape, " Test:", test.shape)

    counts = pd.Series(y_train).value_counts().sort_index()
    target_count = int(min(counts[1], counts[2], counts[3]))
    sampling_strategy = {cls: max(target_count, int(counts[cls])) for cls in (0, 4)}

    smote = SMOTENC(categorical_features=cat_indices,
                    sampling_strategy=sampling_strategy,
                    random_state=seed)
    train_bal, y_train_bal = smote.fit_resample(train, y_train)
    n_synthetic = len(y_train_bal) - len(y_train)

    print("Training classes before SMOTE:", counts.tolist())
    print("Training classes after SMOTE: ", pd.Series(y_train_bal).value_counts().sort_index().tolist())
    print("Synthetic training rows:", n_synthetic)

    # save
    out_train = train_bal.copy()
    out_train["label"] = y_train_bal
    out_train["is_synthetic"] = np.r_[np.zeros(len(y_train), int), np.ones(n_synthetic, int)]
    out_val = val.copy()
    out_val["label"] = y_val
    out_test = test.copy()
    out_test["label"] = y_test

    out_train.to_csv(f"ashrae_run{run}_train.csv", index=False)
    out_val.to_csv(f"ashrae_run{run}_val.csv", index=False)
    out_test.to_csv(f"ashrae_run{run}_test.csv", index=False)
    print(f"Saved: ashrae_run{run}_train.csv, ashrae_run{run}_val.csv, ashrae_run{run}_test.csv")

print("\nPreprocessing completed (5 runs).")


# %%
