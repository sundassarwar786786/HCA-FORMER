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
from imblearn.over_sampling import SMOTE

DATA_PATH = "./dataset/ashrae_db2.01.csv"      # change to file location

# "fixed"  = ONE fixed 70/10/20 partition (seed 42), as in the paper: the five runs then use the
#            same partition with a different training seed.
# "random" = five different random 70/10/20 partitions (seeds 42, 1, 2, 3, 4).
SPLIT_MODE = "fixed"
SPLIT_SEEDS = [42] if SPLIT_MODE == "fixed" else [42, 1, 2, 3, 4]

APPLY_DBSCAN = True

# LOAD DATASET
try:
    df = pd.read_csv(DATA_PATH, low_memory=False)
except UnicodeDecodeError:
    df = pd.read_csv(DATA_PATH, encoding="ISO-8859-1", low_memory=False)

print("Original shape:", df.shape)

# DROP DUPLICATE ROWS

n_before = len(df)
df = df.drop_duplicates()
print("Duplicate rows removed:", n_before - len(df))

# THERMAL SENSATION: 7 POINTS -> 5 POINTS

# -3, -2 -> -2 ;  -1 -> -1 ;  0 -> 0 ;  +1 -> +1 ;  +2, +3 -> +2
# Some votes are not whole numbers (e.g. -0.5, 2.1); they are first
# rounded to the nearest scale point. Rows without a vote are removed.

TARGET = "Thermal sensation"

df = df.dropna(subset=[TARGET]).copy()
tsv_5 = np.clip(np.round(df[TARGET]), -2, 2)
df["label"] = (tsv_5 + 2).astype(int)        
df = df.drop(columns=[TARGET])

print("\nFive-point distribution (0..4 = -2..+2):")
print(df["label"].value_counts().sort_index().tolist())

# REMOVE REDUNDANT COLUMNS

# Fahrenheit and fpm columns repeat the Celsius and m/s columns.
df = df.drop(columns=[c for c in df.columns if "(F)" in c or "(fpm)" in c])

# RULE-BASED PHYSICAL VALIDATION

# Implausible readings are set to missing (the row is kept and the
# value is filled later by KNN imputation).
physical_limits = {
    "Air temperature (C)": (5, 45), "Ta_h (C)": (5, 45), "Ta_m (C)": (5, 45), "Ta_l (C)": (5, 45),
    "Operative temperature (C)": (5, 45), "Radiant temperature (C)": (5, 50),
    "Globe temperature (C)": (5, 50), "Tg_h (C)": (5, 50), "Tg_m (C)": (5, 50), "Tg_l (C)": (5, 50),
    "Relative humidity (%)": (0, 100),
    "Air velocity (m/s)": (0, 5), "Velocity_h (m/s)": (0, 5),
    "Velocity_m (m/s)": (0, 5), "Velocity_l (m/s)": (0, 5),
    "Clo": (0, 3), "Met": (0.6, 4), "Age": (10, 100),
    "height (cm)": (100, 220), "weight (kg)": (25, 250),
    "Outdoor monthly air temperature (C)": (-40, 50),
}

invalid_counts = {}
for col, (lower, upper) in physical_limits.items():
    invalid = df[col].notna() & ~df[col].between(lower, upper)
    if invalid.any():
        invalid_counts[col] = int(invalid.sum())
        df.loc[invalid, col] = np.nan

print("\nImplausible values set to missing:", invalid_counts)

# ZONE-LEVEL DESCRIPTORS (MEAN, VARIANCE, MIN-MAX RANGE)

zones = {
    "Ta_zone": ["Ta_h (C)", "Ta_m (C)", "Ta_l (C)"],
    "Tg_zone": ["Tg_h (C)", "Tg_m (C)", "Tg_l (C)"],
    "Velocity_zone": ["Velocity_h (m/s)", "Velocity_m (m/s)", "Velocity_l (m/s)"],
}
for name, cols in zones.items():
    df[name + "_mean"] = df[cols].mean(axis=1)
    df[name + "_var"] = df[cols].var(axis=1, ddof=0)
    df[name + "_range"] = df[cols].max(axis=1) - df[cols].min(axis=1)
    df = df.drop(columns=cols)

# FEATURE TYPES

# Five contextual features used by HCA.
contextual_features = ["Building type", "City", "Season",
                       "Koppen climate classification", "Country"]

df = df.reset_index(drop=True)
y = df.pop("label").to_numpy()

# Text columns (and the context columns) are categorical; the rest are numerical.
categorical_features = [c for c in df.columns
                        if df[c].dtype == object or str(df[c].dtype).startswith("str")
                        or c in contextual_features]
numerical_features_all = [c for c in df.columns if c not in categorical_features]
for col in categorical_features:
    df[col] = df[col].map(lambda v: str(v).strip() if pd.notna(v) else np.nan).astype(object)

print("\nRows:", len(df), "| numerical features:", len(numerical_features_all),
      "| categorical features:", len(categorical_features))

# SPLIT

for run, seed in enumerate(SPLIT_SEEDS, start=1):
    print(f"\n========== Split {run} (seed {seed}) ==========")

    # ---------------- 70 % train, 10 % validation, 20 % test (stratified)
    idx = np.arange(len(df))
    train_idx, rest_idx = train_test_split(idx, test_size=0.30, stratify=y, random_state=seed)
    val_idx, test_idx = train_test_split(rest_idx, test_size=2 / 3, stratify=y[rest_idx],
                                         random_state=seed)

    train, val, test = df.iloc[train_idx].copy(), df.iloc[val_idx].copy(), df.iloc[test_idx].copy()
    y_train, y_val, y_test = y[train_idx], y[val_idx], y[test_idx]
    print("Train:", train.shape, " Validation:", val.shape, " Test:", test.shape)

    # categorical: most frequent value of the training set
    for col in categorical_features:
        most_frequent = train[col].mode(dropna=True).iloc[0]
        for part in (train, val, test):
            part[col] = part[col].fillna(most_frequent)

    # numerical columns that are empty in the training set cannot be used
    numerical_features = [c for c in numerical_features_all if train[c].notna().any()]
    raw_train = train[numerical_features].copy()          # unscaled copy, for the outlier inspection
    for part in (train, val, test):
        part.drop(columns=[c for c in numerical_features_all if c not in numerical_features], inplace=True)

    # numerical: KNN imputer (k = 5), fitted on the training set
    print("KNN imputation ...")
    imputer = KNNImputer(n_neighbors=5)
    imputer.fit(train[numerical_features].to_numpy())
    for part in (train, val, test):
        values = part[numerical_features].to_numpy()
        # transformed in batches of 1,000 rows to keep memory use low
        part[numerical_features] = np.vstack(
            [imputer.transform(values[i:i + 1000]) for i in range(0, len(values), 1000)])

    scaler = StandardScaler()
    scaler.fit(train[numerical_features].to_numpy())
    for part in (train, val, test):
        part[numerical_features] = scaler.transform(part[numerical_features].to_numpy())

    if APPLY_DBSCAN:
        dbscan = DBSCAN(eps=9.3, min_samples=3, n_jobs=-1)
        keep = dbscan.fit_predict(train[numerical_features].to_numpy()) != -1
        print("DBSCAN (eps = 9.3, min_samples = 3) outliers removed:", int((~keep).sum()),
              f"of {len(train)} training rows ({100 * (~keep).mean():.2f} %)")

        if (~keep).any():
            inspect = pd.DataFrame({
                "median_removed": raw_train[~keep].median(),
                "median_kept": raw_train[keep].median(),
                "missing_removed_%": 100 * raw_train[~keep].isna().mean(),
                "missing_kept_%": 100 * raw_train[keep].isna().mean(),
            }).round(2)
            print("Removed vs kept rows (largest median differences):")
            gap = (inspect["median_removed"] - inspect["median_kept"]).abs()
            print(inspect.loc[gap.sort_values(ascending=False).index[:10]])
            print("Classes of removed rows (0..4 = -2..+2):",
                  pd.Series(y_train[~keep]).value_counts().sort_index().to_dict())
            inspect.to_csv(f"ashrae_split{run}_dbscan_inspection.csv")

        train, y_train = train[keep], y_train[keep]

    for col in categorical_features:
        categories = sorted(train[col].astype(str).unique())
        codes = {value: i for i, value in enumerate(categories)}
        for part in (train, val, test):
            part[col] = part[col].astype(str).map(codes).fillna(len(categories)).astype(int)

    # SMOTE on the training set only
    counts = pd.Series(y_train).value_counts().sort_index()
    target_count = int(min(counts[1], counts[2], counts[3]))
    sampling_strategy = {cls: max(target_count, int(counts[cls])) for cls in (0, 4)}

    smote = SMOTE(sampling_strategy=sampling_strategy, random_state=seed)
    train_bal, y_train_bal = smote.fit_resample(train, y_train)
    for col in categorical_features:
        n_codes = int(train[col].max())
        train_bal[col] = np.clip(np.rint(train_bal[col]), 0, n_codes).astype(int)
    n_synthetic = len(y_train_bal) - len(y_train)

    print("Training classes before SMOTE:", counts.tolist())
    print("Training classes after SMOTE: ", pd.Series(y_train_bal).value_counts().sort_index().tolist())
    print("Synthetic training rows:", n_synthetic)

    columns = contextual_features + [c for c in train_bal.columns if c not in contextual_features]
    tag = f"ashrae_split{run}" + ("" if APPLY_DBSCAN else "_noDBSCAN")

    out_train = train_bal[columns].copy()
    out_train["label"] = y_train_bal
    out_train["is_synthetic"] = np.r_[np.zeros(len(y_train), int), np.ones(n_synthetic, int)]
    out_val = val[columns].copy()
    out_val["label"] = y_val
    out_test = test[columns].copy()
    out_test["label"] = y_test

    out_train.to_csv(f"{tag}_train.csv", index=False)
    out_val.to_csv(f"{tag}_val.csv", index=False)
    out_test.to_csv(f"{tag}_test.csv", index=False)
    print(f"Saved: {tag}_train.csv, {tag}_val.csv, {tag}_test.csv")

print("\nPreprocessing completed.")




