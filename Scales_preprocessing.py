#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Lun Lug 07 14:53:11 2025

@author: Sundas Sarwar
"""

# %%

import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from imblearn.over_sampling import SMOTE

DATA_PATH = "./dataset/Scales_Dataset.csv"     

# "fixed"  = ONE fixed 70/10/20 partition (seed 42), as in the paper: the five runs then use the
#            same partition with a different training seed.
# "random" = five different random 70/10/20 partitions (seeds 42, 1, 2, 3, 4).
SPLIT_MODE = "fixed"
SPLIT_SEEDS = [42] if SPLIT_MODE == "fixed" else [42, 1, 2, 3, 4]

PARTICIPANT_ID = "unID"      # subject-wise split: no participant in more than one subset

# DBSCAN setting
APPLY_DBSCAN = True          
DBSCAN_EPS = 9.3
DBSCAN_MIN_SAMPLES = 3

ROLLING_WINDOW = 3           # observations in the rolling median / running average

# LOAD DATASET

df = pd.read_csv(DATA_PATH, encoding="ISO-8859-1", low_memory=False)
print("Original shape:", df.shape)

# Five contextual features used by HCA (same as the official code).
contextual_features = ["SurvCity", "SurvCountry", "Season", "SeasonDesc", "SensType"]

# DROP DUPLICATE ROWS AND MOSTLY-EMPTY COLUMNS

n_before = len(df)
df = df.drop_duplicates()
print("Duplicate rows removed:", n_before - len(df))

missing_share = df.isna().mean()
mostly_empty = [c for c in missing_share[missing_share > 0.5].index if c not in contextual_features]
df = df.drop(columns=mostly_empty)
print("Columns with > 50 % missing values removed:", len(mostly_empty))

# THERMAL SENSATION: 7 POINTS -> 5 POINTS

# StateTSVF is coded 1..7  ->  -3..+3  ->  5 points:
# -3, -2 -> -2 ;  -1 -> -1 ;  0 -> 0 ;  +1 -> +1 ;  +2, +3 -> +2
# Rows without a vote are removed.

TARGET = "StateTSVF"

df = df.dropna(subset=[TARGET]).copy()
tsv_5 = np.clip(np.round(df[TARGET] - 4), -2, 2)
df["label"] = (tsv_5 + 2).astype(int)          # class index 0..4  <->  -2..+2
df = df.drop(columns=[TARGET])

print("\nFive-point distribution (0..4 = -2..+2):")
print(df["label"].value_counts().sort_index().tolist())

# RULE-BASED PHYSICAL VALIDATION

# Implausible readings are set to missing (the row is kept and the
# value is filled later by KNN imputation).
physical_limits = {
    "TinAirSurv": (5, 45), "TinGlobeSurv": (5, 50), "TinRadSurv": (5, 50), "TinOpSurv": (5, 45),
    "Tin": (5, 45), "RHinSurv": (0, 100), "AvinSurv": (0, 5),
    "ToutSurv": (-40, 50), "ToutDay": (-40, 50), "ToutWeek": (-40, 50),
    "RHoutSurv": (0, 100), "RHoutDay": (0, 100), "RHoutWeek": (0, 100),
}

invalid_counts = {}
for col, (lower, upper) in physical_limits.items():
    if col in df.columns:
        invalid = df[col].notna() & ~df[col].between(lower, upper)
        if invalid.any():
            invalid_counts[col] = int(invalid.sum())
            df.loc[invalid, col] = np.nan

print("\nImplausible values set to missing:", invalid_counts)

# TEMPORAL CONTEXT FEATURES

timestamp = pd.to_datetime(df["Date"].astype(str) + " " + df["Time"].astype(str),
                           format="%m/%d/%Y %H:%M", errors="coerce")
df["hour_of_day"] = timestamp.dt.hour
df["month"] = timestamp.dt.month
df["weekday"] = timestamp.dt.weekday

# ROLLING-MEDIAN SMOOTHING + RUNNING AVERAGES

df["_time"] = timestamp
df = df.sort_values(["SurvCity", "_time"], kind="mergesort")
by_city = df.groupby("SurvCity", sort=False)

smoothed = [c for c in ["Tin", "TinAirSurv", "RHinSurv", "ToutSurv", "RHoutSurv"] if c in df.columns]
for col in smoothed:
    df[col] = by_city[col].transform(lambda s: s.rolling(ROLLING_WINDOW, min_periods=1).median())

for col in [c for c in ["Tin", "RHinSurv"] if c in df.columns]:
    df[col + "_running_avg"] = by_city[col].transform(
        lambda s: s.rolling(ROLLING_WINDOW, min_periods=1).mean())

print("Smoothed columns:", smoothed)

# ZONE-LEVEL DESCRIPTORS (MEAN, VARIANCE, MIN-MAX RANGE)

zone_cols = [c for c in ["TinAirSurv", "TinGlobeSurv", "TinRadSurv", "TinOpSurv"] if c in df.columns]
if len(zone_cols) >= 2:
    df["Tin_zone_mean"] = df[zone_cols].mean(axis=1)
    df["Tin_zone_var"] = df[zone_cols].var(axis=1, ddof=0)
    df["Tin_zone_range"] = df[zone_cols].max(axis=1) - df[zone_cols].min(axis=1)
    df = df.drop(columns=zone_cols)

# FEATURE TYPES

df = df.reset_index(drop=True)
y = df.pop("label").to_numpy()
groups = df[PARTICIPANT_ID].astype(str).to_numpy()      # participant of every row

# Identifiers, free text and raw date/time strings are not model inputs.
not_features = ["Index", "SurvID", "SurvID.1", "unID", "Date", "Time", "SurvDate",
                "SurvTimeStart", "SurvTimeEnd", "_time", "Comment", "SurvObs",
                "WeathStaCode", "WeathStaType", "Variable.name"]
df = df.drop(columns=[c for c in not_features if c in df.columns])

# Text columns, the context columns and the integer-coded questionnaire answers
# are categorical; all other columns are numerical.
coded_answers = ["Place", "Sex", "Age", "PeriodRes", "Native", "Adapt"]
categorical_features = [c for c in df.columns
                        if df[c].dtype == object or str(df[c].dtype).startswith("str")
                        or c in contextual_features or c in coded_answers]
numerical_features_all = [c for c in df.columns if c not in categorical_features]
for col in categorical_features:
    df[col] = df[col].map(lambda v: str(v).strip() if pd.notna(v) else np.nan).astype(object)
for col in numerical_features_all:
    df[col] = pd.to_numeric(df[col], errors="coerce")

print("\nRows:", len(df), "| participants:", len(set(groups)),
      "| numerical features:", len(numerical_features_all),
      "| categorical features:", len(categorical_features))

# SPLIT

for run, seed in enumerate(SPLIT_SEEDS, start=1):
    print(f"\n========== Split {run} (seed {seed}) ==========")

    # ---------------- subject-wise 70 % train, 10 % validation, 20 % test
    idx = np.arange(len(df))
    split1 = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=seed)
    train_idx, rest_idx = next(split1.split(idx, y, groups))
    split2 = GroupShuffleSplit(n_splits=1, test_size=2 / 3, random_state=seed)
    r_val, r_test = next(split2.split(rest_idx, y[rest_idx], groups[rest_idx]))
    val_idx, test_idx = rest_idx[r_val], rest_idx[r_test]

    # check: no participant appears in two subsets
    assert not set(groups[train_idx]) & set(groups[val_idx])
    assert not set(groups[train_idx]) & set(groups[test_idx])
    assert not set(groups[val_idx]) & set(groups[test_idx])

    train, val, test = df.iloc[train_idx].copy(), df.iloc[val_idx].copy(), df.iloc[test_idx].copy()
    y_train, y_val, y_test = y[train_idx], y[val_idx], y[test_idx]
    print("Train:", train.shape, " Validation:", val.shape, " Test:", test.shape)

    # missing values
    # categorical: most frequent value of the training set
    for col in categorical_features:
        if train[col].notna().any():
            most_frequent = train[col].mode(dropna=True).iloc[0]
        else:
            most_frequent = "Unknown"
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
        part[numerical_features] = imputer.transform(part[numerical_features].to_numpy())

    # standardization
    scaler = StandardScaler()
    scaler.fit(train[numerical_features].to_numpy())
    for part in (train, val, test):
        part[numerical_features] = scaler.transform(part[numerical_features].to_numpy())

    # DBSCAN outlier removal
    if APPLY_DBSCAN:
        dbscan = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES, n_jobs=-1)
        keep = dbscan.fit_predict(train[numerical_features].to_numpy()) != -1
        print(f"DBSCAN (eps = {DBSCAN_EPS}, min_samples = {DBSCAN_MIN_SAMPLES}) outliers removed:",
              int((~keep).sum()), f"of {len(train)} training rows ({100 * (~keep).mean():.2f} %)")

        if (~keep).any() and keep.any():
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
            inspect.to_csv(f"scales_split{run}_dbscan_inspection.csv")

        train, y_train = train[keep], y_train[keep]

    # encode categorical features
    for col in categorical_features:
        categories = sorted(train[col].astype(str).unique())
        codes = {value: i for i, value in enumerate(categories)}
        for part in (train, val, test):
            part[col] = part[col].astype(str).map(codes).fillna(len(categories)).astype(int)

    # SMOTE on the training set only
    counts = pd.Series(y_train).value_counts().sort_index()
    target_count = int(min(counts[1], counts[2], counts[3]))
    sampling_strategy = {cls: max(target_count, int(counts[cls])) for cls in (0, 4)}

    k = min(5, int(counts[0]) - 1, int(counts[4]) - 1)    
    smote = SMOTE(sampling_strategy=sampling_strategy, k_neighbors=k, random_state=seed)
    train_bal, y_train_bal = smote.fit_resample(train, y_train)
    for col in categorical_features:
        n_codes = int(train[col].max())
        train_bal[col] = np.clip(np.rint(train_bal[col]), 0, n_codes).astype(int)
    n_synthetic = len(y_train_bal) - len(y_train)

    print("Training classes before SMOTE:", counts.tolist())
    print("Training classes after SMOTE: ", pd.Series(y_train_bal).value_counts().sort_index().tolist())
    print("Synthetic training rows:", n_synthetic)

    # ---------------- save (context columns first, then the other features, then the label)
    columns = contextual_features + [c for c in train_bal.columns if c not in contextual_features]
    tag = f"scales_split{run}" + ("" if APPLY_DBSCAN else "_noDBSCAN")

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
