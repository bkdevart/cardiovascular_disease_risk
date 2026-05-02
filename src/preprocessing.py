"""
Step 2: Data Pre-Processing — The 4C Strategy
Correcting, Completing, Converting, Creating.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.config import (
    TARGET_COL, BINARY_COLS, ORDINAL_MAPPINGS, NUMERICAL_COLS,
    SEX_MAPPING, RANDOM_STATE, TEST_SIZE, PROCESSED_DATA_PATH,
)
from src.utils import print_section, print_step


def correct_data(df):
    """
    2a. Correcting — Identify and handle outliers.
    Uses IQR method to cap extreme values in numerical columns.
    Lower bound is floored at 0 for variables that cannot be negative.
    """
    print_step("Correcting: Handling outliers")
    df = df.copy()

    for col in NUMERICAL_COLS:
        if col not in df.columns:
            continue
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        # Floor at 0 — these variables cannot be negative
        lower = max(lower, 0)

        n_outliers = ((df[col] < lower) | (df[col] > upper)).sum()
        if n_outliers > 0:
            print(f"  {col}: {n_outliers:,} outliers capped to [{lower:.1f}, {upper:.1f}]")
            df[col] = df[col].clip(lower, upper)

    return df


def complete_data(df):
    """
    2b. Completing — Handle missing values.
    Median for numerical, mode for categorical.
    """
    print_step("Completing: Handling missing values")
    df = df.copy()

    total_missing = df.isnull().sum().sum()
    if total_missing == 0:
        print("  No missing values found — data is complete.")
        return df

    for col in df.columns:
        n_missing = df[col].isnull().sum()
        if n_missing > 0:
            if df[col].dtype in ["float64", "int64"]:
                fill_val = df[col].median()
                df[col].fillna(fill_val, inplace=True)
                print(f"  {col}: {n_missing} nulls filled with median ({fill_val:.2f})")
            else:
                fill_val = df[col].mode()[0]
                df[col].fillna(fill_val, inplace=True)
                print(f"  {col}: {n_missing} nulls filled with mode ({fill_val})")

    print(f"  Remaining nulls: {df.isnull().sum().sum()}")
    return df


def convert_data(df):
    """
    2c. Converting — Encode categorical variables.
    Ordinal encoding for ordered categories, binary encoding for yes/no.
    """
    print_step("Converting: Encoding categorical variables")
    df = df.copy()

    # Encode target variable
    df[TARGET_COL] = df[TARGET_COL].map({"No": 0, "Yes": 1})
    print(f"  {TARGET_COL}: Yes/No → 1/0")

    # Encode binary columns
    for col in BINARY_COLS:
        if col in df.columns:
            df[col] = df[col].map({"No": 0, "Yes": 1})
            print(f"  {col}: Yes/No → 1/0")

    # Encode Sex
    if "Sex" in df.columns:
        df["Sex"] = df["Sex"].map(SEX_MAPPING)
        print(f"  Sex: Female/Male → 0/1")

    # Ordinal encoding
    for col, mapping in ORDINAL_MAPPINGS.items():
        if col in df.columns:
            unmapped = set(df[col].unique()) - set(mapping.keys())
            if unmapped:
                print(f"  WARNING: {col} has unmapped values: {unmapped}")
            df[col] = df[col].map(mapping)
            n_unmapped = df[col].isnull().sum()
            if n_unmapped > 0:
                df[col].fillna(df[col].mode()[0], inplace=True)
            print(f"  {col}: ordinal encoded ({len(mapping)} levels)")

    return df


def create_features(df):
    """
    2d. Creating — Drop redundant body-size columns to prevent multicollinearity.
    Height and Weight are dropped because BMI already captures body-size information.
    No new features are engineered so logistic regression coefficients remain
    clean, interpretable log-odds ratios without VIF inflation.
    """
    print_step("Creating: Dropping redundant body-size columns")
    df = df.copy()

    cols_to_drop = [c for c in ["Height_(cm)", "Weight_(kg)"] if c in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
        print(f"  Dropped: {cols_to_drop}")

    return df


def split_and_scale(df):
    """
    Split into train/test and scale ALL features with StandardScaler.
    Scaling all features ensures logistic regression coefficients are
    comparable across features regardless of their original range.
    """
    print_step("Splitting and scaling data")

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    feature_names = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    print(f"  Train: {X_train.shape[0]:,} samples")
    print(f"  Test:  {X_test.shape[0]:,} samples")
    print(f"  Target distribution (train): {y_train.value_counts(normalize=True).to_dict()}")

    # Scale ALL features for comparable LR coefficients
    scaler = StandardScaler()
    X_train = pd.DataFrame(
        scaler.fit_transform(X_train), columns=feature_names, index=X_train.index
    )
    X_test = pd.DataFrame(
        scaler.transform(X_test), columns=feature_names, index=X_test.index
    )
    print(f"  Scaled all {len(feature_names)} features with StandardScaler")

    return X_train, X_test, y_train, y_test, scaler, feature_names


def run(df=None):
    """Run the full preprocessing pipeline."""
    print_section("STEP 2: DATA PRE-PROCESSING (4C Strategy)")

    if df is None:
        from src.data_loader import load_raw_data
        df = load_raw_data()

    df = correct_data(df)
    df = complete_data(df)
    df = convert_data(df)
    df = create_features(df)

    # Save processed data
    df.to_csv(PROCESSED_DATA_PATH, index=False)
    print(f"\n  Saved processed data to: {PROCESSED_DATA_PATH}")

    # Split and scale
    X_train, X_test, y_train, y_test, scaler, feature_names = split_and_scale(df)

    print(f"\n  ✓ Preprocessing complete: {len(feature_names)} features")
    return X_train, X_test, y_train, y_test, scaler, feature_names


if __name__ == "__main__":
    run()
