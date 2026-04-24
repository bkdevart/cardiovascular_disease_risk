"""
Step 1: Data Acquisition
Downloads the CVD dataset from Kaggle and performs initial inspection.
"""

import os
import pandas as pd
from src.config import KAGGLE_DATASET, DATA_DIR, RAW_DATA_PATH
from src.utils import print_section, print_step


def download_dataset():
    """Download dataset from Kaggle using kagglehub."""
    print_step("Downloading dataset from Kaggle")

    if os.path.exists(RAW_DATA_PATH):
        print(f"  Dataset already exists at {RAW_DATA_PATH}")
        return RAW_DATA_PATH

    try:
        import kagglehub
        path = kagglehub.dataset_download(KAGGLE_DATASET)
        print(f"  Downloaded to: {path}")

        # Find the CSV file in the downloaded directory
        for root, _, files in os.walk(path):
            for f in files:
                if f.endswith(".csv"):
                    src = os.path.join(root, f)
                    # Copy to our data directory
                    df = pd.read_csv(src)
                    df.to_csv(RAW_DATA_PATH, index=False)
                    print(f"  Saved raw data to: {RAW_DATA_PATH}")
                    return RAW_DATA_PATH

        raise FileNotFoundError("No CSV file found in downloaded dataset")

    except Exception as e:
        print(f"  Kaggle download failed: {e}")
        print("  Attempting direct download...")
        # Fallback: try to load from data directory
        for f in os.listdir(DATA_DIR):
            if f.endswith(".csv"):
                return os.path.join(DATA_DIR, f)
        raise


def load_raw_data():
    """Load the raw dataset."""
    path = download_dataset()
    df = pd.read_csv(path)
    return df


def inspect_data(df):
    """Print a comprehensive inspection of the raw data."""
    print_step("Data Inspection")

    print(f"\n  Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")

    print(f"\n  Columns and dtypes:")
    for col in df.columns:
        null_count = df[col].isnull().sum()
        null_str = f" ({null_count} nulls)" if null_count > 0 else ""
        print(f"    {col:<35} {str(df[col].dtype):<10} {df[col].nunique():>6} unique{null_str}")

    print(f"\n  Missing values total: {df.isnull().sum().sum()}")

    print(f"\n  First 5 rows:")
    print(df.head().to_string(index=False))

    print(f"\n  Numerical summary:")
    print(df.describe().round(2).to_string())

    return df


def run():
    """Run the data acquisition step."""
    print_section("STEP 1: DATA ACQUISITION")
    df = load_raw_data()
    inspect_data(df)
    print(f"\n  ✓ Data loaded successfully: {df.shape[0]:,} samples, {df.shape[1]} features")
    return df


if __name__ == "__main__":
    run()
