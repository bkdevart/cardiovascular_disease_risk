"""
Project configuration and constants.
Central place for all paths, parameters, and feature definitions.
"""

import os

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
EDA_DIR = os.path.join(OUTPUT_DIR, "eda")
MODEL_DIR = os.path.join(OUTPUT_DIR, "models")
RESULTS_DIR = os.path.join(OUTPUT_DIR, "results")

RAW_DATA_PATH = os.path.join(DATA_DIR, "raw_cvd_data.csv")
PROCESSED_DATA_PATH = os.path.join(DATA_DIR, "processed_cvd_data.csv")

# Create directories
for d in [DATA_DIR, EDA_DIR, MODEL_DIR, RESULTS_DIR]:
    os.makedirs(d, exist_ok=True)

# ── Dataset ────────────────────────────────────────────────────────────────────
KAGGLE_DATASET = "alphiree/cardiovascular-diseases-risk-prediction-dataset"
TARGET_COL = "Heart_Disease"

# ── Feature Definitions ────────────────────────────────────────────────────────
BINARY_COLS = [
    "Exercise", "Skin_Cancer", "Other_Cancer", "Depression",
    "Arthritis", "Smoking_History",
]

ORDINAL_MAPPINGS = {
    "General_Health": {
        "Poor": 0, "Fair": 1, "Good": 2, "Very Good": 3, "Excellent": 4
    },
    "Checkup": {
        "Never": 0,
        "5 or more years ago": 1,
        "Within the past 5 years": 2,
        "Within the past 2 years": 3,
        "Within the past year": 4,
    },
    "Diabetes": {
        "No": 0,
        "No, pre-diabetes or borderline diabetes": 1,
        "Yes, but only during pregnancy (female)": 2,
        "Yes, but female told only during pregnancy": 2,
        "Yes": 3,
    },
    "Age_Category": {
        "18-24": 0, "25-29": 1, "30-34": 2, "35-39": 3,
        "40-44": 4, "45-49": 5, "50-54": 6, "55-59": 7,
        "60-64": 8, "65-69": 9, "70-74": 10, "75-79": 11,
        "80+": 12,
    },
}

NUMERICAL_COLS = [
    "Height_(cm)", "Weight_(kg)", "BMI",
    "Alcohol_Consumption", "Fruit_Consumption",
    "Green_Vegetables_Consumption", "FriedPotato_Consumption",
]

SEX_MAPPING = {"Female": 0, "Male": 1}

# ── Model Parameters ──────────────────────────────────────────────────────────
RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5
DT_MAX_DEPTH_RANGE = range(3, 16)

# ── Plot Style ─────────────────────────────────────────────────────────────────
PLOT_STYLE = "seaborn-v0_8-whitegrid"
FIG_DPI = 150
COLORS = {"No": "#2196F3", "Yes": "#F44336"}  # Blue for No CVD, Red for Yes CVD
