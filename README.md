# Predictive Modeling of Cardiovascular Disease Risk

**Course:** SIE 433/533 Fundamentals of Data Science for Engineers – Spring 2026  
**Team:** Brandon Knox, Justin Larimore, Feruz, Michael Willey  
**University of Arizona**

---

## Project Overview

This project develops predictive models to classify the presence of cardiovascular disease (CVD) using
demographic, behavioral, and health-related characteristics from the 2021 CDC Behavioral Risk Factor
Surveillance System (BRFSS) dataset.

## Objective

Apply supervised classification techniques to predict CVD risk:
- Implement a **baseline logistic regression** model
- Compare with **decision tree** models
- Use **cross-validation** to assess generalization and prevent overfitting

## Dataset

- **Source:** 2021 CDC BRFSS dataset from Kaggle
- **Link:** https://www.kaggle.com/datasets/alphiree/cardiovascular-diseases-risk-prediction-dataset
- **Size:** 19 variables (12 numerical, 7 categorical)
- **Target:** Binary (Yes/No for CVD)
- **Features:** Age, sex, BMI, smoking history, alcohol consumption, exercise habits, comorbid conditions

## Project Structure

```
├── README.md                  # This file
├── docs/
│   └── PIPELINE.md            # Detailed pipeline documentation
├── data/                      # Dataset directory (auto-downloaded)
├── outputs/                   # Generated plots and results
│   ├── eda/                   # EDA visualizations
│   ├── models/                # Model performance plots
│   └── results/               # Final comparison tables
├── src/
│   ├── config.py              # Project configuration & constants
│   ├── data_loader.py         # Step 1: Data acquisition
│   ├── preprocessing.py       # Step 2: Data pre-processing (4C Strategy)
│   ├── eda.py                 # Step 3: Exploratory data analysis
│   ├── models.py              # Step 4-6: Model training & tuning
│   ├── evaluation.py          # Step 7-8: Validation & evaluation
│   └── utils.py               # Shared utilities
├── main.py                    # Full pipeline runner
└── .env/                      # Python virtual environment
```

## Quick Start

```bash
# Activate virtual environment
source .env/bin/activate

# Run the full pipeline
python main.py

# Or run individual steps
python -m src.data_loader       # Download & inspect data
python -m src.preprocessing     # Clean & encode data
python -m src.eda               # Generate EDA visualizations
python -m src.models            # Train models
python -m src.evaluation        # Evaluate & compare models
```

## Key Challenges Addressed

1. **Class Imbalance** — SMOTE resampling on training data
2. **Categorical Encoding** — Label encoding for ordinal, one-hot for nominal variables
3. **Multicollinearity** — Correlation analysis to identify redundant features
4. **Overfitting** — Stratified k-fold cross-validation, decision tree depth limiting
