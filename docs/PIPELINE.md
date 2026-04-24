# CVD Prediction Pipeline — Detailed Documentation

## Pipeline Overview

```
┌─────────────┐    ┌──────────────────┐    ┌─────────┐    ┌──────────────┐
│ 1. Data      │───▶│ 2. Pre-Processing│───▶│ 3. EDA  │───▶│ 4. Modeling  │
│ Acquisition  │    │ (4C Strategy)    │    │         │    │              │
└─────────────┘    └──────────────────┘    └─────────┘    └──────┬───────┘
                                                                  │
                   ┌──────────────┐    ┌──────────────┐           │
                   │ 6. Interpret │◀───│ 5. Evaluate  │◀──────────┘
                   │ & Conclude   │    │ & Validate   │
                   └──────────────┘    └──────────────┘
```

---

## Step 1: Data Acquisition (`src/data_loader.py`)

**Source:** 2021 CDC Behavioral Risk Factor Surveillance System (BRFSS) from Kaggle  
**Dataset:** `alphiree/cardiovascular-diseases-risk-prediction-dataset`

**What it does:**
- Downloads the dataset via kagglehub (or loads from local `data/` directory)
- Performs initial inspection: shape, dtypes, head, null counts
- Saves raw data to `data/raw_cvd_data.csv`

**Expected variables (19 total):**
| Variable | Type | Description |
|----------|------|-------------|
| General_Health | Categorical | Self-reported health status |
| Checkup | Categorical | Last routine checkup |
| Exercise | Binary | Physical activity in past 30 days |
| Heart_Disease | Binary (Target) | Yes/No CVD diagnosis |
| Skin_Cancer | Binary | Skin cancer history |
| Other_Cancer | Binary | Other cancer history |
| Depression | Binary | Depressive disorder |
| Diabetes | Categorical | Diabetes status |
| Arthritis | Binary | Arthritis diagnosis |
| Sex | Binary | Male/Female |
| Age_Category | Categorical | Age range bucket |
| Height_(cm) | Numerical | Height in centimeters |
| Weight_(kg) | Numerical | Weight in kilograms |
| BMI | Numerical | Body Mass Index |
| Smoking_History | Binary | Ever smoked 100+ cigarettes |
| Alcohol_Consumption | Numerical | Drinks per week |
| Fruit_Consumption | Numerical | Fruit servings per day |
| Green_Vegetables_Consumption | Numerical | Vegetable servings per day |
| FriedPotato_Consumption | Numerical | Fried potato servings per day |

---

## Step 2: Data Pre-Processing — The 4C Strategy (`src/preprocessing.py`)

### 2a. Correcting
- Check for and report any missing values
- Identify outliers in numerical columns (BMI, Height, Weight) using IQR method
- Cap extreme outliers rather than removing them (preserve sample size)

### 2b. Completing
- Impute any missing values (median for numerical, mode for categorical)
- Verify no nulls remain after imputation

### 2c. Converting
- **Ordinal encoding** for ordered categoricals:
  - `General_Health`: Poor=0, Fair=1, Good=2, Very Good=3, Excellent=4
  - `Age_Category`: Ordered by age range
  - `Checkup`: Ordered by recency
  - `Diabetes`: No=0, Pre-diabetes/borderline=1, Yes=2, Yes (during pregnancy)=3
- **Binary encoding** for Yes/No columns:
  - `Heart_Disease`, `Exercise`, `Skin_Cancer`, `Other_Cancer`, `Depression`, `Arthritis`, `Smoking_History` → 0/1
  - `Sex` → 0/1
- **Standard scaling** for continuous numerical features (BMI, Height, Weight, consumption vars)

### 2d. Creating
- Feature: `BMI_Category` (Underweight/Normal/Overweight/Obese) from BMI
- Feature: `Comorbidity_Count` — sum of binary disease indicators
- Train/test split: 80/20 stratified on target variable

---

## Step 3: Exploratory Data Analysis (`src/eda.py`)

**Visualizations produced:**
1. **Target distribution** — Bar chart of CVD Yes vs No (shows class imbalance)
2. **BMI distribution** — Histogram by CVD status
3. **Age category distribution** — Count plot by CVD status
4. **Correlation heatmap** — All numerical features (identifies multicollinearity)
5. **Feature distributions** — Box plots of key numerical predictors by CVD status
6. **Categorical feature analysis** — Stacked bar charts for General_Health, Smoking, Diabetes vs CVD

All plots saved to `outputs/eda/`.

---

## Step 4: Model Training (`src/models.py`)

### Handling Class Imbalance
- Apply **SMOTE** (Synthetic Minority Over-sampling Technique) to training data only
- This avoids data leakage from test set

### Baseline: Logistic Regression
- **Why:** Fundamental binary classifier with interpretable coefficients
- **Config:** `max_iter=1000`, `class_weight='balanced'`, `solver='lbfgs'`
- **Output:** Trained model + coefficient analysis (which features increase/decrease CVD risk)

### Advanced: Decision Tree Classifier
- **Why:** Captures non-linear relationships between health indicators
- **Config:** `max_depth` tuned via cross-validation (range 3-15), `class_weight='balanced'`
- **Output:** Trained model + feature importance ranking + tree visualization

---

## Step 5: Validation & Evaluation (`src/evaluation.py`)

### Cross-Validation
- **Stratified 5-fold CV** on training data for both models
- Reports mean ± std for accuracy, precision, recall, F1, AUC

### Test Set Evaluation
- **Confusion matrix** for each model
- **Classification report** (precision, recall, F1 per class)
- **ROC curves** — Both models on same plot with AUC values
- **Precision-Recall curves** — Important given class imbalance

### Model Comparison
- Side-by-side metric table (Accuracy, Precision, Recall, F1, AUC)
- Feature importance comparison between logistic regression coefficients and decision tree importances
- Statistical significance of performance difference

All plots saved to `outputs/models/` and `outputs/results/`.

---

## Step 6: Interpretation & Conclusions

### Key Questions Answered
1. Which model performs best for CVD prediction?
2. What are the strongest predictors of heart disease?
3. Do the results confirm initial expectations (age, diabetes, smoking, BMI)?
4. How could this model be deployed in a healthcare setting?

### Limitations to Acknowledge
- Self-reported survey data (potential reporting bias)
- Cross-sectional data (no causal inference)
- Limited to features available in BRFSS
- Class imbalance may affect minority class recall
