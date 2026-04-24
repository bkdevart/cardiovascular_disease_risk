"""
Step 4-6: Model Training & Tuning
Logistic Regression (baseline) and Decision Tree (advanced).
Includes SMOTE for class imbalance handling.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import cross_val_score, StratifiedKFold
from imblearn.over_sampling import SMOTE

from src.config import RANDOM_STATE, CV_FOLDS, DT_MAX_DEPTH_RANGE, MODEL_DIR
from src.utils import print_section, print_step, save_plot


def apply_smote(X_train, y_train):
    """Apply SMOTE to handle class imbalance on training data only."""
    print_step("Handling class imbalance with SMOTE")

    print(f"  Before SMOTE: {dict(pd.Series(y_train).value_counts())}")

    smote = SMOTE(random_state=RANDOM_STATE)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

    print(f"  After SMOTE:  {dict(pd.Series(y_resampled).value_counts())}")
    return X_resampled, y_resampled


def train_logistic_regression(X_train, y_train, feature_names):
    """
    Train baseline logistic regression model.
    Returns the trained model and coefficient analysis.
    """
    print_step("Training Logistic Regression (Baseline)")

    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        solver="lbfgs",
        random_state=RANDOM_STATE,
    )
    model.fit(X_train, y_train)

    # Coefficient analysis
    coef_df = pd.DataFrame({
        "Feature": feature_names,
        "Coefficient": model.coef_[0],
        "Abs_Coefficient": np.abs(model.coef_[0]),
    }).sort_values("Abs_Coefficient", ascending=False)

    print("\n  Top 10 features by coefficient magnitude:")
    for _, row in coef_df.head(10).iterrows():
        direction = "↑" if row["Coefficient"] > 0 else "↓"
        print(f"    {direction} {row['Feature']:<35} {row['Coefficient']:+.4f}")

    # Plot coefficients
    fig, ax = plt.subplots(figsize=(10, 8))
    top_n = min(15, len(coef_df))
    top = coef_df.head(top_n).sort_values("Coefficient")
    colors = ["#F44336" if c > 0 else "#2196F3" for c in top["Coefficient"]]
    ax.barh(top["Feature"], top["Coefficient"], color=colors, edgecolor="white")
    ax.set_title("Logistic Regression Coefficients (Top 15)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Coefficient Value", fontsize=12)
    ax.axvline(x=0, color="black", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    save_plot(fig, MODEL_DIR, "logistic_regression_coefficients.png")

    return model, coef_df


def find_best_depth(X_train, y_train):
    """Find optimal decision tree depth via cross-validation."""
    print_step("Tuning Decision Tree depth")

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    depth_scores = {}

    for depth in DT_MAX_DEPTH_RANGE:
        dt = DecisionTreeClassifier(
            max_depth=depth, class_weight="balanced", random_state=RANDOM_STATE
        )
        scores = cross_val_score(dt, X_train, y_train, cv=cv, scoring="roc_auc")
        depth_scores[depth] = scores.mean()
        print(f"    Depth {depth:>2}: AUC = {scores.mean():.4f} ± {scores.std():.4f}")

    best_depth = max(depth_scores, key=depth_scores.get)
    print(f"\n  Best depth: {best_depth} (AUC = {depth_scores[best_depth]:.4f})")

    # Plot depth vs AUC
    fig, ax = plt.subplots(figsize=(10, 5))
    depths = list(depth_scores.keys())
    aucs = list(depth_scores.values())
    ax.plot(depths, aucs, "o-", color="#1565C0", linewidth=2, markersize=8)
    ax.axvline(x=best_depth, color="#F44336", linestyle="--", label=f"Best depth = {best_depth}")
    ax.set_title("Decision Tree: Depth vs AUC (Cross-Validation)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Max Depth", fontsize=12)
    ax.set_ylabel("Mean AUC", fontsize=12)
    ax.legend(fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    save_plot(fig, MODEL_DIR, "decision_tree_depth_tuning.png")

    return best_depth


def train_decision_tree(X_train, y_train, feature_names, max_depth=None):
    """
    Train decision tree classifier with tuned depth.
    Returns the trained model and feature importance analysis.
    """
    print_step("Training Decision Tree (Advanced)")

    if max_depth is None:
        max_depth = find_best_depth(X_train, y_train)

    model = DecisionTreeClassifier(
        max_depth=max_depth,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )
    model.fit(X_train, y_train)

    # Feature importance
    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": model.feature_importances_,
    }).sort_values("Importance", ascending=False)

    print(f"\n  Tree depth: {model.get_depth()}, leaves: {model.get_n_leaves()}")
    print("\n  Top 10 features by importance:")
    for _, row in importance_df.head(10).iterrows():
        bar = "█" * int(row["Importance"] * 50)
        print(f"    {row['Feature']:<35} {row['Importance']:.4f} {bar}")

    # Plot feature importance
    fig, ax = plt.subplots(figsize=(10, 8))
    top_n = min(15, len(importance_df))
    top = importance_df.head(top_n).sort_values("Importance")
    ax.barh(top["Feature"], top["Importance"], color="#1565C0", edgecolor="white")
    ax.set_title("Decision Tree Feature Importance (Top 15)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Importance", fontsize=12)
    ax.spines[["top", "right"]].set_visible(False)
    save_plot(fig, MODEL_DIR, "decision_tree_feature_importance.png")

    # Plot tree structure (limited depth for readability)
    fig, ax = plt.subplots(figsize=(24, 10))
    plot_tree(
        model, max_depth=3, feature_names=feature_names,
        class_names=["No CVD", "CVD"], filled=True, rounded=True,
        fontsize=8, ax=ax
    )
    ax.set_title("Decision Tree Structure (Top 3 Levels)", fontsize=14, fontweight="bold")
    save_plot(fig, MODEL_DIR, "decision_tree_structure.png")

    return model, importance_df


def run(X_train=None, y_train=None, feature_names=None):
    """Run model training pipeline."""
    print_section("STEP 4-6: MODEL TRAINING & TUNING")

    if X_train is None:
        from src.data_loader import load_raw_data
        from src.preprocessing import run as preprocess
        X_train, _, y_train, _, _, feature_names = preprocess()

    # Apply SMOTE
    X_resampled, y_resampled = apply_smote(X_train, y_train)

    # Train models
    lr_model, lr_coefs = train_logistic_regression(X_resampled, y_resampled, feature_names)
    dt_model, dt_importance = train_decision_tree(X_resampled, y_resampled, feature_names)

    print(f"\n  ✓ Both models trained successfully")
    return lr_model, dt_model, lr_coefs, dt_importance


if __name__ == "__main__":
    run()
