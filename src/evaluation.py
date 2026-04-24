"""
Step 7-8: Model Evaluation & Comparison
Cross-validation, ROC curves, confusion matrices, and final comparison.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve,
    confusion_matrix, classification_report, average_precision_score,
)
from sklearn.model_selection import cross_validate, StratifiedKFold

from src.config import RANDOM_STATE, CV_FOLDS, MODEL_DIR, RESULTS_DIR, COLORS
from src.utils import print_section, print_step, save_plot


def cross_validate_models(models, X_train, y_train):
    """Run stratified k-fold cross-validation on all models."""
    print_step(f"Stratified {CV_FOLDS}-Fold Cross-Validation")

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]

    cv_results = {}
    for name, model in models.items():
        results = cross_validate(model, X_train, y_train, cv=cv, scoring=scoring)
        cv_results[name] = {
            metric: (results[f"test_{metric}"].mean(), results[f"test_{metric}"].std())
            for metric in scoring
        }

        print(f"\n  {name}:")
        for metric in scoring:
            mean, std = cv_results[name][metric]
            print(f"    {metric:<12}: {mean:.4f} ± {std:.4f}")

    return cv_results


def evaluate_on_test(models, X_test, y_test):
    """Evaluate all models on the held-out test set."""
    print_step("Test Set Evaluation")

    test_results = {}
    predictions = {}

    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        metrics = {
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, zero_division=0),
            "Recall": recall_score(y_test, y_pred, zero_division=0),
            "F1": f1_score(y_test, y_pred, zero_division=0),
            "AUC": roc_auc_score(y_test, y_prob),
            "Avg_Precision": average_precision_score(y_test, y_prob),
        }
        test_results[name] = metrics
        predictions[name] = {"y_pred": y_pred, "y_prob": y_prob}

        print(f"\n  {name}:")
        for metric, value in metrics.items():
            print(f"    {metric:<15}: {value:.4f}")

        print(f"\n  Classification Report:")
        report = classification_report(y_test, y_pred, target_names=["No CVD", "CVD"])
        for line in report.split("\n"):
            print(f"    {line}")

    return test_results, predictions


def plot_confusion_matrices(models, predictions, y_test):
    """Plot confusion matrices side by side."""
    print_step("Confusion Matrices")

    n_models = len(models)
    fig, axes = plt.subplots(1, n_models, figsize=(7 * n_models, 5))
    if n_models == 1:
        axes = [axes]

    for i, (name, preds) in enumerate(predictions.items()):
        cm = confusion_matrix(y_test, preds["y_pred"])
        sns.heatmap(
            cm, annot=True, fmt=",d", cmap="Blues",
            xticklabels=["No CVD", "CVD"],
            yticklabels=["No CVD", "CVD"],
            ax=axes[i], annot_kws={"size": 14}
        )
        axes[i].set_title(f"{name}\nConfusion Matrix", fontsize=13, fontweight="bold")
        axes[i].set_xlabel("Predicted", fontsize=11)
        axes[i].set_ylabel("Actual", fontsize=11)

    fig.tight_layout()
    save_plot(fig, MODEL_DIR, "confusion_matrices.png")


def plot_roc_curves(predictions, y_test):
    """Plot ROC curves for all models on the same axes."""
    print_step("ROC Curves")

    fig, ax = plt.subplots(figsize=(8, 7))
    model_colors = {"Logistic Regression": "#1565C0", "Decision Tree": "#F44336"}

    for name, preds in predictions.items():
        fpr, tpr, _ = roc_curve(y_test, preds["y_prob"])
        auc = roc_auc_score(y_test, preds["y_prob"])
        color = model_colors.get(name, "#666")
        ax.plot(fpr, tpr, label=f"{name} (AUC = {auc:.4f})", color=color, linewidth=2)

    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random (AUC = 0.5)")
    ax.set_title("ROC Curve Comparison", fontsize=14, fontweight="bold")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.legend(fontsize=11, loc="lower right")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.spines[["top", "right"]].set_visible(False)

    save_plot(fig, MODEL_DIR, "roc_curves.png")


def plot_precision_recall_curves(predictions, y_test):
    """Plot precision-recall curves — important for imbalanced data."""
    print_step("Precision-Recall Curves")

    fig, ax = plt.subplots(figsize=(8, 7))
    model_colors = {"Logistic Regression": "#1565C0", "Decision Tree": "#F44336"}

    for name, preds in predictions.items():
        precision, recall, _ = precision_recall_curve(y_test, preds["y_prob"])
        ap = average_precision_score(y_test, preds["y_prob"])
        color = model_colors.get(name, "#666")
        ax.plot(recall, precision, label=f"{name} (AP = {ap:.4f})", color=color, linewidth=2)

    # Baseline: prevalence
    prevalence = y_test.mean()
    ax.axhline(y=prevalence, color="gray", linestyle="--", alpha=0.5,
               label=f"Baseline (prevalence = {prevalence:.3f})")

    ax.set_title("Precision-Recall Curve Comparison", fontsize=14, fontweight="bold")
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.legend(fontsize=11, loc="upper right")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.spines[["top", "right"]].set_visible(False)

    save_plot(fig, MODEL_DIR, "precision_recall_curves.png")


def plot_model_comparison(test_results):
    """Side-by-side bar chart comparing model metrics."""
    print_step("Model Comparison Summary")

    metrics = ["Accuracy", "Precision", "Recall", "F1", "AUC"]
    df = pd.DataFrame(test_results).T[metrics]

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(metrics))
    width = 0.35
    model_colors = ["#1565C0", "#F44336"]

    for i, (name, row) in enumerate(df.iterrows()):
        bars = ax.bar(x + i * width, row.values, width, label=name,
                      color=model_colors[i], edgecolor="white", linewidth=1)
        for bar, val in zip(bars, row.values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_title("Model Performance Comparison", fontsize=14, fontweight="bold")
    ax.set_ylabel("Score", fontsize=12)
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(metrics, fontsize=12)
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)

    save_plot(fig, RESULTS_DIR, "model_comparison.png")

    # Save comparison table
    table_path = f"{RESULTS_DIR}/model_comparison.csv"
    df.round(4).to_csv(table_path)
    print(f"  Saved comparison table to: {table_path}")

    return df


def compare_feature_importance(lr_coefs, dt_importance, feature_names):
    """Compare what each model considers important."""
    print_step("Feature Importance Comparison")

    # Normalize logistic regression coefficients to [0, 1] for comparison
    lr_norm = lr_coefs.copy()
    lr_norm["Normalized"] = lr_norm["Abs_Coefficient"] / lr_norm["Abs_Coefficient"].max()
    lr_top = set(lr_norm.head(10)["Feature"].values)

    dt_norm = dt_importance.copy()
    dt_norm["Normalized"] = dt_norm["Importance"] / dt_norm["Importance"].max()
    dt_top = set(dt_norm.head(10)["Feature"].values)

    overlap = lr_top & dt_top
    print(f"\n  Top 10 features overlap: {len(overlap)}/{10}")
    print(f"  Shared top features: {', '.join(sorted(overlap))}")

    # Side-by-side plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # LR coefficients
    top_lr = lr_norm.head(10).sort_values("Normalized")
    colors_lr = ["#F44336" if c > 0 else "#2196F3" for c in top_lr["Coefficient"]]
    ax1.barh(top_lr["Feature"], top_lr["Normalized"], color=colors_lr, edgecolor="white")
    ax1.set_title("Logistic Regression\n(Normalized |Coefficient|)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Normalized Importance", fontsize=11)
    ax1.spines[["top", "right"]].set_visible(False)

    # DT importance
    top_dt = dt_norm.head(10).sort_values("Normalized")
    ax2.barh(top_dt["Feature"], top_dt["Normalized"], color="#1565C0", edgecolor="white")
    ax2.set_title("Decision Tree\n(Feature Importance)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Normalized Importance", fontsize=11)
    ax2.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Feature Importance: Model Comparison", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    save_plot(fig, RESULTS_DIR, "feature_importance_comparison.png")


def run(models=None, X_train=None, X_test=None, y_train=None, y_test=None,
        lr_coefs=None, dt_importance=None, feature_names=None):
    """Run the full evaluation pipeline."""
    print_section("STEP 7-8: MODEL EVALUATION & COMPARISON")

    if models is None:
        raise ValueError("Models must be provided. Run the full pipeline via main.py.")

    # Cross-validation
    cv_results = cross_validate_models(models, X_train, y_train)

    # Test set evaluation
    test_results, predictions = evaluate_on_test(models, X_test, y_test)

    # Visualizations
    plot_confusion_matrices(models, predictions, y_test)
    plot_roc_curves(predictions, y_test)
    plot_precision_recall_curves(predictions, y_test)
    comparison_df = plot_model_comparison(test_results)

    # Feature importance comparison
    if lr_coefs is not None and dt_importance is not None:
        compare_feature_importance(lr_coefs, dt_importance, feature_names)

    # Final verdict
    print_step("FINAL VERDICT")
    best_model = comparison_df["AUC"].idxmax()
    best_auc = comparison_df.loc[best_model, "AUC"]
    print(f"\n  Best model by AUC: {best_model} ({best_auc:.4f})")

    # Check initial expectations
    if lr_coefs is not None:
        expected = {"Age_Category", "Diabetes", "Smoking_History", "BMI"}
        top_features = set(lr_coefs.head(10)["Feature"].values)
        confirmed = expected & top_features
        missed = expected - top_features
        print(f"\n  Expected key predictors confirmed: {', '.join(sorted(confirmed)) or 'None'}")
        if missed:
            print(f"  Expected but not in top 10: {', '.join(sorted(missed))}")

    print(f"\n  ✓ Evaluation complete — results saved to {RESULTS_DIR}")
    return cv_results, test_results, comparison_df


if __name__ == "__main__":
    print("Run via main.py for full pipeline evaluation.")
