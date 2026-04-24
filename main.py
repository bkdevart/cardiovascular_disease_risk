"""
Cardiovascular Disease Risk Prediction — Full Pipeline
=======================================================
SIE 433/533 Fundamentals of Data Science for Engineers – Spring 2026
Team: Brandon Knox, Justin Larimore, Feruz, Michael Willey

Runs the complete pipeline:
  1. Data Acquisition
  2. Data Pre-Processing (4C Strategy)
  3. Exploratory Data Analysis
  4-6. Model Training & Tuning (Logistic Regression + Decision Tree)
  7-8. Model Evaluation & Comparison
"""

import warnings
import time

warnings.filterwarnings("ignore")


def main():
    start = time.time()

    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║   Predictive Modeling of Cardiovascular Disease Risk               ║")
    print("║   SIE 433/533 — Spring 2026 — University of Arizona               ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")

    # ── Step 1: Data Acquisition ───────────────────────────────────────────────
    from src.data_loader import run as load_data
    raw_df = load_data()

    # ── Step 3: EDA (on raw data before encoding) ──────────────────────────────
    from src.eda import run as run_eda
    run_eda(raw_df)

    # ── Step 2: Pre-Processing ─────────────────────────────────────────────────
    from src.preprocessing import run as preprocess
    X_train, X_test, y_train, y_test, scaler, feature_names = preprocess(raw_df)

    # ── Steps 4-6: Model Training ──────────────────────────────────────────────
    from src.models import run as train_models
    lr_model, dt_model, lr_coefs, dt_importance = train_models(
        X_train, y_train, feature_names
    )

    # ── Steps 7-8: Evaluation ──────────────────────────────────────────────────
    from src.evaluation import run as evaluate
    models = {
        "Logistic Regression": lr_model,
        "Decision Tree": dt_model,
    }
    cv_results, test_results, comparison_df = evaluate(
        models=models,
        X_train=X_train, X_test=X_test,
        y_train=y_train, y_test=y_test,
        lr_coefs=lr_coefs, dt_importance=dt_importance,
        feature_names=feature_names,
    )

    # ── Summary ────────────────────────────────────────────────────────────────
    elapsed = time.time() - start
    print("\n" + "=" * 70)
    print("  PIPELINE COMPLETE")
    print("=" * 70)
    print(f"\n  Total runtime: {elapsed:.1f} seconds")
    print(f"\n  Model Comparison (Test Set):")
    print(comparison_df.round(4).to_string())
    print(f"\n  Outputs saved to: outputs/")
    print(f"    - EDA plots:     outputs/eda/")
    print(f"    - Model plots:   outputs/models/")
    print(f"    - Results:       outputs/results/")
    print()


if __name__ == "__main__":
    main()
