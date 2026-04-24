"""
Step 3: Exploratory Data Analysis (EDA)
Generates visualizations to understand the dataset before modeling.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import (
    TARGET_COL, EDA_DIR, PLOT_STYLE, COLORS, NUMERICAL_COLS,
)
from src.utils import print_section, print_step, save_plot


def plot_target_distribution(df):
    """Bar chart showing CVD Yes vs No — highlights class imbalance."""
    print_step("Target variable distribution")

    counts = df[TARGET_COL].value_counts()
    pcts = df[TARGET_COL].value_counts(normalize=True) * 100

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(
        counts.index, counts.values,
        color=[COLORS.get(k, "#999") for k in counts.index],
        edgecolor="white", linewidth=1.5
    )

    for bar, count, pct in zip(bars, counts.values, pcts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + counts.max() * 0.01,
            f"{count:,}\n({pct:.1f}%)", ha="center", va="bottom", fontsize=12, fontweight="bold"
        )

    ax.set_title("Cardiovascular Disease Distribution", fontsize=14, fontweight="bold")
    ax.set_xlabel("Heart Disease", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_ylim(0, counts.max() * 1.15)
    ax.spines[["top", "right"]].set_visible(False)

    save_plot(fig, EDA_DIR, "01_target_distribution.png")
    print(f"  Class imbalance ratio: {counts.min() / counts.max():.3f}")


def plot_bmi_distribution(df):
    """Histogram of BMI by CVD status."""
    print_step("BMI distribution by CVD status")

    fig, ax = plt.subplots(figsize=(10, 5))
    for label, color in COLORS.items():
        subset = df[df[TARGET_COL] == label]["BMI"]
        ax.hist(subset, bins=50, alpha=0.6, label=f"CVD={label}", color=color, edgecolor="white")

    ax.set_title("BMI Distribution by CVD Status", fontsize=14, fontweight="bold")
    ax.set_xlabel("BMI", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.legend(fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)

    save_plot(fig, EDA_DIR, "02_bmi_distribution.png")


def plot_age_distribution(df):
    """Count plot of age categories by CVD status."""
    print_step("Age category distribution by CVD status")

    fig, ax = plt.subplots(figsize=(12, 5))
    age_order = [
        "18-24", "25-29", "30-34", "35-39", "40-44", "45-49",
        "50-54", "55-59", "60-64", "65-69", "70-74", "75-79", "80+"
    ]
    available_ages = [a for a in age_order if a in df["Age_Category"].unique()]

    sns.countplot(
        data=df, x="Age_Category", hue=TARGET_COL,
        order=available_ages, palette=COLORS, ax=ax, edgecolor="white"
    )

    ax.set_title("Age Category Distribution by CVD Status", fontsize=14, fontweight="bold")
    ax.set_xlabel("Age Category", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.tick_params(axis="x", rotation=45)
    ax.legend(title="Heart Disease", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)

    save_plot(fig, EDA_DIR, "03_age_distribution.png")


def plot_correlation_heatmap(df):
    """Correlation heatmap of all numerical features."""
    print_step("Correlation heatmap")

    # Select numerical columns for correlation
    num_df = df.select_dtypes(include=[np.number])

    corr = num_df.corr()

    fig, ax = plt.subplots(figsize=(14, 10))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
        center=0, vmin=-1, vmax=1, square=True, linewidths=0.5,
        ax=ax, annot_kws={"size": 8}
    )
    ax.set_title("Feature Correlation Heatmap", fontsize=14, fontweight="bold")

    save_plot(fig, EDA_DIR, "04_correlation_heatmap.png")

    # Report high correlations
    high_corr = []
    for i in range(len(corr.columns)):
        for j in range(i + 1, len(corr.columns)):
            if abs(corr.iloc[i, j]) > 0.5:
                high_corr.append((corr.columns[i], corr.columns[j], corr.iloc[i, j]))

    if high_corr:
        print("  High correlations (|r| > 0.5):")
        for c1, c2, r in sorted(high_corr, key=lambda x: abs(x[2]), reverse=True):
            print(f"    {c1} ↔ {c2}: {r:.3f}")


def plot_numerical_boxplots(df):
    """Box plots of key numerical features by CVD status."""
    print_step("Numerical feature box plots")

    plot_cols = [c for c in NUMERICAL_COLS if c in df.columns]
    n_cols = min(len(plot_cols), 4)
    n_rows = (len(plot_cols) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes = np.array(axes).flatten()

    for i, col in enumerate(plot_cols):
        sns.boxplot(
            data=df, x=TARGET_COL, y=col, palette=COLORS,
            ax=axes[i], fliersize=2
        )
        axes[i].set_title(col, fontsize=11, fontweight="bold")
        axes[i].spines[["top", "right"]].set_visible(False)

    # Hide unused axes
    for j in range(len(plot_cols), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Numerical Features by CVD Status", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    save_plot(fig, EDA_DIR, "05_numerical_boxplots.png")


def plot_categorical_analysis(df):
    """Stacked bar charts for key categorical features vs CVD."""
    print_step("Categorical feature analysis")

    cat_cols = ["General_Health", "Smoking_History", "Diabetes", "Sex", "Exercise"]
    available = [c for c in cat_cols if c in df.columns]

    fig, axes = plt.subplots(1, len(available), figsize=(5 * len(available), 5))
    if len(available) == 1:
        axes = [axes]

    for i, col in enumerate(available):
        ct = pd.crosstab(df[col], df[TARGET_COL], normalize="index") * 100
        ct.plot(kind="bar", stacked=True, color=[COLORS["No"], COLORS["Yes"]],
                ax=axes[i], edgecolor="white", linewidth=0.5)
        axes[i].set_title(f"{col} vs CVD", fontsize=11, fontweight="bold")
        axes[i].set_ylabel("Percentage (%)", fontsize=10)
        axes[i].set_xlabel(col, fontsize=10)
        axes[i].legend(title="CVD", labels=["No", "Yes"], fontsize=9)
        axes[i].tick_params(axis="x", rotation=45)
        axes[i].spines[["top", "right"]].set_visible(False)

    fig.suptitle("Categorical Features vs CVD Status", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    save_plot(fig, EDA_DIR, "06_categorical_analysis.png")


def run(df=None):
    """Run all EDA visualizations."""
    print_section("STEP 3: EXPLORATORY DATA ANALYSIS")

    if df is None:
        from src.data_loader import load_raw_data
        df = load_raw_data()

    plt.style.use(PLOT_STYLE)

    plot_target_distribution(df)
    plot_bmi_distribution(df)
    plot_age_distribution(df)
    plot_correlation_heatmap(df)
    plot_numerical_boxplots(df)
    plot_categorical_analysis(df)

    print(f"\n  ✓ EDA complete — all plots saved to {EDA_DIR}")


if __name__ == "__main__":
    run()
