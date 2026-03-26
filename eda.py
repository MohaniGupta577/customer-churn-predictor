"""
Customer Churn Predictor — eda.py
Author  : Mohani Gupta | mohanigupta279@gmail.com
Purpose : Standalone EDA script — generates 6 insight charts
Usage   : python src/eda.py
"""

import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")
sns.set_theme(style="darkgrid", font_scale=1.05)
os.makedirs("outputs", exist_ok=True)

COLORS = ["#34d399", "#ef4444", "#4f8ef7", "#f59e0b", "#a78bfa"]


def run_eda(filepath: str = "data/raw/telco_churn.csv") -> None:

    df = pd.read_csv(filepath)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["Churn_bin"]    = (df["Churn"] == "Yes").astype(int)

    print(f"Dataset : {df.shape[0]} rows × {df.shape[1]} cols")
    print(f"Churn   : {df['Churn'].value_counts(normalize=True).to_dict()}")
    print(f"Nulls   : {df.isnull().sum().sum()} total null values")
    print(f"\nNumeric summary:\n{df.describe().T[['mean','std','min','max']].round(2)}")

    # ── Figure 1: Overview (2×3) ──────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Customer Churn — EDA Overview  |  Mohani Gupta",
                 fontsize=16, fontweight="bold", y=1.01)

    # 1a. Churn distribution
    vc = df["Churn"].value_counts()
    axes[0, 0].bar(vc.index, vc.values,
                   color=[COLORS[0], COLORS[1]], edgecolor="white", linewidth=1.5)
    for i, v in enumerate(vc.values):
        axes[0, 0].text(i, v + 20, f"{v}\n({v/len(df)*100:.1f}%)",
                        ha="center", fontsize=10, fontweight="bold")
    axes[0, 0].set_title("Churn Distribution", fontweight="bold")
    axes[0, 0].set_xlabel("Churn Status")
    axes[0, 0].set_ylabel("Count")

    # 1b. Tenure by churn (KDE)
    for label, color in zip(["No", "Yes"], [COLORS[0], COLORS[1]]):
        df[df["Churn"] == label]["tenure"].plot(
            kind="kde", ax=axes[0, 1], color=color, linewidth=2, label=label
        )
    axes[0, 1].set_title("Tenure Distribution by Churn", fontweight="bold")
    axes[0, 1].set_xlabel("Tenure (months)")
    axes[0, 1].legend(title="Churn")

    # 1c. Monthly charges box
    df.boxplot(column="MonthlyCharges", by="Churn",
               ax=axes[0, 2],
               boxprops=dict(color=COLORS[2]),
               medianprops=dict(color=COLORS[1], linewidth=2))
    axes[0, 2].set_title("Monthly Charges by Churn", fontweight="bold")
    axes[0, 2].set_xlabel("Churn Status")
    plt.sca(axes[0, 2]); plt.title("Monthly Charges by Churn")

    # 1d. Contract type
    ct = df.groupby(["Contract", "Churn"]).size().unstack(fill_value=0)
    ct.plot(kind="bar", ax=axes[1, 0],
            color=[COLORS[0], COLORS[1]], edgecolor="white")
    axes[1, 0].set_title("Churn by Contract Type", fontweight="bold")
    axes[1, 0].tick_params(rotation=20)
    axes[1, 0].legend(title="Churn")

    # 1e. Internet service
    isc = df.groupby(["InternetService", "Churn"]).size().unstack(fill_value=0)
    isc.plot(kind="bar", ax=axes[1, 1],
             color=[COLORS[0], COLORS[1]], edgecolor="white")
    axes[1, 1].set_title("Churn by Internet Service", fontweight="bold")
    axes[1, 1].tick_params(rotation=15)
    axes[1, 1].legend(title="Churn")

    # 1f. Correlation with churn
    num_df = df.select_dtypes("number")
    corr   = num_df.corr()[["Churn_bin"]].drop("Churn_bin").sort_values("Churn_bin")
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdYlGn",
                center=0, ax=axes[1, 2], cbar=False, linewidths=0.5)
    axes[1, 2].set_title("Numeric Correlation with Churn", fontweight="bold")

    plt.tight_layout()
    plt.savefig("outputs/01_eda_overview.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✓ outputs/01_eda_overview.png")

    # ── Figure 2: Churn rates by category ────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Churn Rates Across Key Segments", fontsize=14, fontweight="bold")

    for ax, col in zip(axes, ["SeniorCitizen", "Partner", "Dependents"]):
        rates = df.groupby(col)["Churn_bin"].mean() * 100
        ax.bar(rates.index.astype(str), rates.values,
               color=[COLORS[0], COLORS[1]], edgecolor="white")
        for i, v in enumerate(rates.values):
            ax.text(i, v + 0.5, f"{v:.1f}%",
                    ha="center", fontsize=11, fontweight="bold")
        ax.set_title(f"Churn Rate by {col}", fontweight="bold")
        ax.set_ylabel("Churn Rate (%)")
        ax.set_ylim(0, rates.max() * 1.2)

    plt.tight_layout()
    plt.savefig("outputs/02_segment_churn_rates.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✓ outputs/02_segment_churn_rates.png")

    print("\n✅ EDA complete — check outputs/ folder")


if __name__ == "__main__":
    run_eda()
