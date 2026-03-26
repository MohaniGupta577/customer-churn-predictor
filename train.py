# ================================================================
# Customer Churn Predictor — train.py
# Author  : Mohani Gupta | mohanigupta279@gmail.com
# Purpose : Train, compare & save ML classification models
# ================================================================

import os
import pickle
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, roc_auc_score, roc_curve
)

warnings.filterwarnings("ignore")
os.makedirs("outputs", exist_ok=True)


# ── STEP 1: LOAD & PREPROCESS ──────────────────────────────────

def preprocess(filepath: str):
    df = pd.read_csv(filepath)
    print(f"Dataset shape: {df.shape}")
    print(f"Churn rate   : {df['Churn'].value_counts(normalize=True).to_dict()}")

    # Fix TotalCharges (spaces for new customers → NaN)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

    # Drop non-predictive ID
    df.drop("customerID", axis=1, errors="ignore", inplace=True)

    # Encode target
    df["Churn"] = (df["Churn"] == "Yes").astype(int)

    # Binary columns
    df["gender"]     = (df["gender"] == "Male").astype(int)
    df["Partner"]    = (df["Partner"] == "Yes").astype(int)
    df["Dependents"] = (df["Dependents"] == "Yes").astype(int)
    df["PhoneService"] = (df["PhoneService"] == "Yes").astype(int)
    df["PaperlessBilling"] = (df["PaperlessBilling"] == "Yes").astype(int)

    # One-hot encode multi-class categoricals
    ohe_cols = [
        "InternetService", "Contract", "PaymentMethod",
        "MultipleLines", "OnlineSecurity", "OnlineBackup",
        "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"
    ]
    df = pd.get_dummies(df, columns=[c for c in ohe_cols if c in df.columns],
                        drop_first=True)

    # Feature engineering
    df["tenure_group"] = pd.cut(
        df["tenure"],
        bins=[0, 12, 24, 48, 72],
        labels=[0, 1, 2, 3]
    ).astype(int)
    df["charges_per_month"] = df["TotalCharges"] / (df["tenure"] + 1)
    df["high_value"]        = (df["MonthlyCharges"] > df["MonthlyCharges"].median()).astype(int)

    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    scaler   = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    return X_scaled, y


# ── STEP 2: EDA ────────────────────────────────────────────────

def plot_eda(filepath: str) -> None:
    df = pd.read_csv(filepath)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["Churn_binary"] = (df["Churn"] == "Yes").astype(int)

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle("Customer Churn — Exploratory Data Analysis",
                 fontsize=16, fontweight="bold", y=1.01)

    # Churn distribution
    df["Churn"].value_counts().plot(kind="bar", ax=axes[0, 0],
                                    color=["#34d399", "#ef4444"])
    axes[0, 0].set_title("Churn Distribution")
    axes[0, 0].set_xlabel("Churn"); axes[0, 0].tick_params(rotation=0)

    # Tenure vs Churn
    df.groupby("Churn")["tenure"].plot(kind="kde", ax=axes[0, 1],
                                       legend=True)
    axes[0, 1].set_title("Tenure Distribution by Churn")

    # Monthly charges
    df.boxplot(column="MonthlyCharges", by="Churn", ax=axes[0, 2])
    axes[0, 2].set_title("Monthly Charges by Churn")
    plt.sca(axes[0, 2]); plt.title("Monthly Charges by Churn")

    # Contract type
    ct = df.groupby(["Contract", "Churn"]).size().unstack(fill_value=0)
    ct.plot(kind="bar", ax=axes[1, 0], color=["#34d399", "#ef4444"])
    axes[1, 0].set_title("Churn by Contract Type")
    axes[1, 0].tick_params(rotation=20)

    # Internet service
    isc = df.groupby(["InternetService", "Churn"]).size().unstack(fill_value=0)
    isc.plot(kind="bar", ax=axes[1, 1], color=["#34d399", "#ef4444"])
    axes[1, 1].set_title("Churn by Internet Service")
    axes[1, 1].tick_params(rotation=15)

    # Correlation heatmap (numeric only)
    num_cols = df.select_dtypes("number").columns.tolist()
    corr = df[num_cols].corr()[["Churn_binary"]].drop("Churn_binary")
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdYlGn",
                center=0, ax=axes[1, 2], cbar=False)
    axes[1, 2].set_title("Correlation with Churn")

    plt.tight_layout()
    plt.savefig("outputs/eda_overview.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✓ EDA chart saved → outputs/eda_overview.png")


# ── STEP 3: TRAIN & COMPARE ────────────────────────────────────

def train_and_compare(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {
        "Logistic Regression"  : LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest"        : RandomForestClassifier(n_estimators=150, random_state=42),
        "Gradient Boosting"    : GradientBoostingClassifier(n_estimators=150, random_state=42),
    }

    results = {}
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print("\n" + "=" * 58)
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        acc    = accuracy_score(y_test, y_pred)
        roc    = roc_auc_score(y_test, y_prob)
        cv     = cross_val_score(model, X, y, cv=skf, scoring="accuracy").mean()

        results[name] = {
            "model": model, "accuracy": acc,
            "roc_auc": roc, "cv_score": cv,
            "y_pred": y_pred, "y_prob": y_prob
        }

        print(f"\nModel       : {name}")
        print(f"Accuracy    : {acc:.4f}  ({acc*100:.1f}%)")
        print(f"ROC-AUC     : {roc:.4f}")
        print(f"CV Score    : {cv:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred,
              target_names=["No Churn", "Churn"]))

    # ── Save best model ──────────────────────────────────────
    best_name  = max(results, key=lambda k: results[k]["accuracy"])
    best       = results[best_name]
    with open("outputs/best_model.pkl", "wb") as f:
        pickle.dump(best["model"], f)
    print(f"\n🏆 Best model: {best_name} ({best['accuracy']*100:.1f}%)")
    print("   Saved → outputs/best_model.pkl")

    # ── ROC Curve ────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#4f8ef7", "#34d399", "#f59e0b"]
    for i, (name, r) in enumerate(results.items()):
        fpr, tpr, _ = roc_curve(y_test, r["y_prob"])
        ax.plot(fpr, tpr, color=colors[i], linewidth=2,
                label=f"{name} (AUC={r['roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax.set_title("ROC Curves — Model Comparison", fontsize=14, fontweight="bold")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()
    plt.tight_layout()
    plt.savefig("outputs/roc_curves.png", dpi=150)
    plt.close()
    print("  ✓ ROC curves saved → outputs/roc_curves.png")

    # ── Feature Importance (Random Forest) ───────────────────
    rf = results["Random Forest"]["model"]
    fi = pd.Series(rf.feature_importances_, index=X.columns).nlargest(15)

    fig, ax = plt.subplots(figsize=(10, 6))
    fi.sort_values().plot(kind="barh", ax=ax, color="#4f8ef7", alpha=0.85)
    ax.set_title("Top 15 Feature Importances (Random Forest)",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Importance Score")
    plt.tight_layout()
    plt.savefig("outputs/feature_importance.png", dpi=150)
    plt.close()
    print("  ✓ Feature importance chart → outputs/feature_importance.png")

    print("\n🔑 Top 10 Churn Drivers:")
    for feat, imp in fi.head(10).items():
        print(f"  {feat:<40} {imp:.4f}")

    return results


# ── MAIN ───────────────────────────────────────────────────────
if __name__ == "__main__":
    DATA_PATH = "data/raw/telco_churn.csv"

    print("📊 Step 1: EDA")
    plot_eda(DATA_PATH)

    print("\n🔧 Step 2: Preprocessing")
    X, y = preprocess(DATA_PATH)

    print("\n🤖 Step 3: Training & Evaluation")
    train_and_compare(X, y)
