"""
Exploratory Data Analysis (EDA) Script
Smart Healthcare Diagnosis API (Disease Prediction System)
Analyzes feature distributions, disguised zeros, class balance, outliers, and correlations.
Saves summary charts into `eda_reports/`.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configure styling
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({"font.size": 10, "figure.autolayout": True})

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(BASE_DIR, "dataset", "diabetes.csv")
REPORTS_DIR = os.path.join(BASE_DIR, "eda_reports")

ZERO_DISGUISED_COLS = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]


def run_eda():
    os.makedirs(REPORTS_DIR, exist_ok=True)
    print("=" * 60)
    print("STARTING EXPLORATORY DATA ANALYSIS (EDA)")
    print("=" * 60)

    # 1. Load Dataset
    df = pd.read_csv(DATASET_PATH)
    print(f"\n[1] Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    print(df.info())

    # 2. Summary Statistics
    print("\n[2] Summary Statistics:")
    desc = df.describe().T
    print(desc[["mean", "std", "min", "50%", "max"]])

    # 3. Disguised Zeros Check
    print("\n[3] Disguised Zeros Analysis:")
    zero_stats = {}
    for col in ZERO_DISGUISED_COLS:
        zero_count = (df[col] == 0).sum()
        pct = (zero_count / len(df)) * 100
        zero_stats[col] = (zero_count, pct)
        print(f"  - {col}: {zero_count} zeros ({pct:.2f}%)")

    # 4. Class Balance
    class_counts = df["Outcome"].value_counts()
    print("\n[4] Class Distribution:")
    print(f"  - Non-Diabetic (Outcome=0): {class_counts[0]} ({class_counts[0]/len(df)*100:.1f}%)")
    print(f"  - Diabetic (Outcome=1): {class_counts[1]} ({class_counts[1]/len(df)*100:.1f}%)")

    # 5. Outlier Detection using IQR
    print("\n[5] Outlier Detection (IQR Method):")
    for col in df.columns[:-1]:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outliers = ((df[col] < lower) | (df[col] > upper)).sum()
        print(f"  - {col}: {outliers} potential outliers (Bounds: [{lower:.2f}, {upper:.2f}])")

    # ------------------- VISUALIZATIONS -------------------

    # Visualization 1: Feature Distributions
    fig, axes = plt.subplots(4, 2, figsize=(12, 14))
    axes = axes.flatten()
    features = [c for c in df.columns if c != "Outcome"]

    for idx, feature in enumerate(features):
        sns.histplot(df, x=feature, hue="Outcome", kde=True, ax=axes[idx], bins=30,
                     palette={0: "#2b5c8f", 1: "#d95f02"}, alpha=0.6)
        axes[idx].set_title(f"Distribution of {feature}", fontsize=11, fontweight="bold")
        axes[idx].set_xlabel(feature)
        axes[idx].set_ylabel("Count")

    plt.tight_layout()
    dist_path = os.path.join(REPORTS_DIR, "feature_distributions.png")
    plt.savefig(dist_path, dpi=200)
    plt.close()
    print(f"\n[+] Saved: {dist_path}")

    # Visualization 2: Outcome Balance (Countplot + Pie)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))
    sns.countplot(data=df, x="Outcome", ax=ax1, palette=["#2b5c8f", "#d95f02"])
    ax1.set_xticklabels(["Non-Diabetic (0)", "Diabetic (1)"])
    ax1.set_title("Class Frequency", fontweight="bold")
    ax1.set_ylabel("Count")
    for p in ax1.patches:
        ax1.annotate(f"{int(p.get_height())}", (p.get_x() + p.get_width() / 2., p.get_height() / 2),
                     ha="center", va="center", color="white", fontweight="bold", fontsize=12)

    df["Outcome"].value_counts().plot.pie(
        ax=ax2,
        autopct="%1.1f%%",
        labels=["Non-Diabetic (0)", "Diabetic (1)"],
        colors=["#2b5c8f", "#d95f02"],
        explode=[0, 0.05],
        startangle=140,
        textprops={"fontsize": 11, "fontweight": "bold"}
    )
    ax2.set_ylabel("")
    ax2.set_title("Outcome Ratio", fontweight="bold")
    plt.tight_layout()
    outcome_path = os.path.join(REPORTS_DIR, "outcome_distribution.png")
    plt.savefig(outcome_path, dpi=200)
    plt.close()
    print(f"[+] Saved: {outcome_path}")

    # Visualization 3: Correlation Heatmap
    plt.figure(figsize=(10, 8))
    corr = df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1,
                linewidths=0.5, cbar_kws={"shrink": 0.8})
    plt.title("Feature Correlation Matrix (with Outcome)", fontsize=13, fontweight="bold", pad=12)
    heatmap_path = os.path.join(REPORTS_DIR, "correlation_heatmap.png")
    plt.savefig(heatmap_path, dpi=200)
    plt.close()
    print(f"[+] Saved: {heatmap_path}")

    # Visualization 4: Boxplots for Outlier Detection
    fig, axes = plt.subplots(2, 4, figsize=(14, 8))
    axes = axes.flatten()
    for idx, feature in enumerate(features):
        sns.boxplot(data=df, y=feature, x="Outcome", ax=axes[idx], palette=["#2b5c8f", "#d95f02"])
        axes[idx].set_title(f"{feature} by Outcome", fontsize=10, fontweight="bold")
        axes[idx].set_xticklabels(["Non-Diabetic", "Diabetic"])
    plt.tight_layout()
    box_path = os.path.join(REPORTS_DIR, "boxplots_outliers.png")
    plt.savefig(box_path, dpi=200)
    plt.close()
    print(f"[+] Saved: {box_path}")

    # 6. Save textual summary report
    summary_txt = f"""SMART HEALTHCARE DIAGNOSIS API - EDA SUMMARY
==================================================
1. Dataset Overview:
   - Total records: {df.shape[0]}
   - Features: {len(features)} clinical attributes + 1 target ('Outcome')
   - Class Balance:
     * Non-Diabetic (0): {class_counts[0]} ({class_counts[0]/len(df)*100:.1f}%)
     * Diabetic (1): {class_counts[1]} ({class_counts[1]/len(df)*100:.1f}%)

2. Disguised Zeros Detected:
   Biologically impossible 0 values found in medical fields:
   * Glucose: {zero_stats['Glucose'][0]} zeros ({zero_stats['Glucose'][1]:.2f}%)
   * BloodPressure: {zero_stats['BloodPressure'][0]} zeros ({zero_stats['BloodPressure'][1]:.2f}%)
   * SkinThickness: {zero_stats['SkinThickness'][0]} zeros ({zero_stats['SkinThickness'][1]:.2f}%)
   * Insulin: {zero_stats['Insulin'][0]} zeros ({zero_stats['Insulin'][1]:.2f}%)
   * BMI: {zero_stats['BMI'][0]} zeros ({zero_stats['BMI'][1]:.2f}%)

   Recommendation: Replace 0 with NaN and impute using median strategy
   computed strictly on the training set to prevent data leakage.

3. Key Correlations with Diabetes Outcome:
   - Top positive predictor: Glucose (r = {corr.loc['Glucose', 'Outcome']:.3f})
   - Second predictor: BMI (r = {corr.loc['BMI', 'Outcome']:.3f})
   - Third predictor: Age (r = {corr.loc['Age', 'Outcome']:.3f})
   - Fourth predictor: Pregnancies (r = {corr.loc['Pregnancies', 'Outcome']:.3f})

4. Next Pipeline Step:
   - Imputation + Feature Scaling (StandardScaler)
   - Multi-model evaluation (Logistic Regression, Decision Tree, Random Forest, SVM, KNN, Naive Bayes)
==================================================
"""
    summary_path = os.path.join(REPORTS_DIR, "eda_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary_txt)
    print(f"[+] Saved Summary Report: {summary_path}")
    print("\nEDA Completed Successfully!")


if __name__ == "__main__":
    run_eda()
