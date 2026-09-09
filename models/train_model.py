"""
Model Training & Evaluation Pipeline
Smart Healthcare Diagnosis API (Disease Prediction System)

Trains, compares, and evaluates 6 ML algorithms:
1. Logistic Regression
2. Decision Tree Classifier
3. Random Forest Classifier
4. Support Vector Machine (SVM)
5. K-Nearest Neighbors (KNN)
6. Gaussian Naive Bayes

Selects the best model prioritizing F1-Score & Recall (reducing False Negatives),
and serializes artifacts into `saved_models/`.
"""

import os
import sys
import json
from datetime import datetime

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# Ensure root directory in sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from utils.preprocessing import HealthcareDataPreprocessor, FEATURE_NAMES

DATASET_PATH = os.path.join(BASE_DIR, "dataset", "diabetes.csv")
SAVED_MODELS_DIR = os.path.join(BASE_DIR, "saved_models")
REPORTS_DIR = os.path.join(BASE_DIR, "eda_reports")


def get_candidate_models():
    """Returns candidate classification algorithms."""
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, min_samples_split=10, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42),
        "Support Vector Machine": SVC(kernel="rbf", C=1.0, probability=True, random_state=42),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=7),
        "Gaussian Naive Bayes": GaussianNB(),
    }


def train_and_evaluate():
    os.makedirs(SAVED_MODELS_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    print("=" * 70)
    print("STARTING MODEL TRAINING & BENCHMARKING PIPELINE")
    print("=" * 70)

    # 1. Load Data
    df = pd.read_csv(DATASET_PATH)
    X = df[FEATURE_NAMES]
    y = df["Outcome"]

    print(f"Dataset: {len(df)} samples, {len(FEATURE_NAMES)} features.")

    # 2. Train-Test Split (80/20 Stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"Train Set: {X_train.shape[0]} samples | Test Set: {X_test.shape[0]} samples (Stratified)")

    # 3. Preprocessing (Avoid Data Leakage)
    preprocessor = HealthcareDataPreprocessor()
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # 4. Train & Evaluate Candidates
    candidate_models = get_candidate_models()
    results = {}
    roc_curves = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print("\n" + "-" * 70)
    print(f"{'Algorithm':<25} | {'Acc':<6} | {'Prec':<6} | {'Recall':<6} | {'F1':<6} | {'AUC':<6} | {'CV-F1':<6}")
    print("-" * 70)

    for name, model in candidate_models.items():
        # Train
        model.fit(X_train_processed, y_train)

        # Cross Validation F1
        cv_f1_scores = cross_val_score(model, X_train_processed, y_train, cv=cv, scoring="f1")
        mean_cv_f1 = np.mean(cv_f1_scores)

        # Test Predictions
        y_pred = model.predict(X_test_processed)
        y_prob = model.predict_proba(X_test_processed)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        cm = confusion_matrix(y_test, y_pred)

        results[name] = {
            "accuracy": round(float(acc), 4),
            "precision": round(float(prec), 4),
            "recall": round(float(rec), 4),
            "f1_score": round(float(f1), 4),
            "roc_auc": round(float(auc), 4),
            "cv_f1_mean": round(float(mean_cv_f1), 4),
            "confusion_matrix": cm.tolist(),
        }

        # Store ROC curve data
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_curves[name] = (fpr, tpr, auc)

        print(f"{name:<25} | {acc:.4f} | {prec:.4f} | {rec:.4f} | {f1:.4f} | {auc:.4f} | {mean_cv_f1:.4f}")

    print("-" * 70)

    # 5. Model Selection Strategy (Healthcare Priority: F1-score & Recall)
    # Score = 0.6 * F1 + 0.4 * Recall (penalizes False Negatives)
    best_model_name = max(
        results.keys(),
        key=lambda m: (0.6 * results[m]["f1_score"] + 0.4 * results[m]["recall"], results[m]["roc_auc"]),
    )
    best_model = candidate_models[best_model_name]
    best_metrics = results[best_model_name]

    print(f"\n[*] Selected Best Model: {best_model_name}")
    print(f"    - F1-Score: {best_metrics['f1_score']}")
    print(f"    - Recall:   {best_metrics['recall']}")
    print(f"    - Accuracy: {best_metrics['accuracy']}")
    print(f"    - ROC-AUC:  {best_metrics['roc_auc']}")

    # 6. Visualizations
    # Comparison Bar Chart
    plt.figure(figsize=(12, 6))
    metrics_names = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
    df_metrics = pd.DataFrame(results).T[metrics_names]
    ax = df_metrics.plot(kind="bar", figsize=(12, 6), colormap="viridis", width=0.8)
    plt.title("Candidate Model Performance Comparison", fontsize=14, fontweight="bold")
    plt.ylabel("Score (0.0 to 1.0)")
    plt.xlabel("Algorithm")
    plt.xticks(rotation=25, ha="right")
    plt.ylim(0.4, 1.0)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.legend(["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"], loc="lower right")
    plt.tight_layout()
    comp_path = os.path.join(REPORTS_DIR, "model_comparison.png")
    plt.savefig(comp_path, dpi=200)
    plt.close()
    print(f"[+] Saved comparison plot: {comp_path}")

    # ROC Curves Plot
    plt.figure(figsize=(9, 7))
    for name, (fpr, tpr, auc) in roc_curves.items():
        linewidth = 2.5 if name == best_model_name else 1.2
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})", linewidth=linewidth)
    plt.plot([0, 1], [0, 1], "k--", alpha=0.6, label="Random Guess (AUC = 0.50)")
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    plt.ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=11)
    plt.title("Receiver Operating Characteristic (ROC) Curves", fontsize=13, fontweight="bold")
    plt.legend(loc="lower right")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    roc_path = os.path.join(REPORTS_DIR, "roc_curves.png")
    plt.savefig(roc_path, dpi=200)
    plt.close()
    print(f"[+] Saved ROC curves: {roc_path}")

    # 7. Serialization
    model_pkl_path = os.path.join(SAVED_MODELS_DIR, "diabetes_model.pkl")
    scaler_pkl_path = os.path.join(SAVED_MODELS_DIR, "scaler.pkl")
    imputer_pkl_path = os.path.join(SAVED_MODELS_DIR, "imputer.pkl")
    metadata_json_path = os.path.join(SAVED_MODELS_DIR, "model_metadata.json")

    joblib.dump(best_model, model_pkl_path)
    joblib.dump(preprocessor.scaler, scaler_pkl_path)
    joblib.dump(preprocessor.imputer, imputer_pkl_path)

    metadata = {
        "project": "Smart Healthcare Diagnosis API",
        "disease": "Diabetes",
        "trained_at": datetime.now().isoformat(),
        "selected_model": best_model_name,
        "selected_model_metrics": best_metrics,
        "feature_names": FEATURE_NAMES,
        "target_mapping": {"0": "No Diabetes", "1": "Diabetes"},
        "all_model_evaluations": results,
    }

    with open(metadata_json_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n[+] Serialized Model:   {model_pkl_path}")
    print(f"[+] Serialized Scaler:  {scaler_pkl_path}")
    print(f"[+] Serialized Imputer: {imputer_pkl_path}")
    print(f"[+] Serialized Metadata: {metadata_json_path}")
    print("=" * 70)
    print("TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    train_and_evaluate()
