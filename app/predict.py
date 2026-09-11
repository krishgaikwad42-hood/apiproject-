"""
Model Inference Service
Loads serialized artifacts and executes the end-to-end prediction pipeline.
"""

import os
import json
import uuid
import math
from datetime import datetime
from typing import Dict, Any

import joblib
import numpy as np
import pandas as pd

from utils.preprocessing import FEATURE_NAMES, ZERO_DISGUISED_COLS
from app.logger import logger

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAVED_MODELS_DIR = os.path.join(BASE_DIR, "saved_models")

MODEL_PATH = os.path.join(SAVED_MODELS_DIR, "diabetes_model.pkl")
SCALER_PATH = os.path.join(SAVED_MODELS_DIR, "scaler.pkl")
IMPUTER_PATH = os.path.join(SAVED_MODELS_DIR, "imputer.pkl")
METADATA_PATH = os.path.join(SAVED_MODELS_DIR, "model_metadata.json")

# ─── Threshold Constants ───────────────────────────────────────────────────────
# Minimum probability to classify a patient as diabetic positive.
DIABETIC_THRESHOLD: float = 0.50

# Risk stratification thresholds (probability ranges).
RISK_HIGH_THRESHOLD: float = 0.65
RISK_MODERATE_THRESHOLD: float = 0.35


def _stratify_risk(prob: float) -> str:
    """
    Converts a diabetic probability score into a clinical risk tier.

    Args:
        prob: Float probability in [0.0, 1.0].

    Returns:
        One of 'High Risk', 'Moderate Risk', or 'Low Risk'.
    """
    if prob >= RISK_HIGH_THRESHOLD:
        return "High Risk"
    if prob >= RISK_MODERATE_THRESHOLD:
        return "Moderate Risk"
    return "Low Risk"


def _validate_numeric_inputs(vitals: Dict[str, Any]) -> None:
    """
    Raises ValueError if any numeric vital contains NaN or ±Inf.

    Args:
        vitals: Raw input dictionary from the request payload.

    Raises:
        ValueError: If a non-finite value is detected in any numeric field.
    """
    numeric_fields = [
        "Glucose", "BloodPressure", "SkinThickness",
        "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Pregnancies",
    ]
    for field in numeric_fields:
        val = vitals.get(field)
        if val is not None and isinstance(val, (int, float)):
            if math.isnan(val) or math.isinf(val):
                raise ValueError(
                    f"Invalid numeric value for '{field}': {val!r}. "
                    "NaN and infinite values are not permitted."
                )


# ─── Inference Service ─────────────────────────────────────────────────────────

class DiabetesInferenceService:
    """
    Singleton service that manages model artifact loading and prediction inference.

    Artifacts loaded at startup:
        - diabetes_model.pkl  — trained scikit-learn classifier
        - scaler.pkl          — StandardScaler fitted on training data
        - imputer.pkl         — SimpleImputer for missing/zero-disguised values
        - model_metadata.json — training metrics and model selection info
    """

    def __init__(self):
        self.model = None
        self.scaler = None
        self.imputer = None
        self.metadata: Dict[str, Any] = {}
        self.is_loaded: bool = False
        self.load_artifacts()

    def load_artifacts(self) -> None:
        """Loads serialized model, preprocessors, and metadata from disk."""
        try:
            if not os.path.exists(MODEL_PATH):
                logger.warning(f"Model artifact not found at: {MODEL_PATH}")
                return

            self.model = joblib.load(MODEL_PATH)
            self.scaler = joblib.load(SCALER_PATH)
            self.imputer = joblib.load(IMPUTER_PATH)

            if os.path.exists(METADATA_PATH):
                with open(METADATA_PATH, "r", encoding="utf-8") as f:
                    self.metadata = json.load(f)

            self.is_loaded = True
            logger.info(
                f"Artifacts loaded successfully. "
                f"Model: {self.metadata.get('selected_model', 'Unknown')}"
            )
        except Exception as e:
            logger.error(f"Failed to load model artifacts: {str(e)}", exc_info=True)
            self.is_loaded = False

    def predict(self, vitals: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes the full inference pipeline on a patient vitals payload.

        Pipeline steps:
            1. Validate numeric inputs (NaN / Inf guard).
            2. Normalize gender and enforce pregnancy constraints.
            3. Construct ordered feature DataFrame.
            4. Replace disguised-zero values with NaN for imputation.
            5. Impute missing values → StandardScaler normalization.
            6. Predict class probability via the loaded classifier.
            7. Stratify risk level and format output.

        Args:
            vitals: Dictionary of raw patient vitals from the validated schema.

        Returns:
            Dictionary containing prediction, confidence, probability, risk level,
            model name, gender, prediction_id, status, and timestamp.

        Raises:
            RuntimeError: If model artifacts are not loaded.
            ValueError: If any input vital is NaN or infinite.
        """
        if not self.is_loaded or self.model is None:
            raise RuntimeError(
                "Model artifacts are not loaded. Please train the model first."
            )

        # 1. Guard against NaN / Inf inputs
        _validate_numeric_inputs(vitals)

        # 2. Gender normalization and Pregnancies enforcement
        gender = vitals.get("Gender") or vitals.get("gender") or "female"
        gender_clean = str(gender).strip().lower()
        gender_label = "male" if gender_clean in ["m", "male", "man", "boy"] else "female"

        if gender_label == "male":
            vitals["Pregnancies"] = 0
        elif vitals.get("Pregnancies") is None:
            vitals["Pregnancies"] = 0

        # 3. Construct DataFrame in exact trained feature order
        df_input = pd.DataFrame([vitals], columns=FEATURE_NAMES)

        # 4. Replace disguised zeros with NaN for imputation
        for col in ZERO_DISGUISED_COLS:
            if col in df_input.columns:
                df_input[col] = df_input[col].replace(0, np.nan)

        # 5. Impute then scale
        imputed_data = self.imputer.transform(df_input)
        scaled_data = self.scaler.transform(imputed_data)

        # 6. Predict probability
        if hasattr(self.model, "predict_proba"):
            probabilities = self.model.predict_proba(scaled_data)[0]
            diabetic_prob = float(probabilities[1])
        else:
            raw_pred = self.model.predict(scaled_data)[0]
            diabetic_prob = 1.0 if raw_pred == 1 else 0.0

        # 7. Determine outcome, confidence, and risk
        is_diabetic = diabetic_prob >= DIABETIC_THRESHOLD
        prediction_label = "Diabetes" if is_diabetic else "No Diabetes"
        confidence_val = diabetic_prob if is_diabetic else (1.0 - diabetic_prob)
        confidence_str = f"{confidence_val * 100:.1f}%"
        risk_level = _stratify_risk(diabetic_prob)
        model_name = self.metadata.get("selected_model", type(self.model).__name__)

        return {
            "status": "success",
            "prediction_id": str(uuid.uuid4()),
            "prediction": prediction_label,
            "confidence": confidence_str,
            "probability": round(diabetic_prob, 4),
            "risk_level": risk_level,
            "model_used": model_name,
            "patient_gender": gender_label,
            "timestamp": datetime.now().isoformat(),
        }


# ─── Module-Level Singleton ────────────────────────────────────────────────────
inference_service = DiabetesInferenceService()
