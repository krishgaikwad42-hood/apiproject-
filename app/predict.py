"""
Model Inference Service
Loads serialized artifacts and executes the end-to-end prediction pipeline.
"""

import os
import json
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


class DiabetesInferenceService:
    """Manages model loading and prediction inference."""

    def __init__(self):
        self.model = None
        self.scaler = None
        self.imputer = None
        self.metadata = {}
        self.is_loaded = False
        self.load_artifacts()

    def load_artifacts(self):
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
                f"Successfully loaded artifacts. Selected model: {self.metadata.get('selected_model', 'Unknown')}"
            )
        except Exception as e:
            logger.error(f"Error loading model artifacts: {str(e)}")
            self.is_loaded = False

    def predict(self, vitals: Dict[str, Any]) -> Dict[str, Any]:
        """
        Runs inference on patient vitals.
        Returns prediction, confidence percentage, probability, and risk level.
        """
        if not self.is_loaded or self.model is None:
            raise RuntimeError("Model artifacts are not loaded in memory. Please train the model first.")

        # Construct DataFrame in exact feature order
        df_input = pd.DataFrame([vitals], columns=FEATURE_NAMES)

        # Handle disguised zeros
        for col in ZERO_DISGUISED_COLS:
            if col in df_input.columns:
                df_input[col] = df_input[col].replace(0, np.nan)

        # Preprocessing: Impute then Scale
        imputed_data = self.imputer.transform(df_input)
        scaled_data = self.scaler.transform(imputed_data)

        # Predict probability
        if hasattr(self.model, "predict_proba"):
            probabilities = self.model.predict_proba(scaled_data)[0]
            diabetic_prob = float(probabilities[1])
        else:
            raw_pred = self.model.predict(scaled_data)[0]
            diabetic_prob = 1.0 if raw_pred == 1 else 0.0

        # Determine outcome and confidence
        is_diabetic = diabetic_prob >= 0.50
        prediction_label = "Diabetes" if is_diabetic else "No Diabetes"
        confidence_val = diabetic_prob if is_diabetic else (1.0 - diabetic_prob)
        confidence_str = f"{confidence_val * 100:.1f}%"

        # Stratify clinical risk level
        if diabetic_prob >= 0.65:
            risk_level = "High Risk"
        elif diabetic_prob >= 0.35:
            risk_level = "Moderate Risk"
        else:
            risk_level = "Low Risk"

        model_name = self.metadata.get("selected_model", type(self.model).__name__)

        return {
            "status": "success",
            "prediction": prediction_label,
            "confidence": confidence_str,
            "probability": round(diabetic_prob, 4),
            "risk_level": risk_level,
            "model_used": model_name,
            "timestamp": datetime.now().isoformat(),
        }


# Singleton instance
inference_service = DiabetesInferenceService()
