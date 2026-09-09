"""
Data Preprocessing Pipeline
Handles disguised zeros, median imputation, and feature scaling.
Avoids data leakage by fitting transformers strictly on training split.
"""

from typing import List, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

FEATURE_NAMES = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]

ZERO_DISGUISED_COLS = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]


class HealthcareDataPreprocessor:
    """
    Production-grade preprocessor for healthcare vitals.
    Replaces biologically invalid zeros with NaN, imputes via median, and scales via StandardScaler.
    """

    def __init__(self):
        self.feature_names = FEATURE_NAMES
        self.zero_disguised_cols = ZERO_DISGUISED_COLS
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()
        self.is_fitted = False

    def replace_disguised_zeros(self, df: pd.DataFrame) -> pd.DataFrame:
        """Replace physiologically impossible 0s with NaN."""
        df_clean = df.copy()
        for col in self.zero_disguised_cols:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].replace(0, np.nan)
        return df_clean

    def fit(self, X: Union[pd.DataFrame, np.ndarray]):
        """Fit imputer and scaler on training data."""
        if isinstance(X, np.ndarray):
            X_df = pd.DataFrame(X, columns=self.feature_names)
        else:
            X_df = X[self.feature_names].copy()

        X_clean = self.replace_disguised_zeros(X_df)
        X_imputed = self.imputer.fit_transform(X_clean)
        self.scaler.fit(X_imputed)
        self.is_fitted = True
        return self

    def transform(self, X: Union[pd.DataFrame, np.ndarray, dict]) -> np.ndarray:
        """Transform new patient vitals using fitted imputer and scaler."""
        if not self.is_fitted:
            raise RuntimeError("Preprocessor has not been fitted yet. Call fit() or load serialized artifacts.")

        if isinstance(X, dict):
            X_df = pd.DataFrame([X], columns=self.feature_names)
        elif isinstance(X, np.ndarray):
            if X.ndim == 1:
                X_df = pd.DataFrame([X], columns=self.feature_names)
            else:
                X_df = pd.DataFrame(X, columns=self.feature_names)
        else:
            X_df = X[self.feature_names].copy()

        X_clean = self.replace_disguised_zeros(X_df)
        X_imputed = self.imputer.transform(X_clean)
        X_scaled = self.scaler.transform(X_imputed)
        return X_scaled

    def fit_transform(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Fit and transform in a single step."""
        return self.fit(X).transform(X)
