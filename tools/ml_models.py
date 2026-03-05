"""ML model wrappers for bone metastasis risk prediction.

Wraps sklearn MLP, GradientBoosting, and Random Forest classifiers with a
uniform interface. Supports separate scaled/unscaled training to match
team benchmark (MLP uses StandardScaler, tree models use raw features).
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Optional, List

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except (ImportError, OSError, Exception):
    HAS_XGBOOST = False


RANDOM_SEED = 42


class BasePredictor:
    """Base class for risk predictors."""

    def __init__(self):
        self.model = None
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray):
        raise NotImplementedError

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() or load() first.")
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return self.model.predict_proba(X)[:, 1]

    def predict(self, features) -> float:
        X = np.array(features).reshape(1, -1)
        return float(self.predict_proba(X)[0])

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self.model, f)

    def load(self, path: str):
        with open(path, "rb") as f:
            self.model = pickle.load(f)
        self.is_fitted = True


class MLPPredictor(BasePredictor):
    def fit(self, X: np.ndarray, y: np.ndarray):
        self.model = MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            solver="adam",
            max_iter=200,
            random_state=RANDOM_SEED,
        )
        self.model.fit(X, y)
        self.is_fitted = True
        return self


class XGBPredictor(BasePredictor):
    def fit(self, X: np.ndarray, y: np.ndarray):
        if HAS_XGBOOST:
            self.model = XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                use_label_encoder=False,
                eval_metric="logloss",
                random_state=RANDOM_SEED,
                verbosity=0,
            )
        else:
            self.model = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                random_state=RANDOM_SEED,
            )
            print("  [XGBPredictor] Using sklearn GradientBoostingClassifier (XGBoost unavailable)")
        self.model.fit(X, y)
        self.is_fitted = True
        return self


class RFPredictor(BasePredictor):
    def fit(self, X: np.ndarray, y: np.ndarray):
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=RANDOM_SEED,
            n_jobs=-1,
        )
        self.model.fit(X, y)
        self.is_fitted = True
        return self


class EnsemblePredictor:
    """Weighted ensemble of MLP, XGBoost/GradientBoosting, and Random Forest.

    Supports benchmark-style training: MLP on scaled features,
    tree models on raw features.
    """

    def __init__(self, model_dir: str = "data/models"):
        self.model_dir = Path(model_dir)
        self.mlp = MLPPredictor()
        self.xgb = XGBPredictor()
        self.rf = RFPredictor()
        self.scaler: Optional[StandardScaler] = None
        self.default_weights = [0.35, 0.40, 0.25]  # mlp, xgb, rf

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Train all models on the same features (no scaling distinction)."""
        self.mlp.fit(X, y)
        self.xgb.fit(X, y)
        self.rf.fit(X, y)
        return self

    def fit_with_scaling(self, X_raw: np.ndarray, X_scaled: np.ndarray,
                         y: np.ndarray):
        """Train MLP on scaled, tree models on raw (matching benchmark)."""
        self.mlp.fit(X_scaled, y)
        self.xgb.fit(X_raw, y)
        self.rf.fit(X_raw, y)
        return self

    def save(self):
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.mlp.save(str(self.model_dir / "mlp.pkl"))
        self.xgb.save(str(self.model_dir / "xgb.pkl"))
        self.rf.save(str(self.model_dir / "rf.pkl"))

    def save_scaler(self, scaler: StandardScaler):
        """Save the StandardScaler used for MLP."""
        self.scaler = scaler
        with open(str(self.model_dir / "scaler.pkl"), "wb") as f:
            pickle.dump(scaler, f)

    def load(self):
        self.mlp.load(str(self.model_dir / "mlp.pkl"))
        self.xgb.load(str(self.model_dir / "xgb.pkl"))
        self.rf.load(str(self.model_dir / "rf.pkl"))
        scaler_path = self.model_dir / "scaler.pkl"
        if scaler_path.exists():
            with open(str(scaler_path), "rb") as f:
                self.scaler = pickle.load(f)

    def predict(self, features, model: str = "ensemble",
                weights: Optional[list] = None) -> dict:
        """Predict risk score(s) for a single patient feature vector.

        If a scaler is loaded, MLP uses scaled features automatically.
        """
        X_raw = np.array(features).reshape(1, -1)
        X_mlp = self.scaler.transform(X_raw) if self.scaler else X_raw

        mlp_score = float(self.mlp.predict_proba(X_mlp)[0])
        xgb_score = float(self.xgb.predict_proba(X_raw)[0])
        rf_score = float(self.rf.predict_proba(X_raw)[0])

        w = weights if weights and len(weights) == 3 else self.default_weights
        w_sum = sum(w)
        w = [wi / w_sum for wi in w]

        ensemble_score = w[0] * mlp_score + w[1] * xgb_score + w[2] * rf_score

        scores = {
            "mlp": round(mlp_score, 4),
            "xgb": round(xgb_score, 4),
            "rf": round(rf_score, 4),
            "ensemble": round(ensemble_score, 4),
            "weights_used": [round(wi, 3) for wi in w],
        }

        if model == "ensemble":
            scores["selected_score"] = scores["ensemble"]
        elif model in scores:
            scores["selected_score"] = scores[model]
        else:
            scores["selected_score"] = scores["ensemble"]

        scores["confidence"] = _compute_confidence(mlp_score, xgb_score, rf_score)
        return scores

    def predict_scaled(self, features_raw, features_scaled,
                       weights: Optional[list] = None) -> dict:
        """Predict with pre-computed scaled features (for batch evaluation)."""
        X_raw = np.array(features_raw).reshape(1, -1)
        X_scaled = np.array(features_scaled).reshape(1, -1)

        mlp_score = float(self.mlp.predict_proba(X_scaled)[0])
        xgb_score = float(self.xgb.predict_proba(X_raw)[0])
        rf_score = float(self.rf.predict_proba(X_raw)[0])

        w = weights if weights and len(weights) == 3 else self.default_weights
        w_sum = sum(w)
        w = [wi / w_sum for wi in w]

        ensemble_score = w[0] * mlp_score + w[1] * xgb_score + w[2] * rf_score

        return {
            "mlp": round(mlp_score, 4),
            "xgb": round(xgb_score, 4),
            "rf": round(rf_score, 4),
            "ensemble": round(ensemble_score, 4),
            "weights_used": [round(wi, 3) for wi in w],
            "confidence": _compute_confidence(mlp_score, xgb_score, rf_score),
        }


def _compute_confidence(mlp: float, xgb: float, rf: float) -> str:
    """Compute prediction confidence based on model agreement."""
    scores = [mlp, xgb, rf]
    spread = max(scores) - min(scores)
    if spread < 0.1:
        return "high"
    elif spread < 0.2:
        return "moderate"
    else:
        return "low"
