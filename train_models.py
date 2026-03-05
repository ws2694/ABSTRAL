"""ML model training pipeline for bone metastasis prediction.

Aligned with team benchmark (2.17 Benchmarks.py):
  - Uses raw filtered features (same columns, same filtering)
  - StandardScaler for distance/gradient-sensitive models
  - Same train/test split (80/20, stratified, seed=42)

Usage:
    python train_models.py --data-dir data/
    python train_models.py --data-dir data/ --output data/models
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             brier_score_loss, accuracy_score, f1_score)

from tools.ml_models import EnsemblePredictor


def load_features(data_dir: str) -> tuple:
    """Load raw features and labels from data_processor output."""
    data_dir = Path(data_dir)

    features_path = data_dir / "features.npy"
    labels_path = data_dir / "labels.npy"
    columns_path = data_dir / "feature_columns.json"

    if features_path.exists() and labels_path.exists():
        X = np.load(str(features_path))
        y = np.load(str(labels_path))
        with open(str(columns_path)) as f:
            feature_names = json.load(f)
        print(f"Loaded raw features: {X.shape}, labels: {y.shape}")
        print(f"Feature columns: {len(feature_names)}")
        return X, y, feature_names

    # Fallback: load from parquet using feature engineering
    parquet_files = list(data_dir.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No features.npy or parquet files in {data_dir}")

    print(f"No features.npy found, falling back to parquet: {parquet_files[0]}")
    return _load_from_parquet(str(parquet_files[0]))


def _load_from_parquet(data_path: str) -> tuple:
    """Fallback: load from parquet using feature engineering."""
    from tools.feature_engineer import engineer_features, FEATURE_NAMES
    from tools.ops_calculator import compute_ops

    df = pd.read_parquet(data_path)
    print(f"Loaded {len(df)} patients from {data_path}")

    label_col = None
    for candidate in ["bone_metastasis", "label", "target", "MST", "METS_BONE"]:
        if candidate in df.columns:
            label_col = candidate
            break
    if label_col is None:
        raise ValueError(f"Cannot find label column: {list(df.columns)}")

    y = df[label_col].values.astype(int)

    pid_col = None
    for candidate in ["patient_id", "id", "pid"]:
        if candidate in df.columns:
            pid_col = candidate
            break
    if pid_col is None:
        df["patient_id"] = [f"P{i:05d}" for i in range(len(df))]
        pid_col = "patient_id"

    features_list = []
    for i, row in df.iterrows():
        record = _row_to_patient_record(row, df.columns)
        ops_result = compute_ops(record)
        feat = engineer_features(record, ops_result)
        features_list.append(feat["feature_vector"])

    X = np.array(features_list)
    return X, y, list(FEATURE_NAMES)


def _row_to_patient_record(row: pd.Series, columns: pd.Index) -> dict:
    """Convert a DataFrame row to patient_record dict (fallback)."""
    record = {"patient_id": str(row.get("patient_id", "unknown"))}

    demographics = {}
    for col, key in [("age", "age"), ("sex", "sex")]:
        if col in columns and pd.notna(row.get(col)):
            val = row[col]
            if key == "sex" and isinstance(val, (int, float)):
                val = "female" if val == 1 else "male"
            demographics[key] = val
    record["demographics"] = demographics

    meds = []
    if "medications" in columns and pd.notna(row.get("medications")):
        raw = row["medications"]
        if isinstance(raw, str):
            meds = json.loads(raw)
        elif isinstance(raw, list):
            meds = raw
    record["medications"] = meds

    comorbidities = []
    if "comorbidities" in columns and pd.notna(row.get("comorbidities")):
        raw = row["comorbidities"]
        if isinstance(raw, str):
            comorbidities = json.loads(raw)
        elif isinstance(raw, list):
            comorbidities = raw
    record["comorbidities"] = comorbidities

    for col in ["cci_score", "cci"]:
        if col in columns and pd.notna(row.get(col)):
            record["cci_score"] = int(row[col])
            break

    record["labs"] = {}
    for col in ["observation_months", "follow_up_months"]:
        if col in columns and pd.notna(row.get(col)):
            record["observation_months"] = int(row[col])
            break
    else:
        record["observation_months"] = 24

    return record


def train_and_evaluate(X: np.ndarray, y: np.ndarray, output_dir: str,
                       feature_names: list = None):
    """Train MLP, XGBoost, RF ensemble and print evaluation metrics.

    Uses StandardScaler for MLP (matching benchmark approach).
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nTrain: {len(X_train)} | Test: {len(X_test)}")
    print(f"Train class dist: 0={np.sum(y_train==0)}, 1={np.sum(y_train==1)}")
    print(f"Test class dist:  0={np.sum(y_test==0)}, 1={np.sum(y_test==1)}")

    # StandardScaler (matching benchmark)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train ensemble (MLP uses scaled, XGB/RF use raw — matching benchmark)
    print("\nTraining models...")
    ensemble = EnsemblePredictor(model_dir=output_dir)
    ensemble.fit_with_scaling(X_train, X_train_scaled, y_train)

    # Evaluate
    print("\n" + "=" * 65)
    print(f"{'Model':<20} {'AUC':>6} {'AUPRC':>6} {'Brier':>6} {'Acc':>6} {'F1':>6}")
    print("=" * 65)

    for name, model, use_scaled in [
        ("MLP", ensemble.mlp, True),
        ("GradientBoosting", ensemble.xgb, False),
        ("RandomForest", ensemble.rf, False),
    ]:
        Xte = X_test_scaled if use_scaled else X_test
        proba = model.predict_proba(Xte)
        pred = (proba >= 0.5).astype(int)
        auc = roc_auc_score(y_test, proba)
        auprc = average_precision_score(y_test, proba)
        brier = brier_score_loss(y_test, proba)
        acc = accuracy_score(y_test, pred)
        f1 = f1_score(y_test, pred, zero_division=0)
        print(f"  {name:<18} {auc:>6.4f} {auprc:>6.4f} {brier:>6.4f} {acc:>6.4f} {f1:>6.4f}")

    # Ensemble
    ensemble_proba = np.array([
        ensemble.predict_scaled(X_test[i], X_test_scaled[i])["ensemble"]
        for i in range(len(X_test))
    ])
    pred = (ensemble_proba >= 0.5).astype(int)
    auc = roc_auc_score(y_test, ensemble_proba)
    auprc = average_precision_score(y_test, ensemble_proba)
    brier = brier_score_loss(y_test, ensemble_proba)
    acc = accuracy_score(y_test, pred)
    f1 = f1_score(y_test, pred, zero_division=0)
    print(f"  {'Ensemble':<18} {auc:>6.4f} {auprc:>6.4f} {brier:>6.4f} {acc:>6.4f} {f1:>6.4f}")
    print("=" * 65)

    # Save
    ensemble.save()
    ensemble.save_scaler(scaler)
    print(f"\nModels saved to {output_dir}/")

    # Save metadata
    meta = {
        "n_train": len(X_train),
        "n_test": len(X_test),
        "n_features": X.shape[1],
        "feature_names": feature_names or [],
        "ensemble_auc": float(auc),
        "ensemble_auprc": float(auprc),
        "class_distribution": {
            "train": {"0": int(np.sum(y_train == 0)), "1": int(np.sum(y_train == 1))},
            "test": {"0": int(np.sum(y_test == 0)), "1": int(np.sum(y_test == 1))}
        }
    }
    meta_path = Path(output_dir) / "model_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Metadata saved to {meta_path}")


def main():
    parser = argparse.ArgumentParser(description="Train ML models for ABSTRAL")
    parser.add_argument("--data-dir", default="data/",
                        help="Directory with features.npy + labels.npy")
    parser.add_argument("--output", default="data/models",
                        help="Directory to save trained models")
    args = parser.parse_args()

    Path(args.output).mkdir(parents=True, exist_ok=True)

    X, y, feature_names = load_features(args.data_dir)
    train_and_evaluate(X, y, args.output, feature_names)


if __name__ == "__main__":
    main()
