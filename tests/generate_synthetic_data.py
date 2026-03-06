"""Generate synthetic patient data for testing.

Creates a small parquet + features.npy + labels.npy + trains models.
"""

import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

DRUG_CLASSES = [
    "bisphosphonate", "calcitonin", "denosumab", "estrogen",
    "raloxifene", "teriparatide", "radiation_therapy", "chemotherapy_platinum",
]

CONDITIONS = [
    "diabetes", "copd", "heart_failure", "osteoporosis",
    "chronic_kidney_disease", "liver_disease", "malignancy",
]


def generate_patients(n=100, seed=42):
    rng = random.Random(seed)
    np_rng = np.random.RandomState(seed)

    rows = []
    for i in range(n):
        pid = f"P{i:05d}"
        age = rng.randint(45, 85)
        sex = rng.choice([0, 1])  # 0=male, 1=female
        obs_months = rng.randint(12, 36)

        # Generate medications
        n_meds = rng.randint(0, 4)
        meds = []
        for _ in range(n_meds):
            dc = rng.choice(DRUG_CLASSES)
            start = rng.randint(0, max(1, obs_months - 6))
            end = rng.randint(start + 1, obs_months)
            meds.append({"drug_class": dc, "start_month": start, "end_month": end})

        # Generate comorbidities
        n_comorb = rng.randint(0, 3)
        comorbs = []
        for _ in range(n_comorb):
            cond = rng.choice(CONDITIONS)
            month = rng.randint(0, max(1, obs_months // 2))
            comorbs.append({"condition": cond, "diagnosed_month": month})

        cci = len(comorbs) + rng.randint(0, 2)

        # Generate label with some signal from features
        risk = 0.2
        if age > 70:
            risk += 0.15
        if any(m["drug_class"] == "chemotherapy_platinum" for m in meds):
            risk += 0.2
        if any(c["condition"] == "osteoporosis" for c in comorbs):
            risk += 0.15
        if any(m["drug_class"] == "bisphosphonate" for m in meds):
            risk += 0.1  # paradox: signals existing bone disease
        if cci >= 3:
            risk += 0.1
        label = 1 if np_rng.random() < min(risk, 0.9) else 0

        rows.append({
            "patient_id": pid,
            "age": age,
            "sex": sex,
            "observation_months": obs_months,
            "medications": json.dumps(meds),
            "comorbidities": json.dumps(comorbs),
            "cci_score": cci,
            "alkaline_phosphatase": round(np_rng.normal(90, 30), 1),
            "calcium_level": round(np_rng.normal(9.5, 0.8), 1),
            "bone_density_tscore": round(np_rng.normal(-1.5, 1.2), 1),
            "bone_metastasis": label,
        })

    return pd.DataFrame(rows)


def main():
    data_dir = Path(__file__).parent.parent / "data"
    data_dir.mkdir(exist_ok=True)
    model_dir = data_dir / "models"
    model_dir.mkdir(exist_ok=True)

    print("Generating synthetic patient data...")
    df = generate_patients(n=100, seed=42)
    parquet_path = data_dir / "oncoagent.parquet"
    df.to_parquet(str(parquet_path), index=False)
    print(f"  Saved {len(df)} patients to {parquet_path}")
    print(f"  Label distribution: 0={sum(df['bone_metastasis']==0)}, 1={sum(df['bone_metastasis']==1)}")

    # Generate features using the real feature engineer
    print("\nEngineering features...")
    from tools.feature_engineer import engineer_features, N_FEATURES, FEATURE_NAMES
    from tools.ops_calculator import compute_ops

    features = []
    labels = []
    for _, row in df.iterrows():
        record = {
            "patient_id": row["patient_id"],
            "demographics": {"age": row["age"], "sex": "female" if row["sex"] == 1 else "male"},
            "medications": json.loads(row["medications"]),
            "comorbidities": json.loads(row["comorbidities"]),
            "cci_score": row["cci_score"],
            "labs": {
                "alkaline_phosphatase": row["alkaline_phosphatase"],
                "calcium_level": row["calcium_level"],
                "bone_density_tscore": row["bone_density_tscore"],
            },
            "observation_months": row["observation_months"],
        }
        ops = compute_ops(record)
        feat = engineer_features(record, ops)
        features.append(feat["feature_vector"])
        labels.append(row["bone_metastasis"])

    X = np.array(features)
    y = np.array(labels)
    print(f"  Feature matrix: {X.shape}")

    np.save(str(data_dir / "features.npy"), X)
    np.save(str(data_dir / "labels.npy"), y)
    with open(str(data_dir / "feature_columns.json"), "w") as f:
        json.dump(list(FEATURE_NAMES), f)
    print("  Saved features.npy, labels.npy, feature_columns.json")

    # Train models
    print("\nTraining models...")
    from train_models import train_and_evaluate
    train_and_evaluate(X, y, str(model_dir), list(FEATURE_NAMES))

    print("\nDone! Data and models ready for Level 3 tests.")


if __name__ == "__main__":
    main()
