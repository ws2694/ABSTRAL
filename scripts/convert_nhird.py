"""Convert NHIRD lung cancer CSV to ABSTRAL parquet format.

Reads 12.31_data_with_relative_days.csv and the column definitions,
produces a parquet file + features.npy + labels.npy + trained models.

Usage:
    python3 data/convert_nhird.py --csv path/to/12.31_data_with_relative_days.csv
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

# ── NHIRD drug columns → ABSTRAL drug_class mapping ──────────────────────────
NHIRD_DRUG_MAP = {
    "ALENDRONATE":  "bisphosphonate",
    "CALCITONIN":   "calcitonin",
    "DENOSUMAB":    "denosumab",
    "ETIDRONATE":   "bisphosphonate",
    "ESTROGEN":     "estrogen",
    "IBANDRONIC":   "bisphosphonate",
    "PAMIDRONATE":  "bisphosphonate",
    "RALOXIFENE":   "raloxifene",
    "RISEDRONATE":  "bisphosphonate",
    "TERIPARATIDE": "teriparatide",
    "ZOLEDRONIC":   "bisphosphonate",
}

# ── NHIRD comorbidity columns → ABSTRAL condition mapping ─────────────────────
NHIRD_COMORB_MAP = {
    "MI":             "myocardial_infarction",
    "CHF":            "heart_failure",
    "PVD":            "peripheral_vascular_disease",
    "CD":             "cerebrovascular_disease",
    "DEM":            "dementia",
    "CPD":            "copd",
    "RD":             "rheumatic_disease",
    "PUD":            "peptic_ulcer_disease",
    "MLD":            "liver_disease",
    "DIABETES_NO_CC": "diabetes",
    "DIABETES_CC":    "diabetes_with_complications",
    "HOP":            "hemiplegia",
    "RENAL":          "chronic_kidney_disease",
    "MA":             "malignancy",
    "MSLD":           "liver_disease_severe",
    "MST":            "metastatic_solid_tumor",
    "HIV":            "hiv",
}

# Charlson Comorbidity Index weights
CCI_WEIGHTS = {
    "MI": 1, "CHF": 1, "PVD": 1, "CD": 1, "DEM": 1, "CPD": 1,
    "RD": 1, "PUD": 1, "MLD": 1, "DIABETES_NO_CC": 1, "DIABETES_CC": 2,
    "HOP": 2, "RENAL": 2, "MA": 2, "MSLD": 3, "MST": 6, "HIV": 6,
}

DAYS_PER_MONTH = 30.44


def days_to_months(days: float) -> int:
    """Convert relative days to months (rounded)."""
    return round(days / DAYS_PER_MONTH)


def convert_row(row: pd.Series) -> dict:
    """Convert one NHIRD row to ABSTRAL record format."""
    # Demographics
    sex = int(row["ID_SEX"])
    age = float(row["ID_AGE_Y2001"])

    # Lung cancer info
    lc_rt = int(row["LUNG_CA_RT"])       # radiation therapy
    lc_ct = int(row["LUNG_CA_CT"])       # chemotherapy
    lc_op = int(row["LUNG_CA_OP"])       # surgery
    lc_cnt = int(row["LUNG_CA_CNT"])     # visit count
    lc_locations = sum(int(row[f"LUNG_CA_LOCATION{i}"]) for i in range(1, 7))

    # Medications from osteoporosis drug columns
    meds = []
    for nhird_name, drug_class in NHIRD_DRUG_MAP.items():
        flag_col = f"OSTEOPOROSIS_{nhird_name}"
        if int(row[flag_col]) == 1:
            start_days = float(row[f"OSTEOPOROSIS_{nhird_name}_DATE_FST"])
            end_days = float(row[f"OSTEOPOROSIS_{nhird_name}_DATE_END"])
            start_month = days_to_months(start_days)
            end_month = days_to_months(end_days)
            dose_sum = float(row[f"OSTEOPOROSIS_{nhird_name}_DDSUM"])
            meds.append({
                "drug_class": drug_class,
                "drug_name": nhird_name.lower(),
                "start_month": start_month,
                "end_month": end_month,
                "dose_days": dose_sum,
            })

    # Add chemo/radiation as medication events if present
    if lc_rt:
        rt_start = days_to_months(float(row["LUNG_CA_RT_DATE_FST"]))
        rt_end = days_to_months(float(row["LUNG_CA_RT_DATE_LST"]))
        meds.append({
            "drug_class": "radiation_therapy",
            "start_month": rt_start,
            "end_month": rt_end,
        })
    if lc_ct:
        ct_start = days_to_months(float(row["LUNG_CA_CT_DATE_FST"]))
        ct_end = days_to_months(float(row["LUNG_CA_CT_DATE_LST"]))
        meds.append({
            "drug_class": "chemotherapy_platinum",
            "start_month": ct_start,
            "end_month": ct_end,
        })

    # Comorbidities
    comorbidities = []
    cci_score = 0
    for nhird_col, condition in NHIRD_COMORB_MAP.items():
        if int(row[nhird_col]) == 1:
            date_col = f"{nhird_col}_DATE"
            diag_month = days_to_months(float(row[date_col]))
            comorbidities.append({
                "condition": condition,
                "diagnosed_month": diag_month,
            })
            cci_score += CCI_WEIGHTS.get(nhird_col, 0)

    # Osteoporosis as comorbidity
    if int(row["OSTEOPOROSIS"]) == 1:
        osteo_month = days_to_months(float(row["OSTEOPOROSIS_DATE"]))
        comorbidities.append({
            "condition": "osteoporosis",
            "diagnosed_month": osteo_month,
        })

    # Fractures
    fractures = []
    for prefix, frac_type in [("VF", "vertebral_fracture"),
                               ("HF", "hip_fracture"),
                               ("WF", "wrist_fracture")]:
        if int(row[prefix]) == 1:
            fractures.append({
                "type": frac_type,
                "diagnosed_month": days_to_months(float(row[f"{prefix}_DATE"])),
                "visit_count": int(row[f"{prefix}_CNT"]) if f"{prefix}_CNT" in row.index else 0,
            })
            # Also add as comorbidity for feature engineering
            comorbidities.append({
                "condition": "prior_fracture",
                "diagnosed_month": days_to_months(float(row[f"{prefix}_DATE"])),
            })

    # Observation months: 36 months post-diagnosis (standard NHIRD follow-up)
    observation_months = 36

    # Target label
    label = int(row["path"])

    return {
        "patient_id": "",  # filled by caller
        "age": age,
        "sex": sex,
        "observation_months": observation_months,
        "medications": json.dumps(meds),
        "comorbidities": json.dumps(comorbidities),
        "cci_score": cci_score,
        "lung_cancer_radiation": lc_rt,
        "lung_cancer_chemotherapy": lc_ct,
        "lung_cancer_surgery": lc_op,
        "lung_cancer_visit_count": lc_cnt,
        "lung_cancer_location_count": lc_locations,
        "fractures": json.dumps(fractures),
        "osteoporosis": int(row["OSTEOPOROSIS"]),
        "osteoporosis_therapy": int(row["OSTEOPOROSIS_THERAPY"]),
        "bone_metastasis": label,
    }


def main():
    parser = argparse.ArgumentParser(description="Convert NHIRD CSV to ABSTRAL format")
    parser.add_argument("--csv", required=True, help="Path to 12.31_data_with_relative_days.csv")
    parser.add_argument("--output-dir", default="data/", help="Output directory")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    model_dir = output_dir / "models"
    model_dir.mkdir(exist_ok=True)

    # Read CSV
    print(f"Reading {args.csv}...")
    raw_df = pd.read_csv(args.csv)
    print(f"  {len(raw_df)} patients, {len(raw_df.columns)} columns")
    print(f"  Label distribution: path=0: {sum(raw_df['path']==0)}, path=1: {sum(raw_df['path']==1)}")

    # Convert rows
    print("\nConverting to ABSTRAL format...")
    rows = []
    for i, (_, row) in enumerate(raw_df.iterrows()):
        record = convert_row(row)
        record["patient_id"] = f"P{i:05d}"
        rows.append(record)

    df = pd.DataFrame(rows)
    parquet_path = output_dir / "oncoagent.parquet"
    df.to_parquet(str(parquet_path), index=False)
    print(f"  Saved {len(df)} patients to {parquet_path}")
    print(f"  Columns: {list(df.columns)}")

    # Engineer features
    print("\nEngineering features...")
    from tools.feature_engineer import engineer_features, N_FEATURES, FEATURE_NAMES
    from tools.ops_calculator import compute_ops

    features = []
    labels = []
    for _, row in df.iterrows():
        meds = json.loads(row["medications"])
        comorbidities = json.loads(row["comorbidities"])
        fracture_list = json.loads(row["fractures"])
        record = {
            "patient_id": row["patient_id"],
            "demographics": {
                "age": row["age"],
                "sex": "female" if row["sex"] == 1 else "male",
            },
            "medications": meds,
            "comorbidities": comorbidities,
            "cci_score": row["cci_score"],
            "labs": {},
            "observation_months": row["observation_months"],
            "lung_cancer": {
                "radiation": bool(row["lung_cancer_radiation"]),
                "chemotherapy": bool(row["lung_cancer_chemotherapy"]),
                "surgery": bool(row["lung_cancer_surgery"]),
                "visit_count": row["lung_cancer_visit_count"],
                "location_count": row["lung_cancer_location_count"],
            },
            "fractures": fracture_list,
        }
        ops = compute_ops(record)
        feat = engineer_features(record, ops)
        features.append(feat["feature_vector"])
        labels.append(row["bone_metastasis"])

    X = np.array(features)
    y = np.array(labels)
    print(f"  Feature matrix: {X.shape}")
    print(f"  Label distribution: 0={sum(y==0)}, 1={sum(y==1)}")

    np.save(str(output_dir / "features.npy"), X)
    np.save(str(output_dir / "labels.npy"), y)
    with open(str(output_dir / "feature_columns.json"), "w") as f:
        json.dump(list(FEATURE_NAMES), f)
    print("  Saved features.npy, labels.npy, feature_columns.json")

    # Train models
    print("\nTraining models...")
    from train_models import train_and_evaluate
    train_and_evaluate(X, y, str(model_dir), list(FEATURE_NAMES))

    print(f"\nDone! Real NHIRD data ready at {output_dir}")
    print(f"  Parquet: {parquet_path}")
    print(f"  Features: {output_dir / 'features.npy'} ({X.shape})")
    print(f"  Models: {model_dir}/")


if __name__ == "__main__":
    main()
