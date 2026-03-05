#!/usr/bin/env python3
"""Data processor — transforms NHIRD lung cancer data into ABSTRAL format.

Aligned with team benchmark (2.17 Benchmarks.py):
  - Events are only kept if flag=1 AND date<0 (before lung CA diagnosis)
  - Post-diagnosis data is zeroed out to prevent leakage
  - Raw filtered features are stored for ML training (matching benchmark)
  - Structured patient records are stored for agent system tools

Usage:
    python data_processor.py --input "12.31_data_with_relative_days 2.csv"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# ── Constants ─────────────────────────────────────────────────────────────────

SENTINEL_THRESHOLD = 30000
DAYS_PER_MONTH = 30.44

# Drug column prefix → (drug_class, drug_name)
DRUG_MAP = {
    "OSTEOPOROSIS_ALENDRONATE": ("bisphosphonate", "alendronate"),
    "OSTEOPOROSIS_CALCITONIN": ("calcitonin", "calcitonin"),
    "OSTEOPOROSIS_DENOSUMAB": ("denosumab", "denosumab"),
    "OSTEOPOROSIS_ETIDRONATE": ("bisphosphonate", "etidronate"),
    "OSTEOPOROSIS_ESTROGEN": ("estrogen", "estrogen"),
    "OSTEOPOROSIS_IBANDRONIC": ("bisphosphonate", "ibandronic_acid"),
    "OSTEOPOROSIS_PAMIDRONATE": ("bisphosphonate", "pamidronate"),
    "OSTEOPOROSIS_RALOXIFENE": ("raloxifene", "raloxifene"),
    "OSTEOPOROSIS_RISEDRONATE": ("bisphosphonate", "risedronate"),
    "OSTEOPOROSIS_TERIPARATIDE": ("teriparatide", "teriparatide"),
    "OSTEOPOROSIS_ZOLEDRONIC": ("bisphosphonate", "zoledronic_acid"),
}

# CCI condition column → (condition_name, cci_weight)
CCI_MAP = {
    "MI": ("myocardial_infarction", 1),
    "CHF": ("heart_failure", 1),
    "PVD": ("peripheral_vascular_disease", 1),
    "CD": ("cerebrovascular_disease", 1),
    "DEM": ("dementia", 1),
    "CPD": ("copd", 1),
    "RD": ("rheumatic_disease", 1),
    "PUD": ("peptic_ulcer_disease", 1),
    "MLD": ("liver_disease", 1),
    "DIABETES_NO_CC": ("diabetes", 1),
    "DIABETES_CC": ("diabetes_with_complications", 2),
    "HOP": ("hemiplegia", 2),
    "RENAL": ("chronic_kidney_disease", 2),
    "MA": ("malignancy", 2),
    "MSLD": ("liver_disease_severe", 3),
    "HIV": ("hiv", 6),
}

FRACTURE_MAP = {
    "VF": "vertebral_fracture",
    "HF": "hip_fracture",
    "WF": "wrist_fracture",
}

# ── Drug column groups (matching benchmark structure) ─────────────────────────

DRUG_PREFIXES = [
    "OSTEOPOROSIS_ALENDRONATE", "OSTEOPOROSIS_CALCITONIN", "OSTEOPOROSIS_DENOSUMAB",
    "OSTEOPOROSIS_ETIDRONATE", "OSTEOPOROSIS_ESTROGEN", "OSTEOPOROSIS_IBANDRONIC",
    "OSTEOPOROSIS_PAMIDRONATE", "OSTEOPOROSIS_RALOXIFENE", "OSTEOPOROSIS_RISEDRONATE",
    "OSTEOPOROSIS_TERIPARATIDE", "OSTEOPOROSIS_ZOLEDRONIC",
]

CCI_COLUMNS = [
    "MI", "CHF", "PVD", "CD", "DEM", "CPD", "RD", "PUD",
    "MLD", "DIABETES_NO_CC", "DIABETES_CC", "HOP", "RENAL", "MA", "MSLD", "HIV",
]

FRACTURE_COLUMNS = ["VF", "HF", "WF"]


# ── Benchmark-style filtering ────────────────────────────────────────────────

def apply_filter(df: pd.DataFrame, flag_col: str, date_col: str,
                 extra_cols: list = None) -> pd.DataFrame:
    """Apply benchmark filtering: keep event only if flag=1 AND date<0.

    Matches team benchmark (2.17 Benchmarks.py) logic exactly.
    """
    if flag_col not in df.columns or date_col not in df.columns:
        return df

    extra_cols = extra_cols or []
    valid_mask = (df[flag_col] == 1) & (df[date_col] < 0)

    all_cols = [flag_col, date_col] + [c for c in extra_cols if c in df.columns]
    for col in all_cols:
        df.loc[~valid_mask, col] = 0

    return df


def filter_benchmark_style(df: pd.DataFrame) -> pd.DataFrame:
    """Apply benchmark-consistent filtering to all event modules."""
    filtered = df.copy()

    # Osteoporosis diagnosis
    if "OSTEOPOROSIS" in filtered.columns:
        filtered = apply_filter(
            filtered, "OSTEOPOROSIS", "OSTEOPOROSIS_DATE",
            extra_cols=["OSTEOPOROSIS_CNT"]
        )

    # Osteoporosis drugs (11 drugs, each with flag + date + extras)
    for prefix in DRUG_PREFIXES:
        fst_col = f"{prefix}_DATE_FST"
        extra = [f"{prefix}_DATE_END", f"{prefix}_DDSUM", f"{prefix}_DQSUM"]
        if prefix in filtered.columns and fst_col in filtered.columns:
            filtered = apply_filter(filtered, prefix, fst_col, extra_cols=extra)

    # CCI complications (16 conditions)
    for col in CCI_COLUMNS:
        date_col = f"{col}_DATE"
        if col in filtered.columns and date_col in filtered.columns:
            filtered = apply_filter(filtered, col, date_col)

    # Fractures (3 types)
    for fx in FRACTURE_COLUMNS:
        date_col = f"{fx}_DATE"
        extra = [f"{fx}_CNT"]
        # Surgery/confirmation columns
        if fx == "VF":
            extra += ["VFOP", "VFOP_DATE", "VF_DIG", "VF_DIG_DATE", "VF_TIME"]
        elif fx == "HF":
            extra += ["HF_OP", "HF_OP_DATE", "HF_DIG", "HF_DIG_DATE", "HF_TIME"]
        elif fx == "WF":
            extra += ["WFOP", "WFOP_DATE", "WF_DIG", "WF_DIG_DATE", "WF_TIME"]
        if fx in filtered.columns and date_col in filtered.columns:
            filtered = apply_filter(filtered, fx, date_col, extra_cols=extra)

    # NOTE: Lung CA treatment features (RT, CT, OP) in the static columns
    # are NOT filtered, matching team benchmark behavior. The benchmark treats
    # columns 0-7 as static features that are kept as-is.

    return filtered


def get_benchmark_columns() -> tuple:
    """Get the exact benchmark column layout (2.17 format, 124 columns).

    Returns (feature_cols, target_col) matching team benchmark exactly.
    Static features (0-7) are NOT filtered — matching benchmark behavior.
    """
    static = ["ID_SEX", "ID_AGE_Y2001", "LUNG_CA_DATE", "LUNG_CA_CNT",
              "LUNG_CA_RT", "LUNG_CA_RT_DATE_FST", "LUNG_CA_RT_DATE_LST", "LUNG_CA_CT"]

    osteo_diag = ["OSTEOPOROSIS", "OSTEOPOROSIS_DATE", "OSTEOPOROSIS_CNT"]

    drug_suffixes = ["", "_DATE_FST", "_DATE_END", "_DDSUM", "_DQSUM"]
    drugs = [f"{p}{s}" for p in DRUG_PREFIXES for s in drug_suffixes]

    therapy = ["OSTEOPOROSIS_THERAPY"]

    cci_names = ["MI", "CHF", "PVD", "CD", "DEM", "CPD", "RD", "PUD",
                 "MLD", "DIABETES_NO_CC", "DIABETES_CC", "HOP", "RENAL", "MA", "MSLD", "HIV"]
    cci = []
    for c in cci_names:
        cci.extend([c, f"{c}_DATE"])

    fx_groups = [
        ["VF", "VF_DATE", "VF_CNT", "VFOP", "VFOP_DATE", "VF_DIG", "VF_DIG_DATE", "VF_TIME"],
        ["HF", "HF_DATE", "HF_CNT", "HF_OP", "HF_OP_DATE", "HF_DIG", "HF_DIG_DATE", "HF_TIME"],
        ["WF", "WF_DATE", "WF_CNT", "WFOP", "WFOP_DATE", "WF_DIG", "WF_DIG_DATE", "WF_TIME"],
    ]
    fractures = [c for g in fx_groups for c in g]

    feature_cols = static + osteo_diag + drugs + therapy + cci + fractures
    return feature_cols, "MST"


# ── Structured record building (for agent tools) ─────────────────────────────

def _days_to_months(days: float) -> float:
    return days / DAYS_PER_MONTH


def row_to_patient_record(row: pd.Series, idx: int) -> dict:
    """Convert a filtered row to structured patient record for agent tools."""
    record = {"patient_id": f"P{idx:05d}"}

    # Target
    target = int(row.get("MST", 0))
    record["bone_metastasis"] = target

    # Demographics
    sex_val = row.get("ID_SEX", 0)
    sex = "female" if sex_val == 0 else "male"
    age = float(row.get("ID_AGE_Y2001", 60))
    record["demographics"] = {"sex": sex, "age": age}

    # Medications (already filtered — only pre-diagnosis events remain)
    medications = []
    for prefix, (drug_class, drug_name) in DRUG_MAP.items():
        if row.get(prefix, 0) != 1:
            continue
        fst = row.get(f"{prefix}_DATE_FST", 0)
        end = row.get(f"{prefix}_DATE_END", 0)
        ddsum = row.get(f"{prefix}_DDSUM", 0)
        dqsum = row.get(f"{prefix}_DQSUM", 0)
        medications.append({
            "drug_class": drug_class,
            "drug_name": drug_name,
            "start_month": round(_days_to_months(fst), 1),
            "end_month": round(_days_to_months(end), 1) if end else 0,
            "duration_days": float(ddsum) if pd.notna(ddsum) else 0,
            "quantity": float(dqsum) if pd.notna(dqsum) else 0,
        })
    record["medications"] = medications

    # Comorbidities (already filtered)
    comorbidities = []
    cci_score = 0
    for col, (condition, weight) in CCI_MAP.items():
        if row.get(col, 0) != 1:
            continue
        date_val = row.get(f"{col}_DATE", 0)
        comorbidities.append({
            "condition": condition,
            "diagnosed_month": round(_days_to_months(date_val), 1),
            "cci_weight": weight,
        })
        cci_score += weight

    if row.get("OSTEOPOROSIS", 0) == 1:
        date_val = row.get("OSTEOPOROSIS_DATE", 0)
        comorbidities.append({
            "condition": "osteoporosis",
            "diagnosed_month": round(_days_to_months(date_val), 1),
            "cci_weight": 0,
        })

    record["comorbidities"] = comorbidities
    record["cci_score"] = cci_score

    # Fractures (already filtered)
    fractures = []
    for col, fx_name in FRACTURE_MAP.items():
        if row.get(col, 0) == 1:
            date_val = row.get(f"{col}_DATE", 0)
            fractures.append({
                "type": fx_name,
                "month": round(_days_to_months(date_val), 1),
                "count": int(row.get(f"{col}_CNT", 0)) if pd.notna(row.get(f"{col}_CNT")) else 1,
            })
    if fractures:
        comorbidities.append({
            "condition": "prior_fracture",
            "diagnosed_month": min(f["month"] for f in fractures),
            "cci_weight": 0,
        })
    record["fractures"] = fractures

    # Lung cancer staging (known at diagnosis)
    lc_locations = sum(1 for i in range(1, 7)
                       if f"LUNG_CA_LOCATION{i}" in row.index and row.get(f"LUNG_CA_LOCATION{i}", 0) == 1)
    record["lung_cancer"] = {
        "visit_count": 0,
        "radiation": bool(row.get("LUNG_CA_RT", 0)),
        "chemotherapy": bool(row.get("LUNG_CA_CT", 0)),
        "surgery": bool(row.get("LUNG_CA_OP", 0)),
        "location_count": lc_locations,
    }

    record["observation_months"] = 24
    record["labs"] = {}

    return record


# ── Dataset Processing ────────────────────────────────────────────────────────

def process_dataset(input_path: str, output_path: str) -> pd.DataFrame:
    """Process NHIRD dataset → ABSTRAL parquet.

    Produces two outputs:
      1. {output_path} — parquet with raw filtered features + structured records
      2. {output_dir}/features.npy + labels.npy — for direct ML training
    """
    input_path = Path(input_path)
    print(f"Reading {input_path}...")

    if input_path.suffix in (".xlsx", ".xls"):
        df = pd.read_excel(input_path)
    elif input_path.suffix == ".csv":
        df = pd.read_csv(input_path, low_memory=False)
    elif input_path.suffix == ".parquet":
        df = pd.read_parquet(input_path)
    else:
        raise ValueError(f"Unsupported format: {input_path.suffix}")

    print(f"  Shape: {df.shape[0]} patients x {df.shape[1]} columns")

    # Step 1: Apply benchmark-style filtering (date<0 rule)
    print("  Applying benchmark filtering (flag=1 AND date<0)...")
    filtered = filter_benchmark_style(df)

    # Step 2: Use exact benchmark column layout (123 features + 1 target)
    feature_cols, target_col = get_benchmark_columns()
    missing = [c for c in feature_cols + [target_col] if c not in filtered.columns]
    if missing:
        raise ValueError(f"Missing benchmark columns: {missing}")

    y = filtered[target_col].values.astype(int)
    X = filtered[feature_cols].values.astype(float)
    X = np.nan_to_num(X, nan=0.0)

    print(f"  Features: {len(feature_cols)} columns (benchmark-aligned)")
    print(f"  Target: 0={np.sum(y==0)}, 1={np.sum(y==1)} ({np.mean(y):.1%} positive)")

    # Step 4: Build structured records for agent tools
    records = []
    for i, row in filtered.iterrows():
        idx = int(i) if isinstance(i, (int, np.integer)) else i
        rec = row_to_patient_record(row, idx)
        records.append(rec)

    # Step 5: Build output parquet (structured records + metadata)
    out_rows = []
    for rec in records:
        flat = {
            "patient_id": rec["patient_id"],
            "bone_metastasis": rec["bone_metastasis"],
            "age": rec["demographics"].get("age", 60),
            "sex": rec["demographics"].get("sex", "unknown"),
            "observation_months": rec["observation_months"],
            "cci_score": rec["cci_score"],
            "medications": json.dumps(rec["medications"]),
            "comorbidities": json.dumps(rec["comorbidities"]),
            "labs": json.dumps(rec["labs"]),
            "medication_count": len(rec["medications"]),
            "comorbidity_count": len(rec["comorbidities"]),
            "fracture_count": len(rec.get("fractures", [])),
            "lung_ca_radiation": rec.get("lung_cancer", {}).get("radiation", False),
            "lung_ca_chemotherapy": rec.get("lung_cancer", {}).get("chemotherapy", False),
            "lung_ca_surgery": rec.get("lung_cancer", {}).get("surgery", False),
            "lung_ca_visit_count": rec.get("lung_cancer", {}).get("visit_count", 0),
            "lung_ca_location_count": rec.get("lung_cancer", {}).get("location_count", 0),
        }
        out_rows.append(flat)

    out_df = pd.DataFrame(out_rows)

    # Save parquet (structured records for agent system)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(str(output_path), index=False)

    # Save raw features + labels (for ML training, matching benchmark)
    data_dir = output_path.parent
    np.save(str(data_dir / "features.npy"), X)
    np.save(str(data_dir / "labels.npy"), y)
    # Save feature column names
    with open(str(data_dir / "feature_columns.json"), "w") as f:
        json.dump(feature_cols, f)

    # Summary
    print(f"\nSaved {len(out_df)} patients → {output_path}")
    print(f"  Raw features: {data_dir}/features.npy ({X.shape})")
    print(f"  Labels: {data_dir}/labels.npy")
    print(f"  Feature columns: {data_dir}/feature_columns.json ({len(feature_cols)} cols)")
    print(f"  Target: 0={np.sum(y==0)}, 1={np.sum(y==1)} ({np.mean(y):.1%} positive)")
    print(f"  Age: {out_df['age'].mean():.1f} +/- {out_df['age'].std():.1f}")
    print(f"  CCI: {out_df['cci_score'].mean():.1f} (max {out_df['cci_score'].max()})")

    return out_df


def main():
    parser = argparse.ArgumentParser(description="Process NHIRD data → ABSTRAL parquet")
    parser.add_argument("--input", required=True, help="Input CSV/Excel/Parquet path")
    parser.add_argument("--output", default="data/oncoagent.parquet", help="Output parquet path")
    args = parser.parse_args()

    if not Path(args.input).exists():
        print(f"Error: {args.input} not found")
        sys.exit(1)

    process_dataset(args.input, args.output)
    print(f"\nNext: python train_models.py --data-dir data/")


if __name__ == "__main__":
    main()
