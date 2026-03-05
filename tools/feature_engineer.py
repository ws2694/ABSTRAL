"""Feature engineering: patient records → fixed-dimension feature vector.

Converts heterogeneous patient records into a numeric feature vector
suitable for ML model input. Aligned with NHIRD lung cancer data schema.
"""

from __future__ import annotations

import numpy as np
from typing import Optional

# Drug classes matching NHIRD osteoporosis drug columns
DRUG_CLASSES = [
    "bisphosphonate", "calcitonin", "denosumab", "estrogen",
    "raloxifene", "teriparatide",
    "radiation_therapy", "chemotherapy_platinum",
]

# Comorbidity conditions matching NHIRD CCI + fractures
COMORBIDITY_CONDITIONS = [
    "myocardial_infarction", "heart_failure", "peripheral_vascular_disease",
    "cerebrovascular_disease", "dementia", "copd",
    "rheumatic_disease", "peptic_ulcer_disease", "liver_disease",
    "diabetes", "diabetes_with_complications", "hemiplegia",
    "chronic_kidney_disease", "malignancy", "liver_disease_severe", "hiv",
    "osteoporosis", "prior_fracture",
]

# Feature names for interpretability
FEATURE_NAMES = (
    # Demographics (2)
    ["age_normalized", "sex_female"]
    # Drug presence (8)
    + [f"med_{dc}_present" for dc in DRUG_CLASSES]
    # Drug duration months (8)
    + [f"med_{dc}_months" for dc in DRUG_CLASSES]
    # Comorbidity indicators (18)
    + [f"comorb_{c}" for c in COMORBIDITY_CONDITIONS]
    # CCI score (1)
    + ["cci_score"]
    # OPS features (6)
    + ["ops_mean", "ops_max", "ops_min", "ops_final", "ops_slope", "ops_months"]
    # Lung cancer features (5)
    + ["lc_radiation", "lc_chemotherapy", "lc_surgery", "lc_location_count", "lc_visit_count"]
    # Fracture features (3)
    + ["fracture_vertebral", "fracture_hip", "fracture_wrist"]
    # Temporal features (6)
    + ["total_med_events", "unique_drug_classes", "med_density",
       "first_med_month", "last_med_month", "observation_months"]
    # Drug interaction features (4)
    + ["bisph_present", "polypharmacy_count", "protective_drug_count",
       "harmful_drug_count"]
    # Lab placeholders (3) — for future use with clinical data
    + ["alkaline_phosphatase", "calcium_level", "bone_density_tscore"]
)

N_FEATURES = len(FEATURE_NAMES)


def engineer_features(patient_record: dict, ops_summary: Optional[dict] = None) -> dict:
    """Convert a patient record into a feature vector.

    Args:
        patient_record: Raw patient data dict.
        ops_summary: Pre-computed OPS summary (optional; computed if missing).

    Returns:
        Dict with feature_vector (list[float]), feature_names, and metadata.
    """
    features = np.zeros(N_FEATURES, dtype=np.float64)
    meds = patient_record.get("medications", [])
    comorbidities = patient_record.get("comorbidities", [])
    demographics = patient_record.get("demographics", {})
    labs = patient_record.get("labs", {})
    obs_months = patient_record.get("observation_months", 24)

    idx = 0

    # ── Demographics (2) ─────────────────────────────────────────────────
    features[idx] = demographics.get("age", 60) / 100.0
    idx += 1
    features[idx] = 1.0 if str(demographics.get("sex", "")).lower() in ("female", "f") else 0.0
    idx += 1

    # ── Drug presence & duration (8 + 8 = 16) ────────────────────────────
    med_present = {dc: 0 for dc in DRUG_CLASSES}
    med_months = {dc: 0.0 for dc in DRUG_CLASSES}
    for med in meds:
        dc = med.get("drug_class", "").lower().replace(" ", "_")
        if dc in med_present:
            med_present[dc] = 1
            start = med.get("start_month", 0)
            end = med.get("end_month", obs_months)
            med_months[dc] += max(0, end - start)

    for dc in DRUG_CLASSES:
        features[idx] = float(med_present[dc])
        idx += 1
    for dc in DRUG_CLASSES:
        features[idx] = med_months[dc] / max(obs_months, 1)
        idx += 1

    # ── Comorbidity indicators (18) ──────────────────────────────────────
    patient_conditions = set()
    for c in comorbidities:
        cond = c.get("condition", "").lower().replace(" ", "_")
        patient_conditions.add(cond)
        # Also map aliases
        if "liver" in cond and "severe" not in cond:
            patient_conditions.add("liver_disease")
        if "heart" in cond or "chf" in cond or cond == "congestive_heart_failure":
            patient_conditions.add("heart_failure")

    for cond in COMORBIDITY_CONDITIONS:
        features[idx] = 1.0 if cond in patient_conditions else 0.0
        idx += 1

    # ── CCI score (1) ────────────────────────────────────────────────────
    cci = patient_record.get("cci_score", sum(c.get("cci_weight", 0) for c in comorbidities))
    features[idx] = cci / 15.0
    idx += 1

    # ── OPS features (6) ─────────────────────────────────────────────────
    if ops_summary:
        s = ops_summary.get("summary", {})
        features[idx] = s.get("mean_ops", 0.0)
        features[idx + 1] = s.get("max_ops", 0.0)
        features[idx + 2] = s.get("min_ops", 0.0)
        features[idx + 3] = s.get("final_ops", 0.0)
        features[idx + 4] = s.get("slope", 0.0)
        features[idx + 5] = s.get("observation_months", obs_months) / 60.0
    idx += 6

    # ── Lung cancer features (5) ─────────────────────────────────────────
    lc = patient_record.get("lung_cancer", {})
    features[idx] = 1.0 if lc.get("radiation", False) else 0.0
    idx += 1
    features[idx] = 1.0 if lc.get("chemotherapy", False) else 0.0
    idx += 1
    features[idx] = 1.0 if lc.get("surgery", False) else 0.0
    idx += 1
    features[idx] = lc.get("location_count", 0) / 6.0
    idx += 1
    features[idx] = min(lc.get("visit_count", 0), 50) / 50.0
    idx += 1

    # ── Fracture features (3) ────────────────────────────────────────────
    fractures = patient_record.get("fractures", [])
    fx_types = {f.get("type", "") for f in fractures}
    features[idx] = 1.0 if "vertebral_fracture" in fx_types else 0.0
    idx += 1
    features[idx] = 1.0 if "hip_fracture" in fx_types else 0.0
    idx += 1
    features[idx] = 1.0 if "wrist_fracture" in fx_types else 0.0
    idx += 1

    # ── Temporal features (6) ────────────────────────────────────────────
    features[idx] = len(meds)  # total med events
    idx += 1
    features[idx] = len(set(m.get("drug_class", "") for m in meds))  # unique classes
    idx += 1
    features[idx] = len(meds) / max(obs_months, 1)  # density
    idx += 1
    starts = [m.get("start_month", 0) for m in meds]
    features[idx] = min(starts) / max(obs_months, 1) if starts else 0.0  # first
    idx += 1
    ends = [m.get("end_month", 0) for m in meds]
    features[idx] = max(ends) / max(obs_months, 1) if ends else 0.0  # last
    idx += 1
    features[idx] = obs_months / 60.0  # observation normalized
    idx += 1

    # ── Drug interaction features (4) ────────────────────────────────────
    drug_set = set(m.get("drug_class", "").lower().replace(" ", "_") for m in meds)
    features[idx] = 1.0 if "bisphosphonate" in drug_set else 0.0  # bisph present
    idx += 1
    features[idx] = len(drug_set) / 8.0  # polypharmacy
    idx += 1
    protective = {"bisphosphonate", "denosumab", "calcitonin", "raloxifene", "teriparatide"}
    features[idx] = len(drug_set & protective) / 5.0  # protective count
    idx += 1
    harmful = {"chemotherapy_platinum", "radiation_therapy", "estrogen"}
    features[idx] = len(drug_set & harmful) / 3.0  # harmful count
    idx += 1

    # ── Lab placeholders (3) ─────────────────────────────────────────────
    features[idx] = labs.get("alkaline_phosphatase", 0) / 120.0
    idx += 1
    features[idx] = labs.get("calcium_level", 0) / 12.0
    idx += 1
    features[idx] = labs.get("bone_density_tscore", 0) / 4.0
    idx += 1

    assert idx == N_FEATURES, f"Feature index mismatch: {idx} != {N_FEATURES}"

    return {
        "patient_id": patient_record.get("patient_id", "unknown"),
        "feature_vector": features.tolist(),
        "feature_names": FEATURE_NAMES,
        "n_features": N_FEATURES,
    }
