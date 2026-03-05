"""Osteoporosis Propensity Score (OPS) trajectory calculator.

Computes a monthly OPS score for each patient based on their medication
history, comorbidities, demographics, and risk factors. Deterministic.

The OPS encodes the bone microenvironment state over time — the "seed and
soil" hypothesis: the OPS trajectory captures whether the bone environment
is becoming more hospitable to metastatic colonization.
"""

import numpy as np

# Risk factor weights for OPS computation
OPS_WEIGHTS = {
    # Medications (monthly contribution while active)
    "bisphosphonate": -0.15,      # protective (but paradoxically signals risk)
    "glucocorticoid": 0.25,       # harmful
    "aromatase_inhibitor": 0.20,  # harmful
    "denosumab": -0.20,           # protective
    "chemotherapy_platinum": 0.15, # harmful
    "proton_pump_inhibitor": 0.05, # mildly harmful
    "calcium_vitamin_d": -0.05,    # mildly protective
    "thyroid_hormone": 0.08,       # mildly harmful if over-replaced

    # Comorbidity baseline contributions
    "osteoporosis": 0.30,
    "copd": 0.15,                  # IL-6 → osteoclast pathway
    "diabetes": 0.10,
    "rheumatoid_arthritis": 0.20,  # inflammatory
    "chronic_kidney_disease": 0.15,
    "hyperthyroidism": 0.12,

    # Demographics (one-time baseline)
    "age_over_65": 0.20,
    "age_over_75": 0.35,
    "female": 0.10,
    "low_bmi": 0.08,               # BMI < 18.5
    "smoking_history": 0.12,
}

# Decay factor: how fast medication effects decay after discontinuation
MEDICATION_DECAY = 0.7  # effect halves roughly every 2 months after stopping


def compute_ops(patient_record: dict) -> dict:
    """Compute OPS trajectory for a patient.

    Args:
        patient_record: Dict with keys:
            - medications: list of {drug_class, start_month, end_month}
            - comorbidities: list of {condition, diagnosed_month}
            - demographics: {age, sex, bmi, smoking}
            - observation_months: int (total months of observation)

    Returns:
        Dict with trajectory, summary stats, and risk category.
    """
    obs_months = patient_record.get("observation_months", 24)
    meds = patient_record.get("medications", [])
    comorbidities = patient_record.get("comorbidities", [])
    demographics = patient_record.get("demographics", {})

    # Compute baseline OPS from demographics
    baseline = 0.0
    age = demographics.get("age", 60)
    if age >= 75:
        baseline += OPS_WEIGHTS["age_over_75"]
    elif age >= 65:
        baseline += OPS_WEIGHTS["age_over_65"]
    if demographics.get("sex", "").lower() == "female":
        baseline += OPS_WEIGHTS["female"]
    if demographics.get("bmi", 22) < 18.5:
        baseline += OPS_WEIGHTS["low_bmi"]
    if demographics.get("smoking", False):
        baseline += OPS_WEIGHTS["smoking_history"]

    # Compute comorbidity contributions (active from diagnosis month onward)
    comorbidity_onset = {}
    for c in comorbidities:
        cond = c.get("condition", "").lower().replace(" ", "_")
        month = c.get("diagnosed_month", 0)
        if cond in OPS_WEIGHTS:
            comorbidity_onset[cond] = month

    # Build monthly trajectory
    trajectory = []
    for month in range(obs_months):
        score = baseline

        # Add comorbidity contributions
        for cond, onset in comorbidity_onset.items():
            if month >= onset:
                score += OPS_WEIGHTS[cond]

        # Add medication contributions
        for med in meds:
            drug_class = med.get("drug_class", "").lower().replace(" ", "_")
            start = med.get("start_month", 0)
            end = med.get("end_month", obs_months)

            if drug_class not in OPS_WEIGHTS:
                continue

            if start <= month <= end:
                # Active medication
                score += OPS_WEIGHTS[drug_class]
            elif month > end:
                # Decayed effect after discontinuation
                months_since_stop = month - end
                decay = MEDICATION_DECAY ** months_since_stop
                score += OPS_WEIGHTS[drug_class] * decay

        # Clamp to [0, 1]
        score = max(0.0, min(1.0, score))
        trajectory.append({"month": month, "ops_score": round(score, 4)})

    # Compute summary statistics
    scores = [t["ops_score"] for t in trajectory]
    scores_arr = np.array(scores)

    if len(scores) >= 2:
        slope = float(np.polyfit(range(len(scores)), scores, 1)[0])
    else:
        slope = 0.0

    # Risk categorization
    mean_score = float(np.mean(scores_arr))
    max_score = float(np.max(scores_arr))
    if mean_score >= 0.6 or (slope > 0.01 and max_score >= 0.5):
        risk_category = "high"
    elif mean_score >= 0.35 or max_score >= 0.5:
        risk_category = "moderate"
    else:
        risk_category = "low"

    return {
        "patient_id": patient_record.get("patient_id", "unknown"),
        "trajectory": trajectory,
        "summary": {
            "mean_ops": round(mean_score, 4),
            "max_ops": round(max_score, 4),
            "min_ops": round(float(np.min(scores_arr)), 4),
            "final_ops": round(scores[-1], 4) if scores else 0.0,
            "slope": round(slope, 6),
            "observation_months": obs_months
        },
        "risk_category": risk_category
    }
