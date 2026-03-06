"""Patient store and sandbox execution environment.

Loads patient data from parquet, provides structured access per patient,
and integrates with ML models and tool functions for deterministic execution.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Optional, List

import numpy as np
import pandas as pd

from tools.feature_engineer import engineer_features
from tools.ops_calculator import compute_ops
from tools.ml_models import EnsemblePredictor


class PatientStore:
    """Manages patient data and provides tool-callable interfaces."""

    def __init__(self, data_path: str, model_dir: str = "data/models"):
        self.data_path = data_path
        self.model_dir = model_dir
        self.df = None
        self.patient_records: dict[str, dict] = {}
        self.patient_ids: list[str] = []
        self.labels: dict[str, int] = {}
        self.ensemble = None
        self._feature_cache: dict[str, dict] = {}
        self._ops_cache: dict[str, dict] = {}
        self._benchmark_features: dict[str, np.ndarray] = {}  # pid → 123-dim vector

    @classmethod
    def load(cls, data_path: str, model_dir: str = "data/models") -> "PatientStore":
        store = cls(data_path, model_dir)
        store._load_data()
        store._load_models()
        return store

    def _load_data(self):
        self.df = pd.read_parquet(self.data_path)

        # Identify columns
        pid_col = _find_column(self.df, ["patient_id", "id", "pid", "subject_id"])
        if pid_col is None:
            self.df["patient_id"] = [f"P{i:05d}" for i in range(len(self.df))]
            pid_col = "patient_id"

        label_col = _find_column(self.df, ["bone_metastasis", "label", "target",
                                            "metastasis", "bone_met", "outcome", "y",
                                            "MST", "METS_BONE", "path"])
        if label_col is None:
            raise ValueError("Cannot find label column in data.")

        for _, row in self.df.iterrows():
            pid = str(row[pid_col])
            record = self._row_to_record(row)
            self.patient_records[pid] = record
            self.patient_ids.append(pid)
            self.labels[pid] = int(row[label_col])

        # Load pre-computed benchmark features (123-dim, matches trained models)
        data_dir = Path(self.data_path).parent
        features_path = data_dir / "features.npy"
        if features_path.exists():
            all_features = np.load(str(features_path))
            if len(all_features) == len(self.patient_ids):
                for i, pid in enumerate(self.patient_ids):
                    self._benchmark_features[pid] = all_features[i]
                print(f"  Loaded benchmark features: {all_features.shape[1]}-dim for {len(self.patient_ids)} patients")

    def _load_models(self):
        self.ensemble = EnsemblePredictor(model_dir=self.model_dir)
        self.ensemble.load()

    def _row_to_record(self, row: pd.Series) -> dict:
        """Convert DataFrame row to patient record dict."""
        record = {"patient_id": str(row.get("patient_id", "unknown"))}

        # Demographics
        demographics = {}
        for col, key in [("age", "age"), ("sex", "sex"), ("gender", "sex"),
                         ("bmi", "bmi"), ("smoking", "smoking"),
                         ("smoking_history", "smoking")]:
            if col in self.df.columns and pd.notna(row.get(col)):
                val = row[col]
                if key == "sex" and isinstance(val, (int, float)):
                    val = "female" if val == 1 else "male"
                if key == "smoking" and isinstance(val, (int, float)):
                    val = bool(val)
                demographics[key] = val
        record["demographics"] = demographics

        # Medications
        meds = []
        if "medications" in self.df.columns and pd.notna(row.get("medications")):
            raw = row["medications"]
            if isinstance(raw, str):
                meds = json.loads(raw)
            elif isinstance(raw, list):
                meds = raw
        else:
            for c in self.df.columns:
                if (c.startswith("med_") or c.startswith("drug_")) and pd.notna(row.get(c)) and row[c]:
                    drug_class = c.replace("med_", "").replace("drug_", "")
                    obs = int(row.get("observation_months", 24)) if "observation_months" in self.df.columns else 24
                    meds.append({"drug_class": drug_class, "start_month": 0, "end_month": obs})
        record["medications"] = meds

        # Comorbidities
        comorbidities = []
        if "comorbidities" in self.df.columns and pd.notna(row.get("comorbidities")):
            raw = row["comorbidities"]
            if isinstance(raw, str):
                comorbidities = json.loads(raw)
            elif isinstance(raw, list):
                comorbidities = raw
        else:
            for c in self.df.columns:
                if (c.startswith("comorb_") or c.startswith("comorbidity_")) and pd.notna(row.get(c)) and row[c]:
                    condition = c.replace("comorb_", "").replace("comorbidity_", "")
                    comorbidities.append({"condition": condition, "diagnosed_month": 0})
        record["comorbidities"] = comorbidities

        # CCI
        for col in ["cci", "cci_score", "charlson_comorbidity_index"]:
            if col in self.df.columns and pd.notna(row.get(col)):
                record["cci_score"] = int(row[col])
                break

        # Labs
        labs = {}
        lab_cols = ["alkaline_phosphatase", "calcium_level", "phosphorus_level",
                    "vitamin_d_level", "creatinine", "hemoglobin", "albumin",
                    "ldh", "psa_or_tumor_marker", "bone_density_tscore"]
        for lc in lab_cols:
            if lc in self.df.columns and pd.notna(row.get(lc)):
                labs[lc] = float(row[lc])
        record["labs"] = labs

        # Observation months
        for col in ["observation_months", "follow_up_months", "obs_months"]:
            if col in self.df.columns and pd.notna(row.get(col)):
                record["observation_months"] = int(row[col])
                break
        else:
            record["observation_months"] = 24

        # Lung cancer info
        lc = {}
        lc_map = [("lung_cancer_radiation", "radiation"),
                   ("lung_cancer_chemotherapy", "chemotherapy"),
                   ("lung_cancer_surgery", "surgery")]
        for col, key in lc_map:
            if col in self.df.columns and pd.notna(row.get(col)):
                lc[key] = bool(int(row[col]))
        for col, key in [("lung_cancer_visit_count", "visit_count"),
                          ("lung_cancer_location_count", "location_count")]:
            if col in self.df.columns and pd.notna(row.get(col)):
                lc[key] = int(row[col])
        if lc:
            record["lung_cancer"] = lc

        # Fractures
        if "fractures" in self.df.columns and pd.notna(row.get("fractures")):
            raw = row["fractures"]
            if isinstance(raw, str):
                record["fractures"] = json.loads(raw)
            elif isinstance(raw, list):
                record["fractures"] = raw

        return record

    def get(self, patient_id: str) -> dict:
        """Get raw patient record."""
        if patient_id not in self.patient_records:
            raise KeyError(f"Patient {patient_id} not found")
        return self.patient_records[patient_id]

    def get_structured(self, patient_id: str) -> dict:
        """Get structured patient record with computed summaries."""
        record = self.get(patient_id)
        ops = self.get_ops(patient_id)

        drug_classes = list(set(
            m.get("drug_class", "") for m in record.get("medications", [])
        ))
        conditions = list(set(
            c.get("condition", "") for c in record.get("comorbidities", [])
        ))

        return {
            "patient_id": patient_id,
            "demographics": record.get("demographics", {}),
            "medication_timeline": record.get("medications", []),
            "active_drug_classes": drug_classes,
            "comorbidity_history": record.get("comorbidities", []),
            "conditions": conditions,
            "cci_score": record.get("cci_score", len(conditions)),
            "labs": record.get("labs", {}),
            "observation_months": record.get("observation_months", 24),
            "ops_summary": ops.get("summary", {}),
            "ops_risk_category": ops.get("risk_category", "unknown"),
            "medication_event_count": len(record.get("medications", []))
        }

    def get_ops(self, patient_id: str) -> dict:
        """Get OPS trajectory (cached)."""
        if patient_id not in self._ops_cache:
            record = self.get(patient_id)
            self._ops_cache[patient_id] = compute_ops(record)
        return self._ops_cache[patient_id]

    def get_features(self, patient_id: str) -> dict:
        """Get engineered feature vector (cached)."""
        if patient_id not in self._feature_cache:
            record = self.get(patient_id)
            ops = self.get_ops(patient_id)
            self._feature_cache[patient_id] = engineer_features(record, ops)
        return self._feature_cache[patient_id]

    def predict(self, patient_id: str, model: str = "ensemble",
                weights: Optional[list] = None) -> dict:
        """Run ML prediction for a patient using benchmark features."""
        if patient_id in self._benchmark_features:
            # Use pre-computed 123-dim features that match trained models
            features_raw = self._benchmark_features[patient_id]
            return self.ensemble.predict(features_raw, model=model, weights=weights)
        else:
            # Fallback to feature engineer (may have dimension mismatch)
            feat = self.get_features(patient_id)
            return self.ensemble.predict(feat["feature_vector"], model=model, weights=weights)

    def get_label(self, patient_id: str) -> int:
        """Get ground truth label."""
        return self.labels[patient_id]

    def precompute_all(self, patient_ids: list[str]) -> dict[str, dict[str, dict]]:
        """Pre-compute all tool results for a list of patients.

        Returns {pid: {tool_name: result_dict}} for all 4 tools.
        Results are deterministic and cached, so this is safe to call once
        before the RUN step and inject into agent prompts.
        """
        from tools.drug_kb import DRUG_KB

        precomputed = {}
        for pid in patient_ids:
            record = self.get(pid)
            drug_classes = list(set(
                m.get("drug_class", "") for m in record.get("medications", [])
            ))

            precomputed[pid] = {
                "get_patient_features": self.get_structured(pid),
                "predict_risk": self.predict(pid),
                "compute_ops_trajectory": self.get_ops(pid),
                "lookup_drug_interaction": DRUG_KB.lookup(drug_classes) if drug_classes else {"interactions": [], "note": "No drug classes provided"},
            }
        return precomputed

    def stratified_sample(self, n: int, seed: int = 42) -> list[str]:
        """Sample n patient IDs with stratified sampling on label."""
        rng = random.Random(seed)
        pos = [pid for pid in self.patient_ids if self.labels[pid] == 1]
        neg = [pid for pid in self.patient_ids if self.labels[pid] == 0]

        pos_ratio = len(pos) / len(self.patient_ids)
        n_pos = max(1, int(n * pos_ratio))
        n_neg = n - n_pos

        sampled_pos = rng.sample(pos, min(n_pos, len(pos)))
        sampled_neg = rng.sample(neg, min(n_neg, len(neg)))

        result = sampled_pos + sampled_neg
        rng.shuffle(result)
        return result


def _find_column(df: pd.DataFrame, candidates: list) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None
