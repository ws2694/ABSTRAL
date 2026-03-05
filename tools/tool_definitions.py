"""Tool definitions passed to every Claude API call.

These define the tools that agents can call during execution.
The actual implementations live in the respective tool modules.
"""

ONCO_TOOLS = [
    {
        "name": "predict_risk",
        "description": (
            "Run ML ensemble on patient feature vector. Returns risk scores "
            "from MLP, XGBoost, RF and weighted ensemble. Optionally specify "
            "a single model or custom weights."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "patient_id": {
                    "type": "string",
                    "description": "Patient identifier"
                },
                "model": {
                    "type": "string",
                    "enum": ["mlp", "xgb", "rf", "ensemble"],
                    "description": "Which model to use. Defaults to ensemble."
                },
                "weights": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": (
                        "Optional weights for ensemble [mlp_w, xgb_w, rf_w]. "
                        "Must sum to 1.0. Only used when model=ensemble."
                    )
                }
            },
            "required": ["patient_id"]
        }
    },
    {
        "name": "compute_ops_trajectory",
        "description": (
            "Compute Osteoporosis Propensity Score at each month in patient "
            "history. Returns list of (month, ops_score) pairs, trajectory "
            "summary statistics (mean, slope, max, min), and risk category."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "patient_id": {
                    "type": "string",
                    "description": "Patient identifier"
                }
            },
            "required": ["patient_id"]
        }
    },
    {
        "name": "lookup_drug_interaction",
        "description": (
            "Look up known interactions between drug classes relevant to bone "
            "health. Returns mechanism of action, net bone effect (protective / "
            "harmful / neutral / complex), and clinical significance."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "drug_classes": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "List of drug class names to check interactions for. "
                        "E.g. ['bisphosphonate', 'glucocorticoid']"
                    )
                }
            },
            "required": ["drug_classes"]
        }
    },
    {
        "name": "get_patient_features",
        "description": (
            "Return structured patient record including: medication timeline "
            "(drug classes with start/end dates), comorbidity history (ICD codes "
            "with CCI score), demographics (age, sex), lab values, and computed "
            "feature summary."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "patient_id": {
                    "type": "string",
                    "description": "Patient identifier"
                }
            },
            "required": ["patient_id"]
        }
    }
]


def get_tools_for_agent(tool_names: list[str]) -> list[dict]:
    """Return the subset of ONCO_TOOLS matching the given names."""
    return [t for t in ONCO_TOOLS if t["name"] in tool_names]
