"""Tool executor — routes tool_use blocks from Claude API to Python functions.

This is the bridge between Claude's tool calls and the deterministic tool
substrate. Every tool call produces the same output for the same input.
"""

from tools.drug_kb import DRUG_KB
from tools.ops_calculator import compute_ops


def execute_tool(tool_name: str, tool_input: dict, patient_store) -> dict:
    """Execute a tool call and return the result as a JSON-serializable dict.

    Args:
        tool_name: Name of the tool (from ONCO_TOOLS).
        tool_input: Input parameters from Claude's tool_use block.
        patient_store: PatientStore instance with loaded data and models.

    Returns:
        Dict result that will be sent back to Claude as tool_result content.
    """
    if tool_name == "predict_risk":
        patient_id = tool_input["patient_id"]
        model = tool_input.get("model", "ensemble")
        weights = tool_input.get("weights")
        return patient_store.predict(patient_id, model=model, weights=weights)

    elif tool_name == "compute_ops_trajectory":
        patient_id = tool_input["patient_id"]
        return patient_store.get_ops(patient_id)

    elif tool_name == "lookup_drug_interaction":
        drug_classes = tool_input["drug_classes"]
        return DRUG_KB.lookup(drug_classes)

    elif tool_name == "get_patient_features":
        patient_id = tool_input["patient_id"]
        return patient_store.get_structured(patient_id)

    else:
        return {"error": f"Unknown tool: {tool_name}"}
