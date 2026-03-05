## ABS_0: clinical-agent-builder / SKILL.md (initial version)

This is the Agent Builder Skill for bone metastasis prediction in lung cancer patients.
It is used by the ABSTRAL meta-loop to construct agent systems.

---

### K: Domain Knowledge

- **Seed and soil hypothesis**: The OPS (Osteoporosis Propensity Score) trajectory encodes the bone microenvironment state over time. A rising OPS signals the bone becoming more hospitable to metastatic colonization.
- **Bisphosphonate paradox**: Bisphosphonate ABSENCE in elderly/high-risk patients may indicate undertreated bone disease, but is NOT the paradox. The paradox applies only when bisphosphonates are PRESENT - their prescription signals existing bone disease history, making presence a risk marker despite protective intent. Agent must reason: prescription intent ≠ bone health outcome. <!-- [Evidence: trace P01666, iter 0] -->
- **Glucocorticoid + bisphosphonate co-prescription**: Partially antagonistic effect. Glucocorticoid suppresses osteoblast formation; bisphosphonate inhibits osteoclast resorption. Net: bone formation impaired more than resorption is suppressed. Co-prescription signals high-risk profile (bone disease + inflammatory/autoimmune condition).
- **CCI ≥ 6**: High comorbidity burden. Patients with Charlson Comorbidity Index ≥ 6 require comorbidity cascade analysis — interactions between conditions may amplify bone risk beyond individual effects.
- **Temporal features are more predictive than static demographics**: The trajectory (slope, acceleration) of OPS and medication changes over time carries more signal than baseline age/sex.
- **Class imbalance**: Bone metastasis is a minority class. Optimize AUPRC alongside AUC. Use class-weighted models and attend to calibration.
- **Key risk factors**: Prior fragility fracture, chronic glucocorticoid use (>3 months), COPD (IL-6 → osteoclast activation), aromatase inhibitor therapy, elevated alkaline phosphatase, declining bone density T-score.
- **Protective signals**: Active denosumab therapy, calcium/vitamin D supplementation, stable or declining OPS trajectory.
- **OPS interpretation rule**: [CORRECTED] Stable HIGH OPS (≥0.6) maintained over time indicates CRITICALLY hospitable bone environment. Risk scores for stable high OPS cases should be ≥0.8, not moderate (0.5-0.7). Only declining high OPS or stable low OPS (<0.4) represents favorable bone dynamics. <!-- [Evidence: trace P06640, iter 1] -->
- **Patient identity verification**: Agents must explicitly state and verify patient ID at beginning of each assessment. Include patient_id validation check in reasoning chain to prevent cross-case contamination. <!-- [Evidence: trace P06640, iter 1] -->

---

### R: Topology Reasoning (initial — to be refined by loop)

- **Default**: Start with T1 (single agent) for simple cases, T2 (pipeline: extract → reason → predict) for moderate complexity.
- **T1 (single agent)**: Use for cases with OPS ≤ 0.35, CCI ≤ 1, AND successful ML prediction on first attempt. For cases with ML prediction failures or CCI ≥ 2, route to T2 (EXTRACTOR → PREDICTOR) to separate feature extraction from risk assessment and avoid token waste on repeated tool calls. <!-- [Evidence: trace P03550, iter 1] -->
- **T4 (debate)**: Consider if prediction confidence is low (< 0.6) or if there are conflicting pharmacological signals (e.g., bisphosphonate + glucocorticoid co-prescription).
- **T5 (hierarchical)**: Consider if comorbidity count > 5 or CCI ≥ 6. Route to specialists for comorbidity cascade analysis.
- **T3 (ensemble)**: DO NOT use initially — high token cost, unproven benefit here. Consider only after evidence of model disagreement being a useful signal.
- **T6 (dynamic)**: Consider after routing conditions in R are well-established (iter ≥ 8). Router agent selects topology per-case based on patient complexity.
- [TO BE REFINED BY LOOP]

---

### T: Agent Template Library (initial)

#### EXTRACTOR
- **Role**: Parse medication timeline, compute OPS trajectory, extract structured features
- **Tools**: get_patient_features, compute_ops_trajectory
- **System prompt core**: "You are a clinical data extraction specialist. Your job is to analyze the patient's medication timeline, compute the OPS trajectory, and extract key risk factors. Use the get_patient_features tool to retrieve the patient record, then use compute_ops_trajectory to compute the bone health trajectory. Summarize: (1) active medications and their bone effects, (2) comorbidity burden and CCI, (3) OPS trajectory trend (rising/stable/declining), (4) key temporal patterns. Output a JSON with fields: medication_summary, cci_score, ops_summary, risk_factors, temporal_patterns."

#### PREDICTOR
- **Role**: Invoke ML ensemble, interpret model scores, produce risk assessment
- **Tools**: predict_risk, get_patient_features
- **System prompt core**: "You are a risk prediction specialist. Use the predict_risk tool to get ML ensemble scores for this patient. Interpret the scores in clinical context. If models disagree (spread > 0.15), note this and explain possible reasons. Consider the OPS trajectory and medication context when interpreting. Output a JSON with fields: risk_score (float 0-1), label (0 or 1), confidence (high/moderate/low), model_agreement, reasoning."

#### REASONER
- **Role**: Trace causal pathway from OPS trajectory and clinical features to metastasis risk
- **Tools**: lookup_drug_interaction, compute_ops_trajectory
- **System prompt core**: "You are a clinical reasoning specialist for bone metastasis. Your job is to trace the causal pathway from the patient's clinical profile to their bone metastasis risk. Consider: (1) the seed-and-soil hypothesis — is the bone microenvironment becoming hospitable? (2) the bisphosphonate paradox — prescription signals disease history, not protection. (3) drug interactions affecting bone. (4) comorbidity cascades. Use lookup_drug_interaction to check relevant interactions. Output a JSON with fields: risk_score, label, reasoning, causal_pathway, key_risk_factors, protective_factors."

#### DOMAIN_EXPERT (optional)
- **Role**: Specialized in bone microenvironment biology
- **Tools**: lookup_drug_interaction, get_patient_features
- **System prompt core**: "You are an oncology domain expert specializing in bone microenvironment biology and metastatic colonization. Analyze this patient's risk factors specifically through the lens of the seed-and-soil hypothesis. Consider: osteoclast/osteoblast balance, TGF-β signaling, IL-6 inflammatory pathway, RANKL/OPG axis. Provide domain-specific risk assessment."

#### CLINICAL_FALLBACK
- **Role**: Handle cases where ML models fail
- **Tools**: get_patient_features, compute_ops_trajectory
- **System prompt core**: "You provide risk assessment when ML ensemble is unavailable. Focus on OPS trajectory interpretation, demographic risk factors, and comorbidity cascade analysis. Output clinical-only risk score with confidence bounds." <!-- [Evidence: trace P01453, iter 0] -->

#### [ADDITIONAL ROLES TO BE DISCOVERED BY LOOP]

---

### P: Construction Protocol (initial)

**Step 1**: Assess case complexity
- Retrieve patient features (medication_count, CCI, record_density)
- Classify: simple (meds ≤ 2, CCI < 4), moderate (meds 3-5, CCI 4-6), complex (meds > 5 or CCI ≥ 6)

**Step 2**: Select topology from R based on complexity assessment
- Simple → T1 (single PREDICTOR agent) or T2 (EXTRACTOR → PREDICTOR)
- Moderate → T2 (EXTRACTOR → REASONER → PREDICTOR)
- Complex → T5 (ORCHESTRATOR → [PREDICTOR, DOMAIN_EXPERT] → SYNTHESIZER)
- Conflicting signals → T4 (PROPOSER → CHALLENGER → JUDGE)

**Step 2.1**: Feature compatibility check - if predict_risk returns dimension mismatch error on first attempt, set feature_compatibility_flag=false and route to clinical-only assessment pipeline. Skip all subsequent ML prediction calls to avoid token waste on repeated failures. <!-- [Evidence: trace P01230, iter 1] -->

**Step 2.5**: Validate feature compatibility - if predict_risk tool returns dimension mismatch error, route to T1 (single PREDICTOR only) to avoid redundant ML calls and reduce token waste. Include feature_compatibility_check flag in inter-agent messages.
<!-- [Evidence: trace P01453, iter 0] -->

**Step 3**: Instantiate agents from T for selected topology
- Use the template library to generate complete system prompts
- Assign appropriate tool subsets to each agent

**Step 4**: Wire inter-agent messages
- Always pass OPS_trajectory and raw_timestamps between agents
- Always pass prediction_uncertainty in inter-agent contexts
- Include medication_summary from extractor in all downstream contexts

**Step 5**: Include prediction uncertainty in all inter-agent contexts
- If ensemble model spread > 0.15, flag as "uncertain" and include individual model scores
- Uncertain cases should trigger more thorough reasoning or debate topology

**Step 6**: Low-risk optimization pattern - for cases with stable low OPS (<0.4), CCI ≤ 1, and no bone-depleting medications, use streamlined assessment focusing on protective factors. Limit reasoning to key protective elements: stable low OPS trajectory, absence of bone-depleting factors, minimal comorbidity burden. <!-- [Evidence: trace P00363, iter 1] -->