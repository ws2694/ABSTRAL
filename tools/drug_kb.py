"""Static pharmacological knowledge base for bone-relevant drug interactions.

Deterministic lookup — no external calls, no randomness.
"""

# Drug class → bone effect profile
DRUG_PROFILES = {
    "bisphosphonate": {
        "mechanism": "Inhibits osteoclast-mediated bone resorption by binding to hydroxyapatite",
        "bone_effect": "protective",
        "clinical_note": (
            "PARADOX: Bisphosphonate prescription signals existing bone disease "
            "history (osteoporosis, bone metastasis risk). Presence is a RISK "
            "MARKER, not a protective factor for metastasis prediction. "
            "Agent must reason: prescription intent ≠ bone health outcome."
        ),
        "common_drugs": ["alendronate", "zoledronic_acid", "risedronate", "pamidronate"]
    },
    "glucocorticoid": {
        "mechanism": "Suppresses osteoblast function, increases osteoclast lifespan, reduces calcium absorption",
        "bone_effect": "harmful",
        "clinical_note": (
            "Dose-dependent bone loss. Chronic use (>3 months) significantly "
            "increases fracture risk. Glucocorticoid-induced osteoporosis is "
            "the most common form of secondary osteoporosis."
        ),
        "common_drugs": ["prednisone", "dexamethasone", "methylprednisolone", "hydrocortisone"]
    },
    "aromatase_inhibitor": {
        "mechanism": "Reduces estrogen synthesis, accelerating bone turnover and resorption",
        "bone_effect": "harmful",
        "clinical_note": "Used in breast cancer; significant bone density loss over 2-5 years.",
        "common_drugs": ["letrozole", "anastrozole", "exemestane"]
    },
    "denosumab": {
        "mechanism": "RANKL inhibitor — blocks osteoclast formation and activation",
        "bone_effect": "protective",
        "clinical_note": "Both osteoporosis treatment and bone metastasis prevention.",
        "common_drugs": ["denosumab"]
    },
    "chemotherapy_platinum": {
        "mechanism": "Cytotoxic — indirect bone effects via gonadal toxicity and general catabolic state",
        "bone_effect": "harmful",
        "clinical_note": "Cisplatin/carboplatin cause hypomagnesemia affecting bone mineralization.",
        "common_drugs": ["cisplatin", "carboplatin"]
    },
    "immunotherapy_checkpoint": {
        "mechanism": "Immune activation may affect bone remodeling via inflammatory cytokines",
        "bone_effect": "neutral",
        "clinical_note": "Emerging evidence of bone effects; insufficient data for strong claims.",
        "common_drugs": ["pembrolizumab", "nivolumab", "atezolizumab"]
    },
    "nsaid": {
        "mechanism": "COX inhibition may have mild protective effect on bone via prostaglandin modulation",
        "bone_effect": "neutral",
        "clinical_note": "Minimal direct bone effect at standard doses.",
        "common_drugs": ["ibuprofen", "naproxen", "celecoxib"]
    },
    "proton_pump_inhibitor": {
        "mechanism": "Reduces calcium absorption by increasing gastric pH",
        "bone_effect": "harmful",
        "clinical_note": "Long-term PPI use associated with increased fracture risk.",
        "common_drugs": ["omeprazole", "pantoprazole", "esomeprazole"]
    },
    "thyroid_hormone": {
        "mechanism": "Excess thyroid hormone accelerates bone turnover",
        "bone_effect": "harmful",
        "clinical_note": "Hyperthyroidism or over-replacement increases fracture risk.",
        "common_drugs": ["levothyroxine"]
    },
    "calcium_vitamin_d": {
        "mechanism": "Essential substrates for bone mineralization",
        "bone_effect": "protective",
        "clinical_note": "Supplementation standard of care for osteoporosis prevention.",
        "common_drugs": ["calcium_carbonate", "cholecalciferol"]
    }
}

# Pairwise interaction effects
INTERACTIONS = {
    ("bisphosphonate", "glucocorticoid"): {
        "interaction": "partially_antagonistic",
        "net_bone_effect": "elevated_resorption",
        "mechanism": (
            "Glucocorticoid suppresses osteoblast formation while bisphosphonate "
            "inhibits osteoclast resorption. Net effect: bone formation impaired "
            "more than resorption is suppressed. Glucocorticoid effect dominates "
            "in early treatment; bisphosphonate partially compensates over time."
        ),
        "clinical_significance": "high",
        "recommendation": (
            "Co-prescription suggests patient has BOTH bone disease AND "
            "inflammatory/autoimmune condition requiring steroids. High-risk profile."
        )
    },
    ("bisphosphonate", "denosumab"): {
        "interaction": "synergistic",
        "net_bone_effect": "strongly_protective",
        "mechanism": "Dual anti-resorptive: blocks both osteoclast formation (denosumab) and function (bisphosphonate).",
        "clinical_significance": "medium",
        "recommendation": "Sequential use common; concurrent use rare but signals aggressive bone protection."
    },
    ("glucocorticoid", "chemotherapy_platinum"): {
        "interaction": "additive_harmful",
        "net_bone_effect": "severely_harmful",
        "mechanism": (
            "Combined catabolic effect: glucocorticoid directly suppresses "
            "osteoblasts; chemotherapy causes gonadal toxicity reducing estrogen/testosterone."
        ),
        "clinical_significance": "high",
        "recommendation": "Monitor bone density; consider prophylactic bisphosphonate."
    },
    ("aromatase_inhibitor", "bisphosphonate"): {
        "interaction": "partially_compensatory",
        "net_bone_effect": "moderate_loss",
        "mechanism": "Bisphosphonate partially offsets aromatase inhibitor bone loss, but not fully.",
        "clinical_significance": "medium",
        "recommendation": "Standard co-prescription in breast cancer patients."
    },
    ("glucocorticoid", "proton_pump_inhibitor"): {
        "interaction": "additive_harmful",
        "net_bone_effect": "harmful",
        "mechanism": "Both reduce calcium availability — glucocorticoid reduces absorption, PPI reduces gastric acid needed for calcium dissolution.",
        "clinical_significance": "medium",
        "recommendation": "Common co-prescription; compounding fracture risk."
    }
}


class DrugKB:
    """Pharmacological knowledge base for bone-relevant drug interactions."""

    def lookup(self, drug_classes: list[str]) -> dict:
        """Look up interactions between the given drug classes."""
        drug_classes = [d.lower().strip() for d in drug_classes]
        result = {
            "drugs_queried": drug_classes,
            "profiles": {},
            "interactions": [],
            "unknown_drugs": []
        }

        for dc in drug_classes:
            if dc in DRUG_PROFILES:
                result["profiles"][dc] = DRUG_PROFILES[dc]
            else:
                result["unknown_drugs"].append(dc)

        for i, d1 in enumerate(drug_classes):
            for d2 in drug_classes[i + 1:]:
                key = (d1, d2)
                rev_key = (d2, d1)
                if key in INTERACTIONS:
                    result["interactions"].append({
                        "drugs": [d1, d2],
                        **INTERACTIONS[key]
                    })
                elif rev_key in INTERACTIONS:
                    result["interactions"].append({
                        "drugs": [d1, d2],
                        **INTERACTIONS[rev_key]
                    })

        if not result["interactions"] and len(drug_classes) > 1:
            result["interactions"].append({
                "drugs": drug_classes,
                "interaction": "no_known_interaction",
                "net_bone_effect": "unknown",
                "mechanism": "No documented interaction in knowledge base.",
                "clinical_significance": "low"
            })

        return result


DRUG_KB = DrugKB()
