"""
LLM Prompt Templates for Hierarchical Normalization
Includes ground truth examples (Scenarios 1-3) for few-shot learning
"""

# ==============================================================================
# CANONICAL GROUP EXTRACTION PROMPT
# ==============================================================================

CANONICAL_EXTRACTION_PROMPT = """
You are a medical intervention classifier. Extract the canonical therapeutic group for this intervention.

INTERVENTION: "{intervention_name}"

RULES:
1. Use the base drug/treatment name (remove dosage, route, brand names)
2. Use lowercase
3. If part of a known therapeutic class, use the class name
4. If standalone drug, use normalized drug name
5. NEVER return null or empty string

EXAMPLES (from validated ground truth):

Example 1 - Specific Drug (Unknown Class):
Input: "Pegylated interferon alpha (Peg-IFNα)"
Output: {{"canonical_group": "pegylated interferon alpha", "reasoning": "Specific drug, normalize to base form"}}

Example 2 - Known Therapeutic Class (Probiotic):
Input: "Lactobacillus reuteri DSM 17938"
Output: {{"canonical_group": "probiotics", "reasoning": "Lactobacillus is a probiotic bacterial genus"}}

Example 3 - Known Therapeutic Class (Probiotic):
Input: "Saccharomyces boulardii"
Output: {{"canonical_group": "probiotics", "reasoning": "Probiotic yeast species"}}

Example 4 - Specific Drug (Monoclonal Antibody):
Input: "Cetuximab"
Output: {{"canonical_group": "cetuximab", "reasoning": "Specific monoclonal antibody drug"}}

Example 5 - Biosimilar Variant:
Input: "Cetuximab-β"
Output: {{"canonical_group": "cetuximab", "reasoning": "Biosimilar variant of cetuximab"}}

Example 6 - Known Drug Class:
Input: "atorvastatin 20mg"
Output: {{"canonical_group": "statins", "reasoning": "Atorvastatin is a statin drug"}}

Example 7 - Normalize Dosage:
Input: "metformin 500mg"
Output: {{"canonical_group": "metformin", "reasoning": "Remove dosage, use base drug name"}}

Return ONLY valid JSON with this exact structure:
{{"canonical_group": "lowercase name", "reasoning": "brief explanation"}}
"""

# ==============================================================================
# RELATIONSHIP CLASSIFICATION PROMPT
# ==============================================================================

RELATIONSHIP_CLASSIFICATION_PROMPT = """
You are a medical intervention relationship classifier. Analyze these two interventions and determine their hierarchical relationship based on which layer differs in the 4-layer hierarchy.

INTERVENTION 1: "{intervention_1}"
INTERVENTION 2: "{intervention_2}"
EMBEDDING SIMILARITY: {similarity:.3f}

4-LAYER HIERARCHY:
- Layer 0: Taxonomy category (e.g., supplement, medication, therapy)
- Layer 1: Canonical group (e.g., probiotics, statins, vitamin d)
- Layer 2: Specific variant (e.g., L. reuteri, atorvastatin, cholecalciferol)
- Layer 3: Dosage/strain detail (e.g., DSM 17938, 50mg, 1000 IU)

RELATIONSHIP TYPES (based on which layer differs):

1. EXACT_MATCH - All 4 layers identical (synonyms, spelling variants)
   Layers: 0✓ 1✓ 2✓ 3✓

2. DOSAGE_VARIANT - Layer 3 differs (same variant, different dosage/strain)
   Layers: 0✓ 1✓ 2✓ 3✗

3. SAME_CATEGORY_TYPE_VARIANT - Layer 2 differs (same canonical group, different variants)
   Layers: 0✓ 1✓ 2✗ 3✗

4. SAME_CATEGORY - Layer 1 differs (different canonical groups, same taxonomy category)
   Layers: 0✓ 1✗ 2✗ 3✗

5. DIFFERENT - Layer 0 differs or completely unrelated
   Layers: 0✗ 1✗ 2✗ 3✗

EXAMPLES:

Example 1 - EXACT_MATCH (Similarity: 0.96):
Intervention 1: "Pegylated interferon alpha (Peg-IFNα)"
Intervention 2: "pegylated interferon α (PEG-IFNα)"
Analysis:
  Layer 0: biologics = biologics ✓
  Layer 1: pegylated interferon alpha = pegylated interferon alpha ✓
  Layer 2: pegylated interferon alpha = pegylated interferon alpha ✓
  Layer 3: [none] = [none] ✓
Output: {{
  "relationship_type": "EXACT_MATCH",
  "layer_1_canonical": "pegylated interferon alpha",
  "layer_2_same_variant": true,
  "reasoning": "All layers identical - same drug with spelling/capitalization differences"
}}

Example 2 - DOSAGE_VARIANT (Similarity: 0.92):
Intervention 1: "metformin"
Intervention 2: "metformin 500mg"
Analysis:
  Layer 0: medication = medication ✓
  Layer 1: metformin = metformin ✓
  Layer 2: metformin = metformin ✓
  Layer 3: [none] ≠ 500mg ✗
Output: {{
  "relationship_type": "DOSAGE_VARIANT",
  "layer_1_canonical": "metformin",
  "layer_2_same_variant": true,
  "reasoning": "Layer 3 differs - same drug, different dosage specification"
}}

Example 3 - DOSAGE_VARIANT (Similarity: 0.88):
Intervention 1: "L. reuteri DSM 17938"
Intervention 2: "L. reuteri ATCC 55730"
Analysis:
  Layer 0: supplement = supplement ✓
  Layer 1: probiotics = probiotics ✓
  Layer 2: L. reuteri = L. reuteri ✓
  Layer 3: DSM 17938 ≠ ATCC 55730 ✗
Output: {{
  "relationship_type": "DOSAGE_VARIANT",
  "layer_1_canonical": "probiotics",
  "layer_2_same_variant": true,
  "reasoning": "Layer 3 differs - same probiotic species, different strain identifiers"
}}

Example 4 - SAME_CATEGORY_TYPE_VARIANT (Similarity: 0.78):
Intervention 1: "vitamin D"
Intervention 2: "Vitamin D3"
Analysis:
  Layer 0: supplement = supplement ✓
  Layer 1: vitamin d = vitamin d ✓
  Layer 2: vitamin d (generic) ≠ cholecalciferol (D3 form) ✗
  Layer 3: [none] = [none]
Output: {{
  "relationship_type": "SAME_CATEGORY_TYPE_VARIANT",
  "layer_1_canonical": "vitamin d",
  "layer_2_same_variant": false,
  "reasoning": "Layer 2 differs - vitamin D3 is a specific chemical form of vitamin D family"
}}

Example 5 - SAME_CATEGORY_TYPE_VARIANT (Similarity: 0.72):
Intervention 1: "Lactobacillus reuteri"
Intervention 2: "Saccharomyces boulardii"
Analysis:
  Layer 0: supplement = supplement ✓
  Layer 1: probiotics = probiotics ✓
  Layer 2: L. reuteri ≠ S. boulardii ✗
  Layer 3: [different or none]
Output: {{
  "relationship_type": "SAME_CATEGORY_TYPE_VARIANT",
  "layer_1_canonical": "probiotics",
  "layer_2_same_variant": false,
  "reasoning": "Layer 2 differs - both probiotics but different species"
}}

Example 6 - SAME_CATEGORY_TYPE_VARIANT (Similarity: 0.85):
Intervention 1: "Cetuximab"
Intervention 2: "Cetuximab-β"
Analysis:
  Layer 0: biologics = biologics ✓
  Layer 1: cetuximab = cetuximab ✓
  Layer 2: cetuximab ≠ cetuximab-β (biosimilar) ✗
  Layer 3: [none] = [none]
Output: {{
  "relationship_type": "SAME_CATEGORY_TYPE_VARIANT",
  "layer_1_canonical": "cetuximab",
  "layer_2_same_variant": false,
  "reasoning": "Layer 2 differs - cetuximab-β is a biosimilar variant"
}}

Example 7 - SAME_CATEGORY_TYPE_VARIANT (Similarity: 0.74):
Intervention 1: "atorvastatin"
Intervention 2: "simvastatin"
Analysis:
  Layer 0: medication = medication ✓
  Layer 1: statins = statins ✓
  Layer 2: atorvastatin ≠ simvastatin ✗
  Layer 3: [different or none]
Output: {{
  "relationship_type": "SAME_CATEGORY_TYPE_VARIANT",
  "layer_1_canonical": "statins",
  "layer_2_same_variant": false,
  "reasoning": "Layer 2 differs - both statin drugs but different molecules"
}}

Example 8 - SAME_CATEGORY (Similarity: 0.68):
Intervention 1: "probiotics"
Intervention 2: "magnesium"
Analysis:
  Layer 0: supplement = supplement ✓
  Layer 1: probiotics ≠ magnesium ✗
  Layer 2: [different]
  Layer 3: [different]
Output: {{
  "relationship_type": "SAME_CATEGORY",
  "layer_1_canonical": null,
  "layer_2_same_variant": false,
  "reasoning": "Layer 1 differs - different supplement types, same taxonomy category"
}}

Example 9 - DIFFERENT (Similarity: 0.15):
Intervention 1: "vitamin D"
Intervention 2: "chemotherapy"
Analysis:
  Layer 0: supplement ≠ medication ✗
  Layer 1: [different]
  Layer 2: [different]
  Layer 3: [different]
Output: {{
  "relationship_type": "DIFFERENT",
  "layer_1_canonical": null,
  "layer_2_same_variant": false,
  "reasoning": "Layer 0 differs - completely unrelated (supplement vs medication)"
}}

INSTRUCTIONS:
1. Identify which layers are shared and which differ
2. Use the layer difference pattern to determine relationship type
3. For SAME_CATEGORY and DIFFERENT, set layer_1_canonical to null
4. Set layer_2_same_variant to true only for EXACT_MATCH and DOSAGE_VARIANT
5. Provide brief reasoning explaining which layer differs

GUIDELINES FOR SIMILARITY RANGES:
- 0.95-1.00: Usually EXACT_MATCH or DOSAGE_VARIANT
- 0.90-0.98: Usually DOSAGE_VARIANT
- 0.70-0.90: Usually SAME_CATEGORY_TYPE_VARIANT
- 0.60-0.75: Usually SAME_CATEGORY
- Below 0.60: Usually DIFFERENT

Return ONLY valid JSON with this exact structure:
{{
  "relationship_type": "TYPE_NAME",
  "layer_1_canonical": "canonical name or null",
  "layer_2_same_variant": true/false,
  "reasoning": "brief explanation"
}}
"""

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def format_canonical_extraction_prompt(intervention_name: str) -> str:
    """Format the canonical extraction prompt with intervention name."""
    return CANONICAL_EXTRACTION_PROMPT.format(intervention_name=intervention_name)


def format_relationship_classification_prompt(
    intervention_1: str,
    intervention_2: str,
    similarity: float
) -> str:
    """Format the relationship classification prompt with intervention pair and similarity."""
    return RELATIONSHIP_CLASSIFICATION_PROMPT.format(
        intervention_1=intervention_1,
        intervention_2=intervention_2,
        similarity=similarity
    )


# ==============================================================================
# VALIDATION SCHEMAS
# ==============================================================================

CANONICAL_EXTRACTION_SCHEMA = {
    "required_fields": ["canonical_group", "reasoning"],
    "field_types": {
        "canonical_group": str,
        "reasoning": str
    },
    "field_constraints": {
        "canonical_group": lambda x: len(x) > 0 and x.islower()
    }
}

RELATIONSHIP_CLASSIFICATION_SCHEMA = {
    "required_fields": ["relationship_type", "layer_1_canonical", "layer_2_same_variant", "reasoning"],
    "field_types": {
        "relationship_type": str,
        "layer_1_canonical": (str, type(None)),  # Can be null
        "layer_2_same_variant": bool,
        "reasoning": str
    },
    "valid_relationship_types": [
        "EXACT_MATCH",
        "DOSAGE_VARIANT",
        "SAME_CATEGORY_TYPE_VARIANT",
        "SAME_CATEGORY",
        "DIFFERENT"
    ]
}
