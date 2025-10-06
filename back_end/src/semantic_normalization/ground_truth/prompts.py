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
You are a medical intervention relationship classifier. Analyze these two interventions and determine their hierarchical relationship.

INTERVENTION 1: "{intervention_1}"
INTERVENTION 2: "{intervention_2}"
EMBEDDING SIMILARITY: {similarity:.3f}

RELATIONSHIP TYPES:
1. EXACT_MATCH - Identical interventions (synonyms, same thing)
2. VARIANT - Same therapeutic concept, different formulation/biosimilar
3. SUBTYPE - Related but clinically distinct subtypes
4. SAME_CATEGORY - Different entities in same therapeutic class
5. DOSAGE_VARIANT - Same intervention, different dose specification
6. DIFFERENT - Completely unrelated interventions

EXAMPLES (from validated ground truth):

Example 1 - EXACT_MATCH (Similarity: 0.96):
Intervention 1: "Pegylated interferon alpha (Peg-IFNα)"
Intervention 2: "pegylated interferon α (PEG-IFNα)"
Output: {{
  "relationship_type": "EXACT_MATCH",
  "layer_1_canonical": "pegylated interferon alpha",
  "layer_2_same_variant": true,
  "reasoning": "Same drug, different spelling/capitalization of Greek letter alpha"
}}

Example 2 - SAME_CATEGORY (Similarity: 0.72):
Intervention 1: "Lactobacillus reuteri DSM 17938"
Intervention 2: "Saccharomyces boulardii"
Output: {{
  "relationship_type": "SAME_CATEGORY",
  "layer_1_canonical": "probiotics",
  "layer_2_same_variant": false,
  "reasoning": "Both probiotics but different bacterial genera and species"
}}

Example 3 - VARIANT (Similarity: 0.88):
Intervention 1: "Cetuximab"
Intervention 2: "Cetuximab-β"
Output: {{
  "relationship_type": "VARIANT",
  "layer_1_canonical": "cetuximab",
  "layer_2_same_variant": false,
  "reasoning": "Cetuximab-β is a biosimilar variant of cetuximab"
}}

Example 4 - SUBTYPE (Similarity: 0.78):
Intervention 1: "IBS-D"
Intervention 2: "IBS-C"
Output: {{
  "relationship_type": "SUBTYPE",
  "layer_1_canonical": "irritable bowel syndrome",
  "layer_2_same_variant": false,
  "reasoning": "Both IBS subtypes but clinically distinct (diarrhea vs constipation predominant)"
}}

Example 5 - DOSAGE_VARIANT (Similarity: 0.92):
Intervention 1: "metformin"
Intervention 2: "metformin 500mg"
Output: {{
  "relationship_type": "DOSAGE_VARIANT",
  "layer_1_canonical": "metformin",
  "layer_2_same_variant": true,
  "reasoning": "Same drug, different dosage specification"
}}

Example 6 - SAME_CATEGORY (Similarity: 0.74):
Intervention 1: "atorvastatin"
Intervention 2: "simvastatin"
Output: {{
  "relationship_type": "SAME_CATEGORY",
  "layer_1_canonical": "statins",
  "layer_2_same_variant": false,
  "reasoning": "Both statin drugs but different molecules"
}}

Example 7 - DIFFERENT (Similarity: 0.15):
Intervention 1: "vitamin D"
Intervention 2: "chemotherapy"
Output: {{
  "relationship_type": "DIFFERENT",
  "layer_1_canonical": null,
  "layer_2_same_variant": false,
  "reasoning": "Completely unrelated interventions"
}}

INSTRUCTIONS:
1. Classify the relationship type (1-6) based on similarity score and semantic meaning
2. Identify the Layer 1 canonical group if they share one (use null for DIFFERENT)
3. Determine if they are the same Layer 2 variant (true/false)
4. Provide brief reasoning for your classification

GUIDELINES FOR SIMILARITY RANGES:
- Above 0.95: Usually EXACT_MATCH or DOSAGE_VARIANT
- 0.85-0.95: Usually VARIANT or EXACT_MATCH
- 0.75-0.85: Usually SUBTYPE or VARIANT
- 0.70-0.80: Usually SAME_CATEGORY or SUBTYPE
- Below 0.70: Usually DIFFERENT

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
        "VARIANT",
        "SUBTYPE",
        "SAME_CATEGORY",
        "DOSAGE_VARIANT",
        "DIFFERENT"
    ]
}
