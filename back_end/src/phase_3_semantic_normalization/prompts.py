"""
LLM Prompt Templates for Hierarchical Normalization
Canonical group extraction only - relationship classification moved to Phase 3d (cluster-level).
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
# HELPER FUNCTIONS
# ==============================================================================

def format_canonical_extraction_prompt(intervention_name: str) -> str:
    """Format the canonical extraction prompt with intervention name."""
    return CANONICAL_EXTRACTION_PROMPT.format(intervention_name=intervention_name)


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
