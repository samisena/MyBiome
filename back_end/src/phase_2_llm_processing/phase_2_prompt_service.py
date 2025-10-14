"""
Shared prompt service for intervention extraction.
Eliminates code duplication across different analyzer classes.
"""

from typing import Dict, List
import sys
from pathlib import Path

from back_end.src.data.config import setup_logging
from back_end.src.data.utils import read_fulltext_content
from back_end.src.interventions.taxonomy import InterventionType, intervention_taxonomy

logger = setup_logging(__name__, 'prompt_service.log')


class InterventionPromptService:
    """
    Centralized service for creating intervention extraction prompts.
    Provides consistent prompts across all analyzer classes.
    """
    
    def __init__(self):
        """Initialize the prompt service with intervention taxonomy."""
        self.taxonomy = intervention_taxonomy
        self.logger = logger
    
    def create_extraction_prompt(self, paper: Dict) -> str:
        """
        Create a standardized prompt for multi-intervention extraction.
        
        Args:
            paper: Paper dictionary with title, abstract, and optionally full-text
            
        Returns:
            Formatted prompt string for LLM extraction
        """
        # Prepare paper content
        content_sections = self._prepare_paper_content(paper)
        paper_content = "\n\n".join(content_sections)

        # Build the standardized prompt
        prompt = self._build_intervention_prompt(paper_content)

        return prompt

    def create_entity_matching_prompt(self, term: str, candidate_canonicals: List[str], entity_type: str) -> str:
        """
        Create prompt for matching a single entity term to canonical forms.

        Args:
            term: The term to match
            candidate_canonicals: List of canonical terms to match against
            entity_type: Type of entity (e.g., 'intervention', 'condition')

        Returns:
            Formatted prompt string for entity matching
        """
        candidates_list = "\n".join([f"- {canonical}" for canonical in candidate_canonicals])

        prompt = f"""You are a medical terminology expert. Given the {entity_type} term '{term}', determine if it represents the same medical concept as any of these canonical terms:

{candidates_list}

CRITICAL MEDICAL SAFETY - Be extremely cautious with:
⚠️ OPPOSITE CONDITIONS: Terms with prefixes creating opposite meanings:
   • hyper- vs hypo- (hypertension ≠ hypotension, hyperglycemia ≠ hypoglycemia)
   • tachy- vs brady- (tachycardia ≠ bradycardia)
   • -osis vs -alosis suffixes (acidosis ≠ alkalosis)
⚠️ DIFFERENT INTERVENTIONS: Similar but distinct medical interventions:
   • probiotics (live bacteria) ≠ prebiotics (bacterial food)
   • antibiotics vs antivirals vs antifungals (different drug classes)
   • different surgical procedures, therapy types, supplement categories
⚠️ MEDICAL PRECISION:
   • Dosage forms matter: oral vs topical vs injection routes are different
   • Severity levels: acute vs chronic conditions are different
   • Timing: pre-operative vs post-operative interventions are different
⚠️ SOUND-ALIKE TERMS: Many medical terms sound similar but have different meanings
⚠️ CONTEXT DEPENDENCY: Same intervention can have different medical contexts

MATCHING PRINCIPLES:
- Only match if terms represent the EXACT SAME medical concept
- Consider valid synonyms, abbreviations, and alternative names
- When in doubt, prefer NO MATCH rather than incorrect match
- Medical accuracy is more important than recall

Respond with valid JSON only:
{{
    "match": "exact_canonical_name_from_list_above" or null,
    "confidence": 0.0-1.0,
    "reasoning": "brief medical explanation"
}}"""

        return prompt

    def create_batch_entity_matching_prompt(self, terms: List[str], candidate_canonicals: List[str], entity_type: str) -> str:
        """
        Create prompt for matching multiple entity terms to canonical forms in a single request.

        Args:
            terms: List of terms to match
            candidate_canonicals: List of canonical terms to match against
            entity_type: Type of entity (e.g., 'intervention', 'condition')

        Returns:
            Formatted prompt string for batch entity matching
        """
        candidates_list = "\n".join([f"- {canonical}" for canonical in candidate_canonicals])
        terms_list = "\n".join([f"{i+1}. {term}" for i, term in enumerate(terms)])

        prompt = f"""You are a medical terminology expert. For each {entity_type} term below, determine if it represents the same medical concept as any of the canonical terms.

TERMS TO MATCH:
{terms_list}

CANONICAL TERMS:
{candidates_list}

CRITICAL MEDICAL SAFETY - Be extremely cautious with:
⚠️ OPPOSITE CONDITIONS: Terms with prefixes creating opposite meanings:
   • hyper- vs hypo- (hypertension ≠ hypotension, hyperglycemia ≠ hypoglycemia)
   • tachy- vs brady- (tachycardia ≠ bradycardia)
   • -osis vs -alosis suffixes (acidosis ≠ alkalosis)
⚠️ DIFFERENT INTERVENTIONS: Similar but distinct medical interventions:
   • probiotics (live bacteria) ≠ prebiotics (bacterial food)
   • antibiotics vs antivirals vs antifungals (different drug classes)
   • different surgical procedures, therapy types, supplement categories
⚠️ MEDICAL PRECISION:
   • Dosage forms matter: oral vs topical vs injection routes are different
   • Severity levels: acute vs chronic conditions are different
   • Timing: pre-operative vs post-operative interventions are different
⚠️ SOUND-ALIKE TERMS: Many medical terms sound similar but have different meanings
⚠️ CONTEXT DEPENDENCY: Same intervention can have different medical contexts

MATCHING PRINCIPLES:
- Only match if terms represent the EXACT SAME medical concept
- Consider valid synonyms, abbreviations, and alternative names
- When in doubt, prefer NO MATCH rather than incorrect match
- Medical accuracy is more important than recall

Respond with valid JSON array only:
[
    {{
        "term_number": 1,
        "original_term": "{terms[0] if terms else 'example'}",
        "match": "exact_canonical_name_from_list_above" or null,
        "confidence": 0.0-1.0,
        "reasoning": "brief medical explanation"
    }},
    {{
        "term_number": 2,
        "original_term": "{terms[1] if len(terms) > 1 else 'example2'}",
        "match": "exact_canonical_name_from_list_above" or null,
        "confidence": 0.0-1.0,
        "reasoning": "brief medical explanation"
    }}
]"""

        return prompt

    def create_duplicate_analysis_prompt(self, terms: List[str]) -> str:
        """
        Create prompt for analyzing terms to identify duplicates/synonyms.

        Args:
            terms: List of terms to analyze for duplicates

        Returns:
            Formatted prompt string for duplicate analysis
        """
        terms_list = "\n".join([f"- {term}" for term in terms])

        prompt = f"""You are a medical terminology expert analyzing intervention names extracted from biomedical research papers. Your task is to identify which terms refer to the SAME medical intervention (drug, supplement, exercise, diet, therapy, lifestyle modification, etc.).

**TERMS TO ANALYZE:**
{terms_list}

**GROUPING RULES BY INTERVENTION TYPE:**

1. **PHARMACEUTICAL - Brand vs Generic Names:**
   - "aspirin" = "acetylsalicylic acid" = "ASA"
   - "Tylenol" = "acetaminophen" = "paracetamol"
   - "Lipitor" = "atorvastatin"

2. **SUPPLEMENTS - Chemical Variants:**
   - "vitamin D" = "Vitamin D3" = "cholecalciferol" = "25-hydroxyvitamin D"
   - "omega-3" = "omega-3 fatty acids" = "fish oil" = "EPA/DHA"
   - "CoQ10" = "coenzyme Q10" = "ubiquinone"

3. **EXERCISE - Activity Synonyms:**
   - "walking" = "walking exercise" = "walking training"
   - "aerobic exercise" = "aerobic training" = "cardio"
   - "resistance training" = "strength training" = "weight training"

4. **DIET - Dietary Pattern Synonyms:**
   - "Mediterranean diet" = "Mediterranean dietary pattern"
   - "low-fat diet" = "reduced-fat diet" = "fat-restricted diet"
   - "caloric restriction" = "calorie restriction" = "energy restriction"

5. **THERAPY - Treatment Abbreviations:**
   - "cognitive behavioral therapy" = "CBT" = "cognitive behavior therapy"
   - "ACE inhibitors" = "angiotensin-converting enzyme inhibitors"

6. **CAPITALIZATION - Always Group:**
   - "metformin" = "Metformin" = "METFORMIN"
   - "walking" = "Walking" = "WALKING"

7. **PROTOCOL/TIMING DESCRIPTORS - Group together:**
   - "atorvastatin" = "atorvastatin pretreatment" = "atorvastatin therapy" = "atorvastatin treatment"
   - "exercise" = "exercise intervention" = "exercise training" = "exercise program"
   - "vitamin D" = "vitamin D supplementation" = "vitamin D therapy"

8. **DOSAGE/INTENSITY DESCRIPTORS - Group together:**
   - "aspirin" = "low-dose aspirin" = "high-dose aspirin"
   - "exercise" = "moderate-intensity exercise" = "high-intensity exercise"

**DO NOT GROUP:**
- Different drugs in same class: "atorvastatin" vs "simvastatin" (different statins)
- Different exercises: "walking" vs "swimming" vs "cycling" (different activities)
- Different diets: "Mediterranean diet" vs "ketogenic diet" (different dietary patterns)
- Drug combinations: "aspirin" vs "aspirin + clopidogrel"
- Different therapies: "CBT" vs "psychotherapy" (different treatment types)

**OUTPUT FORMAT (VALID JSON ONLY):**
{{
  "duplicate_groups": [
    ["term1", "term2", "term3"],
    ["term4", "term5"]
  ]
}}

**IMPORTANT GUIDELINES:**
- Return ONLY the JSON, no explanations or markdown formatting
- Each inner array is ONE group of equivalent terms
- Only include groups with 2+ terms
- If NO duplicates found, return: {{"duplicate_groups": []}}
- BE CONSERVATIVE: Only group terms if highly confident they refer to the same intervention
- When in doubt, keep terms separate - false negatives are better than false positives"""

        return prompt

    def create_condition_equivalence_prompt(self, condition1: str, condition2: str,
                                           intervention_name: str, paper_pmid: str = None) -> str:
        """
        Create prompt for checking if two condition names are semantically equivalent.

        This is critical for preventing evidence inflation from dual-model extraction.

        Args:
            condition1: First condition name
            condition2: Second condition name
            intervention_name: The intervention being studied (context)
            paper_pmid: Paper ID for context (optional)

        Returns:
            Formatted prompt string for condition equivalence checking
        """
        paper_context = f" (from paper PMID: {paper_pmid})" if paper_pmid else ""

        prompt = f"""You are analyzing two health condition names extracted from the SAME research paper{paper_context} studying the intervention: "{intervention_name}".

Condition 1: "{condition1}"
Condition 2: "{condition2}"

Determine if these two conditions refer to essentially the same health issue in the context of this paper.

IMPORTANT MEDICAL SAFETY RULES:
⚠️ OPPOSITE CONDITIONS are DIFFERENT:
   • hypertension ≠ hypotension
   • hyperglycemia ≠ hypoglycemia
   • tachycardia ≠ bradycardia

✓ HIERARCHICAL CONDITIONS may be equivalent:
   • "diabetes" and "type 2 diabetes" → same if paper only studies type 2
   • "cognitive impairment" and "diabetes-induced cognitive impairment" → same if paper only studies this subtype
   • "depression" and "major depressive disorder" → same if referring to same condition

✓ SAME CONCEPT, DIFFERENT WORDING:
   • "cardiovascular disease" and "heart disease" → equivalent
   • "type 2 diabetes mellitus" and "type 2 diabetes" → equivalent
   • Abbreviations vs full names → equivalent (e.g., "T2DM" vs "type 2 diabetes mellitus")

DECISION CRITERIA:
1. In the context of THIS paper, do both conditions refer to the same patient population/health issue?
2. Is one condition a more specific variant of the other that's still being studied as the same thing?
3. When in doubt, consider: would a researcher count these as separate findings or the same finding?

Respond with ONLY valid JSON:
{{
    "are_equivalent": true/false,
    "confidence": 0.0-1.0,
    "reasoning": "brief explanation of your decision",
    "preferred_wording": "condition1 or condition2 - which is more accurate/specific",
    "is_hierarchical": true/false,
    "relationship": "same" | "subtype" | "different"
}}

CRITICAL: Set confidence > 0.7 only if you're quite sure they're equivalent."""

        return prompt

    def create_consensus_wording_prompt(self, condition_variants: List[str],
                                       intervention_name: str,
                                       extraction_models: List[str] = None) -> str:
        """
        Create prompt for selecting the best wording from multiple condition variants.

        Args:
            condition_variants: List of different condition wordings
            intervention_name: The intervention being studied
            extraction_models: Which models extracted each variant (optional)

        Returns:
            Formatted prompt for consensus wording selection
        """
        # Format variants with model attribution if available
        if extraction_models and len(extraction_models) == len(condition_variants):
            variants_list = "\n".join([
                f"{i+1}. \"{variant}\" (extracted by {model})"
                for i, (variant, model) in enumerate(zip(condition_variants, extraction_models))
            ])
        else:
            variants_list = "\n".join([
                f"{i+1}. \"{variant}\""
                for i, variant in enumerate(condition_variants)
            ])

        prompt = f"""Multiple AI models extracted the same intervention-condition relationship from a research paper, but used slightly different wordings for the condition name.

Intervention studied: "{intervention_name}"

Condition name variants:
{variants_list}

Select the BEST wording to use as the canonical condition name for this intervention.

SELECTION CRITERIA (in priority order):
1. **Medical Accuracy**: Most medically accurate and specific
2. **Clarity**: Clearest to medical professionals
3. **Specificity**: More specific is better if it's accurate (e.g., "type 2 diabetes" > "diabetes")
4. **Completeness**: Includes relevant qualifiers that add meaning
5. **Consistency**: Uses standard medical terminology

AVOID:
- Overly verbose wordings that add no medical value
- Generic terms when specific ones are available
- Unnecessarily complex medical jargon if simpler is equally accurate

Respond with ONLY valid JSON:
{{
    "selected_wording": "exact wording from list above",
    "variant_number": number (1-based index),
    "confidence": 0.0-1.0,
    "reasoning": "brief explanation of why this wording is best",
    "alternatives": ["other good options if any"]
}}"""

        return prompt

    def _prepare_paper_content(self, paper: Dict) -> List[str]:
        """Prepare paper content sections for the prompt."""
        content_sections = []
        
        # Always include title and abstract
        content_sections.append(f"Title: {paper['title']}")
        content_sections.append(f"Abstract: {paper['abstract']}")
        
        # Add full-text if available
        if paper.get('has_fulltext') and paper.get('fulltext_path'):
            fulltext_content = read_fulltext_content(paper['fulltext_path'])
            if fulltext_content:
                content_sections.append(f"Full Text: {fulltext_content}")
                # Using full-text content (logging removed for performance)
            else:
                self.logger.warning(f"Could not read full-text for paper {paper.get('pmid', 'unknown')}")
        
        return content_sections
    
    def _build_intervention_prompt(self, paper_content: str) -> str:
        """Build the complete intervention extraction prompt (hierarchical format)."""
        return f"""You are a biomedical research expert extracting structured intervention data from medical literature.

TASK: Extract interventions from the paper as a JSON array. Return ONLY the JSON array—no markdown, no explanations.

OUTPUT FORMAT (CRITICAL):
Your response MUST start with [ and end with ]. NO explanations, NO markdown code blocks (```), NO commentary.
WRONG: ```json[{{...}}]``` or "Here are the conditions: [{{...}}]"
CORRECT: [{{...}}]

FORMAT:
[{{
  "health_condition": "primary condition targeted (NOT underlying disease)",
  "study_focus": ["research question 1", "research question 2"],
  "measured_metrics": ["specific measurement 1", "specific measurement 2"],
  "findings": ["key result 1", "key result 2"],
  "study_location": "string or null",
  "publisher": "journal name or null",
  "sample_size": number or null,
  "study_duration": "string or null",
  "study_type": "string or null",
  "population_details": "string or null",
  "interventions": [{{
    "intervention_name": "specific name",
    "dosage": "string or null",
    "duration": "string or null",
    "frequency": "string or null",
    "intensity": "string or null",
    "administration_route": "string or null",
    "mechanism": "concise HOW it works (3-10 words)",
    "correlation_type": "positive|negative|neutral|inconclusive",
    "correlation_strength": "very strong|strong|moderate|weak|very weak|null",
    "delivery_method": "oral|injection|topical|behavioral|etc or null",
    "adverse_effects": "string or null",
    "extraction_confidence": "very high|high|medium|low|very low"
  }}]
}}]

KEY DISTINCTIONS:
- study_focus = WHAT they studied (research questions)
- measured_metrics = HOW they measured it (tools/instruments)
- findings = WHAT they found (results with data when available)

RULES:
1. One entry per condition studied
2. Multiple interventions for same condition → same entry, multiple objects in interventions array
3. Multiple conditions studied → separate entries
4. health_condition = specific complication, NOT underlying disease (e.g., "diabetes foot" not "diabetes")
5. mechanism = biological/behavioral pathway, NOT intervention name repeated
6. Extract 2-5 key findings with quantitative data when available
7. Return [] if no interventions found

EXAMPLES:

Paper: "Family-centered training for diabetes foot prevention in Jodhpur"
Abstract: "RCT with 54 diabetic patients per group. Intervention included family training for foot care. Knowledge scores: 13.4±1.2 vs 9.9±2.7 (p<0.001). Practice scores: 7.9±1.4 vs 6.2±1.3 (p<0.001). Foot ulcers: 0 vs 4 (8%)."

[{{
  "health_condition": "diabetes foot",
  "study_focus": ["foot care knowledge improvement", "foot ulcer prevention"],
  "measured_metrics": ["foot care knowledge scores", "practice scores", "foot ulcer incidence"],
  "findings": ["knowledge scores higher in intervention group (13.4±1.2 vs 9.9±2.7, p<0.001)", "practice scores higher (7.9±1.4 vs 6.2±1.3, p<0.001)", "zero ulcers in intervention vs 4 (8%) in control"],
  "study_location": "India",
  "publisher": null,
  "sample_size": 54,
  "study_duration": "9 months",
  "study_type": "randomized controlled trial",
  "population_details": "diabetic patients aged 18-60 and family members",
  "interventions": [{{
    "intervention_name": "family-centered foot care training",
    "dosage": null,
    "duration": "9 months",
    "frequency": null,
    "intensity": null,
    "administration_route": null,
    "mechanism": "improved adherence through family education",
    "correlation_type": "positive",
    "correlation_strength": "strong",
    "delivery_method": "behavioral",
    "adverse_effects": null,
    "extraction_confidence": "very high"
  }}]
}}]

---

Paper: "Probiotics for IBS: 8-week RCT"
Abstract: "120 IBS patients received Lactobacillus plantarum 10^9 CFU daily or placebo. Pain scores improved significantly (p<0.001). Microbiota analysis showed increased beneficial bacteria."

[{{
  "health_condition": "irritable bowel syndrome",
  "study_focus": ["probiotic efficacy on IBS symptoms", "gut microbiota changes"],
  "measured_metrics": ["abdominal pain scores", "gut microbiota composition", "inflammatory markers"],
  "findings": ["pain scores improved significantly (p<0.001)", "increased beneficial bacteria", "reduced inflammatory markers"],
  "study_location": null,
  "publisher": null,
  "sample_size": 120,
  "study_duration": "8 weeks",
  "study_type": "randomized controlled trial",
  "population_details": "IBS patients",
  "interventions": [{{
    "intervention_name": "Lactobacillus plantarum probiotic",
    "dosage": "10^9 CFU daily",
    "duration": "8 weeks",
    "frequency": "daily",
    "intensity": null,
    "administration_route": "oral",
    "mechanism": "gut microbiome modulation",
    "correlation_type": "positive",
    "correlation_strength": "strong",
    "delivery_method": "oral",
    "adverse_effects": null,
    "extraction_confidence": "very high"
  }}]
}}]

---

Paper: "Exercise vs diet vs combined for type 2 diabetes"
Abstract: "200 patients (50 per group). HbA1c reduction: exercise 0.8% (p<0.01), diet 1.1% (p<0.001), combined 1.5% (p<0.001). Weight loss: -3.1kg, -4.5kg, -8.2kg respectively."

[{{
  "health_condition": "type 2 diabetes",
  "study_focus": ["glycemic control efficacy", "weight management"],
  "measured_metrics": ["HbA1c levels", "body weight", "quality of life scores"],
  "findings": ["exercise reduced HbA1c by 0.8% (p<0.01)", "diet reduced HbA1c by 1.1% (p<0.001)", "combined reduced HbA1c by 1.5% (p<0.001)", "combined achieved greatest weight loss (-8.2kg)"],
  "study_location": null,
  "publisher": "Diabetes Care",
  "sample_size": 200,
  "study_duration": "12 months",
  "study_type": "randomized controlled trial",
  "population_details": "adults with type 2 diabetes",
  "interventions": [
    {{
      "intervention_name": "aerobic exercise",
      "dosage": null,
      "duration": "12 months",
      "frequency": null,
      "intensity": "moderate",
      "administration_route": null,
      "mechanism": "improved insulin sensitivity",
      "correlation_type": "positive",
      "correlation_strength": "moderate",
      "delivery_method": "behavioral",
      "adverse_effects": null,
      "extraction_confidence": "very high"
    }},
    {{
      "intervention_name": "Mediterranean diet",
      "dosage": null,
      "duration": "12 months",
      "frequency": "daily",
      "intensity": null,
      "administration_route": null,
      "mechanism": "reduced inflammation and improved metabolism",
      "correlation_type": "positive",
      "correlation_strength": "strong",
      "delivery_method": "dietary",
      "adverse_effects": null,
      "extraction_confidence": "very high"
    }},
    {{
      "intervention_name": "combined exercise and diet",
      "dosage": null,
      "duration": "12 months",
      "frequency": "daily",
      "intensity": "moderate",
      "administration_route": null,
      "mechanism": "synergistic metabolic improvement",
      "correlation_type": "positive",
      "correlation_strength": "very strong",
      "delivery_method": "behavioral",
      "adverse_effects": null,
      "extraction_confidence": "very high"
    }}
  ]
}}]

PAPER:
{paper_content}"""

    def create_system_message(self) -> str:
        """Create a standardized system message for LLM calls optimized for qwen3:14b."""
        return "Provide only the final JSON output without showing your reasoning process or using <think> tags. Start your response immediately with the [ character."


# Global prompt service instance
prompt_service = InterventionPromptService()