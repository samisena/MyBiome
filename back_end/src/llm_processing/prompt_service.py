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
        
        # Get category descriptions
        category_descriptions = self._get_category_descriptions()
        
        # Build the standardized prompt
        prompt = self._build_intervention_prompt(paper_content, category_descriptions)
        
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
    
    def _get_category_descriptions(self) -> Dict[str, str]:
        """Get intervention category descriptions from taxonomy."""
        category_descriptions = {}
        for cat_type in InterventionType:
            cat_def = self.taxonomy.get_category(cat_type)
            category_descriptions[cat_type.value] = cat_def.description
        return category_descriptions
    
    def _build_intervention_prompt(self, paper_content: str, category_descriptions: Dict[str, str]) -> str:
        """Build the complete intervention extraction prompt."""
        categories = [cat.value for cat in InterventionType]
        
        return f"""You are a biomedical expert analyzing research papers to extract health interventions and their relationships to health conditions.

PAPER:
{paper_content}

TASK: Extract ALL health interventions mentioned in this paper as a JSON array. Include interventions from these categories:

**EXERCISE** ({category_descriptions['exercise']}):
Examples: aerobic exercise, resistance training, yoga, walking, swimming, HIIT

**DIET** ({category_descriptions['diet']}):
Examples: Mediterranean diet, ketogenic diet, intermittent fasting, specific foods

**SUPPLEMENT** ({category_descriptions['supplement']}):
Examples: vitamin D, probiotics, omega-3, herbal supplements, minerals

**MEDICATION** ({category_descriptions['medication']}):
Examples: antidepressants, antibiotics, pain medications, hormones

**THERAPY** ({category_descriptions['therapy']}):
Examples: cognitive behavioral therapy, physical therapy, massage, acupuncture

**LIFESTYLE** ({category_descriptions['lifestyle']}):
Examples: sleep hygiene, stress management, smoking cessation, social support

**SURGERY** ({category_descriptions['surgery']}):
Examples: laparoscopic surgery, cardiac surgery, bariatric surgery, joint replacement

**EMERGING** ({category_descriptions['emerging']}):
Examples: gene therapy, digital therapeutics, precision medicine, AI-guided interventions

Return ONLY valid JSON. No extra text. Each intervention needs these fields:
- intervention_category: one of [{', '.join(f'"{cat}"' for cat in categories)}]
- intervention_name: specific intervention name (e.g., "Mediterranean diet", "aerobic exercise")
- intervention_details: object with category-specific details (duration, dosage, frequency, etc.)
- health_condition: specific condition being treated
- correlation_type: "positive", "negative", "neutral", or "inconclusive"
- correlation_strength: number 0.0-1.0 or null
- extraction_confidence: number 0.0-1.0 - YOUR confidence in extracting this information from the text
- study_confidence: number 0.0-1.0 - the AUTHORS' confidence in their findings/results or null
- sample_size: number or null
- study_duration: duration or null
- study_type: type of study or null
- population_details: population characteristics or null
- delivery_method: how intervention was delivered ("oral", "injection", "topical", "inhalation", "behavioral", "digital", etc.) or null
- severity: condition severity level ("mild", "moderate", "severe") or null
- adverse_effects: any reported side effects or complications or null
- cost_category: cost level ("low", "medium", "high") or null
- supporting_quote: relevant quote from text

IMPORTANT RULES:
- ONLY extract interventions where you can identify specific intervention names
- DO NOT use placeholders like "...", "intervention", "treatment", "therapy" (too generic)
- Each intervention_name must be specific (e.g., "cognitive behavioral therapy" not just "therapy")
- Match intervention_category correctly (exercise interventions go in "exercise", not "therapy")
- Use "emerging" category for novel interventions that don't fit existing categories (include proposed_category and category_rationale in intervention_details)
- Include intervention_details with category-specific information when available
- If no specific interventions are mentioned, return []

Example of valid extraction:
[{{
  "intervention_category": "exercise",
  "intervention_name": "aerobic exercise",
  "intervention_details": {{
    "exercise_type": "aerobic",
    "duration": "30 minutes",
    "frequency": "3 times per week",
    "intensity": "moderate"
  }},
  "health_condition": "depression",
  "correlation_type": "positive",
  "correlation_strength": 0.75,
  "extraction_confidence": 0.9,
  "study_confidence": 0.8,
  "sample_size": 120,
  "study_duration": "12 weeks",
  "study_type": "randomized controlled trial",
  "population_details": "adults aged 25-65 with moderate depression",
  "delivery_method": "behavioral",
  "severity": "moderate",
  "adverse_effects": "mild fatigue initially",
  "cost_category": "low",
  "supporting_quote": "Participants in the aerobic exercise group showed significant improvement in depression scores"
}}]

Return [] if no specific interventions are found."""

    def create_system_message(self) -> str:
        """Create a standardized system message for LLM calls."""
        return "You are a precise biomedical data extraction system. Return only valid JSON arrays with no additional formatting or text."


# Global prompt service instance
prompt_service = InterventionPromptService()