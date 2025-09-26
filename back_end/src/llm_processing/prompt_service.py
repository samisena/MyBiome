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
- confidence_score: number 0.0-1.0 or null
- study_type: type of study or null
- sample_size: number or null
- study_duration: duration or null
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
  "confidence_score": 0.9,
  "study_type": "randomized controlled trial",
  "sample_size": 120,
  "study_duration": "12 weeks",
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