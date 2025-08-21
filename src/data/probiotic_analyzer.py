import json
import time
import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass
from openai import OpenAI

from src.data.database_manager import DatabaseManager


#*Python automaticaly creates a constructor for the class
#* that includes several methods like __init__
@dataclass
class LLMConfig:
    """ Our LLM configuration data"""
    api_key: str      
    base_url: str
    model_name: str
    temperature: float
    max_tokens: int
    cost_per_1k_input_tokens: float
    cost_per_1k_output_tokens: float


class CorrelationExtractor:
    def __init__(self, config: LLMConfig, db_manager: DatabaseManager):
        """
        args:
        config: the LLM configuration data with the API key details
        db_manager: the SQLite database manager instance
        """
        self.config = config
        self.db_manager = db_manager

        #* We initialize the OpenAI client with our data
        self.client = OpenAI(
            api_key = config.api_key,
            base_url = config.base_url
        )

        #* Set up logging
        logging.basicConfig(level=logging.INFO) #? We set logs show only 
                                                #? INFO and above
        self.logger = logging.getLogger(__name__) #? Logs will start off with
                                            #? the name of the current module
        
        #* Cost tracking variables
        self.total_input_tokens = 0
        self.total_output_tokens = 0

        #* Approximate the cost based on the model
        self._set_cost_rates()


    def _set_cost_rates(self):
        """ Defines the token cost rate based on the current model"""
        if 'deepseek' in self.config.model_name.lower():
            #* DeepSeek pricing (example - check their actual rates)
            self.config.cost_per_1k_input_tokens = 0.0001
            self.config.cost_per_1k_output_tokens = 0.0002

        elif 'qwen' in self.config.model_name.lower():
            #* Qwen pricing (example - check their actual rates)
            self.config.cost_per_1k_input_tokens = 0.0002
            self.config.cost_per_1k_output_tokens = 0.0006

        self.logger.info((f"Cost rates set for {self.config.model_name}: "
                        f"${self.config.cost_per_1k_input_tokens}/1k input, "
                        f"${self.config.cost_per_1k_output_tokens}/1k output"))

            
    def estimate_cost(self, text, expected_output_tokens: int = 500
        ) -> Tuple[float,float]:
        """ Rough estimation of the cost of a text processing instance """

        #* Assuming 4 charcters per token
        nbre_of_input_tokens = len(text)/4

        input_token_cost = (nbre_of_input_tokens / 1000) * self.config.cost_per_1k_input_tokens

        output_token_cost = (expected_output_tokens / 1000) * self.config.cost_per_1k_output_tokens

        return input_token_cost, output_token_cost


    def total_cost(self):
        """ Returns the cost of all API calls amde so far"""

        input_cost = (self.total_input_tokens/1000)*self.config.cost_per_1k_input_tokens
        output_cost = (self.total_output_tokens/1000)*self.config.cost_per_1k_output_tokens
        return input_cost + output_cost

    
    def correlation_extraction_prompt(self, paper: Dict) -> str:
        """ Returns the prompt used for correlation extraction
        
        Args:
            paper: Dictionary with 'pmid', 'title', and 'abstract' keys

        Returns:
            the prompt as a string
         """

        prompt = f""" You are a biomedical research analyst specializing in probiotics
        studies. Exctract ALL correlations between probiotic strains and health
        conditions from this paper.

        Title: {paper['title']}
        Abstract: {paper['abstract']}

        For each probiotic-health condition pair found, extract:

        1. PROBIOTIC STRAIN: the specific strain name (e.g., "Lactobacillus rhamnosus GG")
            - If only genus/species given (e.g., "L. acidophilus"), record as is
            - If multiple strains tested together, list as "Strain1 + Strain2"
            - If generic "probiotics" mentioned, record as "probiotics (unspecified)"property

        2. HEALTH CONDITION: The specific condition or outcome studied 
            - Be specific (e.g., "antibiotic-associated diarrhea" not just "diarrhea")
            - Include the population if specified (e.g., "IBS in adults")

        3. CORRELATION TYPE: Classify as one of :
            -"positive": Probiotic improved the condition
            -"negative": Probiotic worsened the condition or caused adverse effects
            -"neutral": No significant effect observed
            -"inconclusive": Mixed or unclear results

        4. CORRELATION STRENGHT: Rate from 0.0 to 1.0
            - 0.0-0.3: Weak effect
            - 0.4-0.6: Moderate effect
            - 0.7-1.0: Strong effect

        5. STUDY TYPE: Identify the study design
            - Examples: "RCT", "meta_analysis", "observational", "case_control", "in-vitro"object

        6. SAMPLE SIZE: Exctract if mentioned (as an integer)

        7. STUDY DURATION: Exctract if mentioned (e.g., "12 weeks", "6 months")

        8. DOSAGE: Extract if mentioned (e.g., "10^9 CFU/day", "twice daily)

        9. CONFIDENCE SCORE: Your confidence in this insigh exctration (0.0 to 1.0)

        10. SUPPORTING QUOTE: The exact text from the abstract supporting this correlation

        Return ONLY a JSON array. Each element should have this struture:
        [
            {{
                "probiotic_strain": "exact strain name",
                "health_condition": "specific condition",
                "correlation_type": "positive|negative|neutral|inconclusive",
                "correlation_strength": 0.0-1.0,
                "study_type": "study design",
                "sample_size": null or integer,
                "study_duration": null or "duration string",
                "dosage": null or "dosage string",
                "confidence_score": 0.0-1.0,
                "supporting_quote": "exact quote from abstract"        
            }}
        ]

        Example for a hypothetical abstract:
        [
            {{
                "probiotic_strain": "Lactobacillus rhamnosus GG",
                "health_condition": "acute gastroenteritis in children",
                "correlation_type": "positive",
                "correlation_strength": 0.7,
                "study_type": "RCT",
                "sample_size": 124,
                "study_duration": "5 days",
                "dosage": "10^10 CFU twice daily",
                "confidence_score": 0.9,
                "supporting_quote": "LGG significantly reduced the duration of diarrhea compared to placebo (3.1 vs 5.2 days, p<0.001)"
            }}
        ]

        Important:
        - Extract ALL strain-condition pairs mentioned
        - If no correlations found, returan an empty array []
        - Ensure all JSON is properly formatted
        - Do not include any text outside the JSON array"""

        return prompt

    def parse_json_response(self, llm_text_output:str, paper_id):
        """"""

        try:

            #* Text cleaning:
            llm_text_output = llm_text_output.strip() #removes white spaces

            if "```json" in llm_text_output:  #only leaves JSON content 
                llm_text_output = llm_text_output.split("```json")[1].split("```")[0]
            elif "```" in llm_text_output:
                response_text = response_text.split("```")[1].split("```")[0]

            # converts JSON to Python list
            correlations = json.loads(llm_text_output)

            # Ensuring the LLM returned a JSON ARRAY as instrcuted
            if not isinstance(correlations, list):
                self.logger.error(f"Response for {paper_id} is not a list/array")
                return []

            # We iterrate over each strain-condition relation found within a paper
            # (we allowed for more than one per paper)
            valited_correlations = []

            # for every pair in the list, if one of these keys is missing:
            for corr in correlations: 

                validation_issues = []

                if not all(key in corr for key in ['probiotic_strain', 'health_condition', 'correlation_type']):
                    self.logger.warning(f"Missing required fields in correlation for {paper_id}")
                    continue

                if corr['correlation_type'] not in ['positive', 'negative',
                    'neutral', 'inconclusive']:
                    self.logger.warning(f"Invalid correlation type '{corr['correlation_type']}' for {paper_id}")
                    corr['correlation_type'] = 'inconclusive'

                if 'correlation_strength' in corr: 
                    try:
                        val = float(corr['correlation_strength'])
                        if not 0.0 <= val <= 1.0:
                            validation_issues.append(f"correlation_strength {val} out of range")
                        else:
                            corr['correlation_strength'] = val
                    except (ValueError, TypeError):
                        validation_issues.append(f"non-numeric correlation_strength: {corr['correlation_strength']}")
                        corr['correlation_strength'] = None



    
  





        
        
