import json
from re import T
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
       
        elif 'groq' in self.config.model_name.lower() or 'mixtral' in self.config.model_name.lower():
            #* Groq is free tier
            self.config.cost_per_1k_input_tokens = 0
            self.config.cost_per_1k_output_tokens = 0
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

    
    def llm_prompt(self, paper: Dict) -> str:
        """ Returns the prompt used for correlation extraction
        
        Args:
            paper: Dictionary with 'pmid', 'title', and 'abstract' keys

        Returns:
            the prompt as a string
        """

        prompt = f"""Extract correlations between probiotic strains and health conditions from this paper.

    Title: {paper['title']}
    Abstract: {paper['abstract']}

    CRITICAL: Return ONLY a JSON array. Do not include any text, explanations, or thinking before or after the JSON.

    For each probiotic-health condition pair found, create a JSON object with these fields:
    - probiotic_strain: specific strain name (e.g., "Lactobacillus rhamnosus GG")
    - health_condition: specific condition studied
    - correlation_type: "positive" | "negative" | "neutral" | "inconclusive"
    - correlation_strength: 0.0 to 1.0
    - study_type: study design (e.g., "RCT", "meta_analysis")
    - sample_size: integer or null
    - study_duration: string or null
    - dosage: string or null
    - confidence_score: 0.0 to 1.0
    - supporting_quote: exact quote from abstract

    Return format - JSON array only:
    [
    {{
        "probiotic_strain": "...",
        "health_condition": "...",
        "correlation_type": "...",
        "correlation_strength": 0.5,
        "study_type": "...",
        "sample_size": 100,
        "study_duration": "...",
        "dosage": "...",
        "confidence_score": 0.8,
        "supporting_quote": "..."
    }}
    ]

    If no correlations found, return: []"""

        return prompt

    def parse_json_response(self, llm_text_output: str, paper_id):
        """ Parses the JSON response output from the LLM
        
        Args:
            llm_text_output: the JSON response
            paper_id: PMID for error reporting

        Returns:
            List of dictionary containing validated correlations
        """
        try:
            # Store original for debugging
            original_output = llm_text_output
            
            # DEBUG: Log the raw input
            self.logger.info(f"Raw LLM output for {paper_id} (first 500 chars):")
            self.logger.info(f"{original_output[:500]}")
            
            # Text cleaning:
            llm_text_output = llm_text_output.strip()  # removes white spaces
            
            # DEBUG: Log after stripping
            self.logger.info(f"After strip (first 200 chars): {llm_text_output[:200]}")

            if "```json" in llm_text_output:  # only leaves JSON content 
                self.logger.info("Found ```json markers, extracting...")
                llm_text_output = llm_text_output.split("```json")[1].split("```")[0]
            elif "```" in llm_text_output:
                self.logger.info("Found ``` markers, extracting...")
                llm_text_output = llm_text_output.split("```")[1].split("```")[0]
            
            # DEBUG: Log final cleaned text
            self.logger.info(f"Final cleaned text (first 200 chars): {llm_text_output[:200]}")
            self.logger.info(f"Length of cleaned text: {len(llm_text_output)}")

            # Check if empty
            if not llm_text_output:
                self.logger.error(f"Cleaned text is empty for {paper_id}")
                return []

            # converts JSON to Python list
            correlations = json.loads(llm_text_output)

            # Ensuring the LLM returned a JSON ARRAY as instructed
            if not isinstance(correlations, list):
                self.logger.error(f"Response for {paper_id} is not a list/array")
                return []

            # We iterate over each strain-condition relation found within a paper
            validated_correlations = []

            for corr in correlations: 
                validation_issues = []

                if not all(key in corr for key in ['probiotic_strain', 'health_condition', 'correlation_type']):
                    self.logger.warning(f"Missing required fields in correlation for {paper_id}")
                    continue

                if corr['correlation_type'] not in ['positive', 'negative', 'neutral', 'inconclusive']:
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

                # Add paper id and model to the correlations
                corr['paper_id'] = paper_id
                corr['extraction_model'] = self.config.model_name

                validated_correlations.append(corr)

            self.logger.info(f"Successfully parsed {len(validated_correlations)} correlations for {paper_id}")
            return validated_correlations
        
        except json.JSONDecodeError as e:
            self.logger.error(f"JSON parsing error for {paper_id}: {e}")
            self.logger.error(f"Failed to parse: {llm_text_output[:100]}...")
            return []
        except Exception as e:
            self.logger.error(f"Unexpected error parsing response for paper {paper_id}: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return []


    def extract_correlations(self, paper: Dict) -> List[Dict]:
        """ Extract correlations from a single paper using the LLM.
        
        Args:
            paper: dictionary with 'pmid', 'title', and 'abstract'

        Returns:
            List of extracted correlations in JSON formatting
        """
        if not paper.get('abstract') or not paper.get('title'):
            self.logger.warning(f"Paper {paper.get('pmid', 'unknown')} missing title or abstract")
            return []

        prompt = self.llm_prompt(paper)

        # Estimate the cost 
        input_cost, output_cost = self.estimate_cost(prompt)
        self.logger.info(f"Estimated cost for {paper['pmid']}: ${input_cost:.4f} + ${output_cost:.4f}")

        try:
            response = self.client.chat.completions.create(
                model=self.config.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a JSON-only response system. Output ONLY valid JSON arrays with no additional text, explanations, or markdown formatting. Do not include thinking or reasoning in your response."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens
            )

            # Extract response text
            response_text = response.choices[0].message.content
            
            # Check if response was cut off
            if hasattr(response.choices[0], 'finish_reason'):
                if response.choices[0].finish_reason == 'length':
                    self.logger.warning(f"Response truncated at max_tokens for {paper['pmid']}!")

            # Log token usage
            if hasattr(response, 'usage'):
                self.total_input_tokens += response.usage.prompt_tokens
                self.total_output_tokens += response.usage.completion_tokens
                self.logger.info(f"Tokens used for {paper['pmid']}: {response.usage.prompt_tokens} in, {response.usage.completion_tokens} out")

            # Parse JSON
            correlations = self.parse_json_response(response_text, paper['pmid'])

            if correlations:
                self.logger.info(f"Extracted {len(correlations)} correlations from {paper['pmid']}")
            else:
                self.logger.warning(f"No correlations extracted from {paper['pmid']}")

            return correlations
            
        except Exception as e:
            self.logger.error(f"API error for paper {paper['pmid']}: {e}")
            return []
        
    def process_papers(self, papers: List[Dict], save_to_db: bool = True):
        """
        Applies the extract_correlations method to multiple papers and optionally
        saves correlations to the database.

        Args:
            papers: List of paper dictionary objects
            save_to_db: save results to SQLite database or no

        Returns:
            A dictionary with containing a summary of the extraction process
        """

        all_correlations = []
        failed_papers = []
        
        self.logger.info(f"String extraction for {len(papers)} papers")

        for i, paper in enumerate(papers, 1):
            self.logger.info(f"\n--- Processing paper {i}/{len(papers)}: {paper['pmid']} ---")

            correlations = self.extract_correlations(paper)

            if correlations:
                all_correlations.extend(correlations)

                if save_to_db:
                    for corr in correlations:
                        try:
                            #Method from DatabaseManager Object 
                            corr['validation_status'] = 'pending'
                            self.db_manager.insert_correlation(corr) 
                        except Exception as e:
                            self.logger.error(f"Database error saving correlation: {e}")

            else:
                failed_papers.append(paper['pmid'])

                #* Avoid overwhelming the API (rate limiting error)
            if i < len(papers):
                time.sleep(1)

        total_cost = self.total_cost()

        results = {
            'total_papers': len(papers),
            'successful_papers': len(papers) - len(failed_papers),
            'failed_papers': failed_papers,
            'total_correlations': len(all_correlations),
            'total_cost': total_cost,
            'average_cost_per_paper': total_cost / len(papers) if papers else 0,
            'correlations': all_correlations
        }                
            
        self.logger.info(f"\n=== Extraction Complete ===")
        self.logger.info(f"Papers processed: {results['successful_papers']}/{results['total_papers']}")
        self.logger.info(f"Correlations found: {results['total_correlations']}")
        self.logger.info(f"Total cost: ${results['total_cost']:.4f}")
        self.logger.info(f"Average cost per paper: ${results['average_cost_per_paper']:.4f}")
        
        return results
        

                



    
  





        
        
