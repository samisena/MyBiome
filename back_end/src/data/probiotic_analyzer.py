import json
import time
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from openai import OpenAI

from src.data.database_manager import DatabaseManager

@dataclass #automaticaly creates a constructor that includes several methods like __init__
class LLMConfig:
    """ Information about the LLM """
    api_key: str      #api key to acess the LLM
    base_url: str     #base url of the LLM
    model_name: str   
    temperature: float  #temperature setting (0: most predictable, 1: most creative)
    max_tokens: int    #maximum number of tokens in the response

class ProbioticAnalyzer:
    def __init__(self, config: LLMConfig, db_manager: DatabaseManager):
        """
        Args:
        config: the LLM configuration data with the API key details
        db_manager: the SQLite database manager instance
        """

        #* We set up the LLM details
        self.config = config

        #* Initialse the database manager object to interact with the database
        self.db_manager = db_manager

        #* The OpenAI client is used to interact with LLMs via their APIs (not juts GPTs)
        self.client = OpenAI(
            api_key = config.api_key,
            base_url = config.base_url
        )

        #* Setting up logging
        logging.basicConfig(level=logging.INFO) #? We set logs show only 
                                                #? INFO and above
        self.logger = logging.getLogger(__name__) #? Logs will start off with
                                            #? the name of the current module
        
        #* Cost tracking variables according to tokens used
        self.total_input_tokens = 0   #tokens sent to the LLM as a prompt
        self.total_output_tokens = 0   #tokens received from the LLM 

    
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

    Return format - JSON array only: [{{"probiotic_strain":"...","health_condition":"...",...}}]

    If no correlations found, return: []"""

        return prompt


    def parse_json_response(self, llm_text_output: str, paper_id):
        """
        Parses the JSON response output from the LLM
        
        Args:
            llm_text_output: the JSON response
            paper_id: PMID for error reporting

        Returns:
            List of dictionaries containing validated correlations that were extracted by the LLM
        """

        try:
            #* Stores raw LLM output for debugging
            original_output = llm_text_output  #'[{"probiotic_strain": "Lactobacillus", "health_condition": "IBS", ...}]'

            self.logger.info(f"Raw LLM output for {paper_id} (first 500 chars):")
            self.logger.info(f"{original_output[:500]}")
            
            #* Text cleaning:
            llm_text_output = llm_text_output.strip()  # removes white spaces
            self.logger.info(f"After strip (first 200 chars): {llm_text_output[:200]}")

            #? grok marks its responses with certain markers
            if "```json" in llm_text_output:  # only leaves JSON content 
                self.logger.info("Found ```json markers, extracting...")
                llm_text_output = llm_text_output.split("```json")[1].split("```")[0]
            elif "```" in llm_text_output:
                self.logger.info("Found ``` markers, extracting...")
                llm_text_output = llm_text_output.split("```")[1].split("```")[0]
            
            #* DEBUG: Log final cleaned text
            self.logger.info(f"Final cleaned text (first 200 chars): {llm_text_output[:200]}")
            self.logger.info(f"Length of cleaned text: {len(llm_text_output)}")

            #* Check if empty
            if not llm_text_output:
                self.logger.error(f"Cleaned text is empty for {paper_id}")
                return []

            #* converts the LLM output (string in JSON format) to Python Object
            #* We expect a list of dictionaries, where each dictionary represents a paper 
            correlations = json.loads(llm_text_output)
            #* Ensuring the LLM response is now in list format
            if not isinstance(correlations, list):
                self.logger.error(f"Response for {paper_id} is not a list/array")
                return []

            #* We iterate over each strain-condition relation found within a paper
            validated_correlations = []
            for corr in correlations:   #for each dictionary (paper) in the list of papers
                validation_issues = []

                #* Checking if one of the key fields is empty
                if not all(key in corr for key in ['probiotic_strain', 'health_condition', 'correlation_type']):
                    self.logger.warning(f"Missing required fields in correlation for {paper_id}")
                    continue

                #* Checking if the correlation type is not 1 of the 4 options
                if corr['correlation_type'] not in ['positive', 'negative', 'neutral', 'inconclusive']:
                    self.logger.warning(f"Invalid correlation type '{corr['correlation_type']}' for {paper_id}")
                    corr['correlation_type'] = 'inconclusive'

                #* Checking if the correlation strength number is between 0 and 1
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

                #* Adds the paper id and the LLM model to the correlations list of dictionaries
                corr['paper_id'] = paper_id
                corr['extraction_model'] = self.config.model_name
                validated_correlations.append(corr) # marks this operation as a success

            self.logger.info(f"Successfully parsed {len(validated_correlations)} correlations for {paper_id}")
            return validated_correlations
        
        #* Error logging if json.load() fails to convert JSON output to Python Object
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
        """ 
        Extract correlations from a single paper using the LLM.
        
        Args:
            paper: dictionary with 'pmid', 'title', and 'abstract'

        Returns:
            List of extracted correlations in JSON formatting
        """

        #* Checking if the dictionary (paper) has an abstarct and a title
        if not paper.get('abstract') or not paper.get('title'):  #.get() to avoid errors crashing the code
            self.logger.warning(f"Paper {paper.get('pmid', 'unknown')} missing title or abstract")
            return []

        #* Loads our prompt method
        prompt = self.llm_prompt(paper)

        #* Loads the OpenAI client API with the model of our choice and passes the paper dictionary
        #* in the prompt
        try:
            response = self.client.chat.completions.create(
                model=self.config.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a JSON-only response system. Output ONLY valid JSON arrays with no additional text, explanations, or markdown formatting. Do not include thinking or reasoning in your response."
                    },
                    {"role": "user", "content": prompt}  #We pass the prompt with the paper and abstract
                ],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens
            )

            #* Extract the response text that's coming back from the LLM
            response_text = response.choices[0].message.content
            
            #* Check if the response was cut off due to token limitation
            if hasattr(response.choices[0], 'finish_reason'):
                if response.choices[0].finish_reason == 'length':
                    self.logger.warning(f"Response truncated at max_tokens for {paper['pmid']}!")

            #* Log token usage
            if hasattr(response, 'usage'):
                self.total_input_tokens += response.usage.prompt_tokens
                self.total_output_tokens += response.usage.completion_tokens
                self.logger.info(f"Tokens used for {paper['pmid']}: {response.usage.prompt_tokens} in, {response.usage.completion_tokens} out")

            #* Parse the JSON string response to Python List of Dictionaries
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
        Applies extract_correlations method to multiple papers (batch) and optionally
        saves correlations to the database.

        Args:
            papers: List of paper dictionary objects
            save_to_db(bool): save results to SQLite database, defaults to True

        Returns:
            A dictionary with containing a summary of the extraction process and saves results to the 
            databse.
        """

        all_correlations = []
        failed_papers = []
        
        self.logger.info(f"String extraction for {len(papers)} papers")

        #* Begins the probiotic-condition extraction process:
        for i, paper in enumerate(papers, 1):
            self.logger.info(f"\n--- Processing paper {i}/{len(papers)}: {paper['pmid']} ---")
            correlations = self.extract_correlations(paper)  #sends the paper details to the LLM API
                                                            # and gets back a list of dictionaries
                                                            # with correlations
            if correlations:
                all_correlations.extend(correlations) #? .extend() adds individual dictionaries
                                                    #? instead of everything as a string (.append()) 
                if save_to_db:  #if we are saving the results to the database
                    for corr in correlations:
                        try:
                            corr['validation_status'] = 'pending'  #indicates that the correlation is 
                                                            # awaiting review by a second model
                            self.db_manager.insert_correlation(corr) #Method from DatabaseManager Object
                        except Exception as e:
                            self.logger.error(f"Database error saving correlation: {e}")
            #* In case of error adds the paper ID to a list of papers that failed the correlation
            #* extraction process:          
            else:
                failed_papers.append(paper['pmid'])

            #* Avoid overwhelming the API (rate limiting error)
            if i < len(papers):
                time.sleep(1)

        #* The summary output
        results = {
            'total_papers': len(papers),
            'successful_papers': len(papers) - len(failed_papers),
            'failed_papers': failed_papers,
            'total_correlations': len(all_correlations),
            'correlations': all_correlations
        }                
            
        self.logger.info(f"\n=== Extraction Complete ===")
        self.logger.info(f"Papers processed: {results['successful_papers']}/{results['total_papers']}")
        self.logger.info(f"Correlations found: {results['total_correlations']}")
        self.logger.info(f"Average cost per paper: ${results['average_cost_per_paper']:.4f}")
        
        return results
        

                



    
  





        
        
