

from src.data.probiotic_analyzer import *


class CorrelationVerifier:
    def __init__(self, config: LLMConfig, db_manager: DatabaseManager):
        """Initialize verifier with different model than extractor."""
        self.config = config
        self.db_manager = db_manager
        
        # Initialize OpenAI client
        self.client = OpenAI(
            api_key=config.api_key,
            base_url=config.base_url
        )
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def create_verification_prompt(self, correlation: Dict, 
                                  paper_abstract: str) -> str:
        """Create prompt to verify an existing correlation extraction."""
        
        prompt = f"""You are a biomedical research verification specialist. 
        Your task is to verify if the following correlation extraction is accurate.

        ORIGINAL PAPER ABSTRACT:
        {paper_abstract}

        EXTRACTED CORRELATION TO VERIFY:
        - Probiotic: {correlation['probiotic_strain']}
        - Condition: {correlation['health_condition']}
        - Type: {correlation['correlation_type']}
        - Strength: {correlation['correlation_strength']}
        - Study Type: {correlation.get('study_type', 'Not specified')}
        - Sample Size: {correlation.get('sample_size', 'Not specified')}
        - Supporting Quote: {correlation.get('supporting_quote', 'None provided')}

        Please verify:
        1. Is the probiotic strain correctly identified?
        2. Is the health condition accurately described?
        3. Is the correlation type (positive/negative/neutral) correct?
        4. Is the correlation strength reasonable given the abstract?
        5. Does the supporting quote actually exist in the abstract?

        Return JSON with this structure:
        {{
            "is_valid": true/false,
            "confidence": 0.0-1.0,
            "issues": [],  // List of any problems found
            "corrections": {{}}  // Suggested corrections for any fields
        }}
        """
        
        return prompt
    
    def verify_correlation(self, correlation: Dict) -> Dict:
        """Verify a single correlation using a different model."""
        
        try:
            prompt = self.create_verification_prompt(
                correlation, 
                correlation['abstract']
            )
            
            response = self.client.chat.completions.create(
                model=self.config.model_name,
                messages=[
                    {"role": "system", "content": "You are a precise verification system."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,  # Lower temperature for consistency
                max_tokens=500
            )
            
            result_text = response.choices[0].message.content
            
            # Parse JSON response
            if "```json" in result_text:
                result_text = result_text.split("```json")[1].split("```")[0]
            
            result = json.loads(result_text.strip())
            
            # Determine validation status
            if result['is_valid']:
                if result['confidence'] >= 0.8:
                    status = 'verified'
                else:
                    status = 'verified'  # But with lower confidence
            else:
                if len(result.get('issues', [])) > 2:
                    status = 'failed'
                else:
                    status = 'conflicted'
            
            return {
                'validation_status': status,
                'validation_issues': json.dumps(result.get('issues', [])),
                'verification_model': self.config.model_name,
                'verification_confidence': result['confidence']
            }
            
        except Exception as e:
            self.logger.error(f"Verification error: {e}")
            return {
                'validation_status': 'failed',
                'validation_issues': str(e),
                'verification_model': self.config.model_name,
                'verification_confidence': 0.0
            }
    
    def batch_verify(self, limit: int = 10):
        """Verify a batch of pending correlations."""
        
        correlations = self.db_manager.get_correlations_for_verification(limit)
        
        self.logger.info(f"Starting verification of {len(correlations)} correlations")
        
        results = {
            'verified': 0,
            'conflicted': 0,
            'failed': 0
        }
        
        for corr in correlations:
            self.logger.info(f"Verifying correlation {corr['id']}")
            
            verification = self.verify_correlation(corr)
            
            success = self.db_manager.update_correlation_verification(
                corr['id'], 
                verification
            )
            
            if success:
                results[verification['validation_status']] = \
                    results.get(verification['validation_status'], 0) + 1
            
            time.sleep(1)  # Rate limiting
        
        self.logger.info(f"Verification complete: {results}")
        return results