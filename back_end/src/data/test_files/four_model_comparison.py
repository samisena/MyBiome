import os
import sys
import time
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
from enum import Enum
from dotenv import load_dotenv

from src.data.pubmed_collector_enhanced import PubMedCollector
from src.data.database_manager_enhanced import DatabaseManager
from src.data.probiotic_analyzer_enhanced import ProbioticAnalyzer, LLMConfig
from src.data.test_files.llm_pipeline import OllamaConfig

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('four_model_comparison.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TestModel(Enum):
    """Enum for the 4 models to test."""
    LLAMA_3_1_8B = "llama3.1:8b"
    BIOMISTRAL = "cniongolo/biomistral"
    QWEN_2_5_7B = "qwen2.5:7b"
    GEMMA2_9B = "gemma2:9b"

class FourModelComparison:
    """
    A class to compare 4 different LLM models on their ability to extract 
    relationships between probiotics and health conditions.
    """
    
    def __init__(self, ollama_base_url: str = "http://localhost:11434/v1"):
        """
        Initialize the comparison with all 4 models.
        
        Args:
            ollama_base_url: Ollama server URL
        """
        self.pubmed_collector = PubMedCollector()
        self.db_manager = DatabaseManager()
        self.ollama_base_url = ollama_base_url
        
        # Initialize models
        self.models = {
            'llama3.1': TestModel.LLAMA_3_1_8B,
            'biomistral': TestModel.BIOMISTRAL,
            'qwen2.5': TestModel.QWEN_2_5_7B,
            'gemma2': TestModel.GEMMA2_9B
        }
        
        # Create configurations for all models
        self.configs = {}
        for model_key, model_enum in self.models.items():
            self.configs[model_key] = OllamaConfig.create_config(
                model=model_enum,
                base_url=ollama_base_url
            )
        
        # Initialize results structure
        self.results = {
            'test_info': {
                'models_tested': [model.value for model in self.models.values()],
                'ollama_url': ollama_base_url,
                'timestamp': time.time()
            },
            'data_collection': {},
            'model_results': {},
            'comparison': {},
            'summary': {}
        }
    
    def validate_setup(self) -> bool:
        """
        Validate that all models are available and system is ready.
        """
        logger.info("Validating setup for all 4 models...")
        
        # Check NCBI API key
        ncbi_key = os.getenv('NCBI_API_KEY')
        if not ncbi_key:
            logger.error("Missing NCBI_API_KEY for PubMed access")
            return False
        logger.info("✓ NCBI_API_KEY found")
        
        # Check each model availability
        for model_key, model_enum in self.models.items():
            if not OllamaConfig.check_ollama_server_and_model(model_enum, self.ollama_base_url):
                logger.error(f"✗ Model {model_enum.value} not available")
                logger.error(f"Pull the model with: ollama pull {model_enum.value}")
                return False
            logger.info(f"✓ Model {model_enum.value} available")
        
        logger.info("All setup validation passed!")
        return True
    
    def collect_test_data(self, condition: str, max_papers: int = 5) -> List[Dict]:
        """
        Collect test data from PubMed for the specified condition.
        
        Args:
            condition: Health condition to search for
            max_papers: Maximum number of papers to collect
            
        Returns:
            List of papers for processing
        """
        logger.info(f"Collecting papers for condition: {condition}")
        logger.info(f"Maximum papers to collect: {max_papers}")
        
        try:
            collection_result = self.pubmed_collector.collect_probiotics_by_condition(
                condition=condition,
                max_results=max_papers
            )
            
            # Store collection info
            self.results['data_collection'] = {
                'condition': condition,
                'requested_papers': max_papers,
                'papers_found': collection_result.get('paper_count', 0),
                'status': collection_result.get('status', 'unknown'),
                'metadata_file': collection_result.get('metadata_file', '')
            }
            
            logger.info(f"Data collection completed:")
            logger.info(f"  - Condition: {condition}")
            logger.info(f"  - Papers found: {collection_result.get('paper_count', 0)}")
            logger.info(f"  - Status: {collection_result.get('status', 'unknown')}")
            
            # Get papers from database if collection was successful
            if collection_result.get('status') == 'success' and collection_result.get('paper_count', 0) > 0:
                papers = self.db_manager.get_papers_by_condition(condition)
                logger.info(f"Retrieved {len(papers)} papers from database for processing")
                return papers[:max_papers]
            else:
                logger.warning("No papers collected or collection failed")
                return []
                
        except Exception as e:
            logger.error(f"Error during paper collection: {e}")
            self.results['data_collection']['error'] = str(e)
            return []
    
    def process_with_model(self, model_key: str, papers: List[Dict]) -> Dict:
        """
        Process papers with a specific model.
        
        Args:
            model_key: Key identifying the model (e.g., 'llama3.1', 'biomistral')
            papers: List of papers to process
            
        Returns:
            Processing results for the model
        """
        model_enum = self.models[model_key]
        config = self.configs[model_key]
        
        logger.info(f"Starting processing with {model_enum.value}...")
        
        try:
            analyzer = ProbioticAnalyzer(config, self.db_manager)
            
            start_time = time.time()
            results = self._process_papers_with_retry(analyzer, papers, save_to_db=True)
            end_time = time.time()
            
            processing_time = end_time - start_time
            
            # Store results for this model
            model_results = {
                'model_name': model_enum.value,
                'model_key': model_key,
                'papers_processed': results['successful_papers'],
                'total_papers': results['total_papers'],
                'failed_papers': results['failed_papers'],
                'correlations_found': results['total_correlations'],
                'processing_time_seconds': processing_time,
                'input_tokens': analyzer.total_input_tokens,
                'output_tokens': analyzer.total_output_tokens,
                'correlations': results['correlations']
            }
            
            self.results['model_results'][model_key] = model_results
            
            logger.info(f"{model_enum.value} processing completed:")
            logger.info(f"  - Papers processed: {results['successful_papers']}/{results['total_papers']}")
            logger.info(f"  - Correlations found: {results['total_correlations']}")
            logger.info(f"  - Processing time: {processing_time:.2f} seconds")
            
            return results
            
        except Exception as e:
            logger.error(f"Error processing with {model_enum.value}: {e}")
            error_results = {
                'model_name': model_enum.value,
                'model_key': model_key,
                'error': str(e),
                'papers_processed': 0,
                'total_papers': len(papers),
                'failed_papers': [p.get('pmid', 'unknown') for p in papers],
                'correlations_found': 0,
                'correlations': []
            }
            self.results['model_results'][model_key] = error_results
            return error_results
    
    def _process_papers_with_retry(self, analyzer: ProbioticAnalyzer, 
                                   papers: List[Dict], save_to_db: bool, 
                                   max_retries: int = 2) -> Dict:
        """Process papers with retry logic for robustness."""
        for attempt in range(max_retries + 1):
            try:
                results = analyzer.process_papers(papers, save_to_db)
                return results
            except Exception as e:
                if attempt < max_retries:
                    logger.warning(f"Processing attempt {attempt+1} failed: {e}. Retrying...")
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    logger.error(f"All processing attempts failed: {e}")
                    raise
    
    def run_comparison(self, condition: str = "IBS", max_papers: int = 5) -> bool:
        """
        Run the complete comparison of all 4 models.
        
        Args:
            condition: Health condition to search for
            max_papers: Maximum number of papers to process
            
        Returns:
            True if comparison completed successfully
        """
        logger.info("="*60)
        logger.info("STARTING FOUR MODEL COMPARISON")
        logger.info("="*60)
        
        # Step 1: Validate setup
        if not self.validate_setup():
            logger.error("Setup validation failed. Exiting...")
            return False
        
        # Step 2: Collect test data
        papers = self.collect_test_data(condition, max_papers)
        if not papers:
            logger.error("No papers collected. Exiting...")
            return False
        
        # Step 3: Process with each model
        for model_key in self.models.keys():
            logger.info(f"\n{'='*30}")
            logger.info(f"Processing with {self.models[model_key].value}")
            logger.info(f"{'='*30}")
            self.process_with_model(model_key, papers)
        
        # Step 4: Generate comparison and summary
        self.generate_comparison()
        self.generate_summary()
        
        # Step 5: Save results
        results_file = self.save_results()
        self.print_detailed_summary()
        
        logger.info("="*60)
        logger.info(f"Four model comparison completed successfully!")
        logger.info(f"Results saved to: {results_file}")
        logger.info("="*60)
        
        return True
    
    def generate_comparison(self):
        """Generate detailed comparison between all 4 models."""
        logger.info("Generating comparison between all models...")
        
        # Extract strains found by each model
        model_strains = {}
        model_correlations = {}
        
        for model_key, results in self.results['model_results'].items():
            correlations = results.get('correlations', [])
            strains = set()
            
            for corr in correlations:
                if 'probiotic_strain' in corr:
                    strains.add(corr['probiotic_strain'])
            
            model_strains[model_key] = strains
            model_correlations[model_key] = len(correlations)
        
        # Find common strains across all models
        if model_strains:
            all_strains = set.union(*model_strains.values()) if model_strains.values() else set()
            common_all = set.intersection(*model_strains.values()) if model_strains.values() else set()
        else:
            all_strains = set()
            common_all = set()
        
        # Pairwise comparisons
        pairwise_comparisons = {}
        model_keys = list(self.models.keys())
        
        for i, model1 in enumerate(model_keys):
            for j, model2 in enumerate(model_keys):
                if i < j:  # Avoid duplicate comparisons
                    strains1 = model_strains.get(model1, set())
                    strains2 = model_strains.get(model2, set())
                    
                    common = strains1.intersection(strains2)
                    union = strains1.union(strains2)
                    agreement = (len(common) / max(len(union), 1)) * 100
                    
                    pair_key = f"{model1}_vs_{model2}"
                    pairwise_comparisons[pair_key] = {
                        'model1': self.models[model1].value,
                        'model2': self.models[model2].value,
                        'common_strains': len(common),
                        'total_unique_strains': len(union),
                        'agreement_percentage': agreement,
                        'model1_only': len(strains1 - strains2),
                        'model2_only': len(strains2 - strains1)
                    }
        
        # Store comparison results
        self.results['comparison'] = {
            'total_unique_strains_all_models': len(all_strains),
            'strains_found_by_all_models': len(common_all),
            'common_strains_list': list(common_all),
            'model_strain_counts': {k: len(v) for k, v in model_strains.items()},
            'model_correlation_counts': model_correlations,
            'pairwise_comparisons': pairwise_comparisons,
            'model_strain_details': {k: list(v) for k, v in model_strains.items()}
        }
        
        logger.info("Comparison analysis completed")
    
    def generate_summary(self):
        """Generate a summary of the comparison results."""
        comparison = self.results.get('comparison', {})
        
        # Find best performing model by correlations
        correlation_counts = comparison.get('model_correlation_counts', {})
        best_correlations_model = max(correlation_counts.items(), key=lambda x: x[1]) if correlation_counts else (None, 0)
        
        # Find best performing model by unique strains
        strain_counts = comparison.get('model_strain_counts', {})
        best_strains_model = max(strain_counts.items(), key=lambda x: x[1]) if strain_counts else (None, 0)
        
        # Calculate processing times
        processing_times = {}
        for model_key, results in self.results['model_results'].items():
            processing_times[model_key] = results.get('processing_time_seconds', 0)
        fastest_model = min(processing_times.items(), key=lambda x: x[1]) if processing_times else (None, 0)
        
        self.results['summary'] = {
            'best_correlations_extractor': {
                'model': best_correlations_model[0],
                'correlations_found': best_correlations_model[1]
            },
            'best_strain_identifier': {
                'model': best_strains_model[0],
                'unique_strains_found': best_strains_model[1]
            },
            'fastest_processor': {
                'model': fastest_model[0],
                'processing_time_seconds': fastest_model[1]
            },
            'total_unique_relationships': comparison.get('total_unique_strains_all_models', 0),
            'consensus_relationships': comparison.get('strains_found_by_all_models', 0)
        }
    
    def save_results(self, output_file: str = None) -> str:
        """Save the comparison results to a JSON file."""
        if output_file is None:
            timestamp = int(time.time())
            output_file = f"four_model_comparison_results_{timestamp}.json"
        
        current_dir = Path(__file__).parent
        output_path = current_dir / output_file
        
        # Add metadata
        self.results['metadata'] = {
            'test_timestamp': time.time(),
            'test_date': time.strftime('%Y-%m-%d %H:%M:%S'),
            'python_version': sys.version,
            'ollama_base_url': self.ollama_base_url,
            'models_tested': [model.value for model in self.models.values()]
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        logger.info(f"Results saved to: {output_path}")
        return str(output_path)
    
    def print_detailed_summary(self):
        """Print a detailed summary of all results."""
        print("\n" + "="*80)
        print("FOUR MODEL COMPARISON SUMMARY")
        print("="*80)
        
        # Data Collection Summary
        dc = self.results.get('data_collection', {})
        print(f"\nDATA COLLECTION:")
        print(f"  Condition searched: {dc.get('condition', 'Unknown')}")
        print(f"  Papers found: {dc.get('papers_found', 0)}")
        print(f"  Status: {dc.get('status', 'Unknown')}")
        
        # Individual Model Results
        print(f"\nINDIVIDUAL MODEL RESULTS:")
        print("-" * 40)
        
        for model_key, results in self.results['model_results'].items():
            model_name = results.get('model_name', 'Unknown')
            print(f"\n{model_name.upper()}:")
            print(f"  Papers processed: {results.get('papers_processed', 0)}/{results.get('total_papers', 0)}")
            print(f"  Relationships found: {results.get('correlations_found', 0)}")
            print(f"  Processing time: {results.get('processing_time_seconds', 0):.2f}s")
            if 'error' in results:
                print(f"  Error: {results['error']}")
        
        # Comparison Results
        comp = self.results.get('comparison', {})
        print(f"\nCOMPARISON RESULTS:")
        print("-" * 40)
        print(f"  Total unique relationships found: {comp.get('total_unique_strains_all_models', 0)}")
        print(f"  Relationships found by ALL models: {comp.get('strains_found_by_all_models', 0)}")
        
        # Model Rankings
        summary = self.results.get('summary', {})
        print(f"\nMODEL RANKINGS:")
        print("-" * 40)
        
        best_corr = summary.get('best_correlations_extractor', {})
        if best_corr.get('model'):
            model_name = self.models[best_corr['model']].value
            print(f"  Most relationships extracted: {model_name} ({best_corr.get('correlations_found', 0)} relationships)")
        
        best_strain = summary.get('best_strain_identifier', {})
        if best_strain.get('model'):
            model_name = self.models[best_strain['model']].value
            print(f"  Most unique strains found: {model_name} ({best_strain.get('unique_strains_found', 0)} strains)")
        
        fastest = summary.get('fastest_processor', {})
        if fastest.get('model'):
            model_name = self.models[fastest['model']].value
            print(f"  Fastest processing: {model_name} ({fastest.get('processing_time_seconds', 0):.2f}s)")
        
        # Pairwise Comparisons
        pairwise = comp.get('pairwise_comparisons', {})
        if pairwise:
            print(f"\nPAIRWISE AGREEMENT:")
            print("-" * 40)
            for pair_key, pair_data in pairwise.items():
                model1 = pair_data['model1']
                model2 = pair_data['model2']
                agreement = pair_data['agreement_percentage']
                print(f"  {model1} vs {model2}: {agreement:.1f}% agreement")
        
        print("="*80)


def main(condition: str = "IBS", max_papers: int = 5, 
         ollama_url: str = "http://localhost:11434/v1"):
    """
    Main function to run the four model comparison.
    
    Args:
        condition: Health condition to search for
        max_papers: Maximum papers to process
        ollama_url: Ollama server URL
    """
    print("Starting Four Model Comparison...")
    print(f"Models to test: {[model.value for model in TestModel]}")
    print(f"Condition: {condition}")
    print(f"Max papers: {max_papers}")
    
    comparator = FourModelComparison(ollama_url)
    success = comparator.run_comparison(condition, max_papers)
    
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)