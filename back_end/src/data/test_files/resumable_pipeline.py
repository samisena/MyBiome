import os
import sys
import time
import json
import signal
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Any
from enum import Enum
from dotenv import load_dotenv

from src.data.pubmed_collector import PubMedCollector
from src.data.database_manager import DatabaseManager
from src.data.probiotic_analyzer import ProbioticAnalyzer, LLMConfig
from src.data.test_files.llm_pipeline import LLMPipeline, OllamaConfig, OllamaModel

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('resumable_pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class PipelineState(Enum):
    """Pipeline execution states"""
    INITIALIZING = "initializing"
    COLLECTING_PAPERS = "collecting_papers"
    PROCESSING_PRIMARY = "processing_primary"
    PROCESSING_SECONDARY = "processing_secondary"
    COMPLETED = "completed"
    FAILED = "failed"


class ResumablePipeline:
    """
    A resumable pipeline that can be stopped and restarted at any point.
    Saves state to disk and can continue exactly where it left off.
    """
    
    def __init__(self, condition: str, max_papers: int, 
                 primary_model: str, secondary_model: str,
                 ollama_url: str = "http://localhost:11434/v1",
                 state_file: str = None):
        """
        Initialize the resumable pipeline.
        
        Args:
            condition: Health condition to search for
            max_papers: Maximum papers to collect
            primary_model: Primary Ollama model name
            secondary_model: Secondary Ollama model name
            ollama_url: Ollama server URL
            state_file: Path to state file (auto-generated if None)
        """
        self.condition = condition
        self.max_papers = max_papers
        self.primary_model_str = primary_model
        self.secondary_model_str = secondary_model
        self.ollama_url = ollama_url
        
        # Generate state file name if not provided
        if state_file is None:
            safe_condition = "".join(c for c in condition if c.isalnum() or c in (' ', '-', '_')).rstrip()
            safe_condition = safe_condition.replace(' ', '_')
            self.state_file = f"pipeline_state_{safe_condition}_{max_papers}papers.json"
        else:
            self.state_file = state_file
            
        self.state_path = Path(self.state_file)
        self.shutdown_requested = False
        
        # Initialize pipeline state
        self.state = {
            'condition': condition,
            'max_papers': max_papers,
            'primary_model': primary_model,
            'secondary_model': secondary_model,
            'ollama_url': ollama_url,
            'current_stage': PipelineState.INITIALIZING.value,
            'start_time': None,
            'last_update': None,
            'papers_collected': [],
            'papers_processed_primary': 0,
            'papers_processed_secondary': 0,
            'primary_results': {},
            'secondary_results': {},
            'errors': [],
            'completed': False
        }
        
        # Initialize components (will be set up in validate_setup)
        self.pubmed_collector = None
        self.db_manager = None
        self.primary_analyzer = None
        self.secondary_analyzer = None
        
        # Set up signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        logger.info(f"Resumable pipeline initialized for condition: {condition}")
        logger.info(f"State file: {self.state_path}")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        logger.info(f"Received signal {signum}. Shutting down gracefully...")
        self.shutdown_requested = True
        self._save_state()
        logger.info("State saved. Pipeline can be resumed later.")
        sys.exit(0)
    
    def _save_state(self):
        """Save current state to disk"""
        self.state['last_update'] = time.time()
        try:
            with open(self.state_path, 'w') as f:
                json.dump(self.state, f, indent=2, default=str)
            logger.debug(f"State saved to {self.state_path}")
        except Exception as e:
            logger.error(f"Failed to save state: {e}")
    
    def _load_state(self) -> bool:
        """Load state from disk if it exists"""
        if not self.state_path.exists():
            logger.info("No previous state found. Starting fresh.")
            return False
        
        try:
            with open(self.state_path, 'r') as f:
                saved_state = json.load(f)
            
            # Validate that saved state matches current parameters
            if (saved_state.get('condition') != self.condition or
                saved_state.get('max_papers') != self.max_papers or
                saved_state.get('primary_model') != self.primary_model_str or
                saved_state.get('secondary_model') != self.secondary_model_str):
                logger.warning("Saved state parameters don't match current parameters.")
                response = input("Do you want to continue with saved state (y) or start fresh (n)? ")
                if response.lower() != 'y':
                    return False
            
            self.state.update(saved_state)
            logger.info(f"Loaded previous state from {self.state_path}")
            logger.info(f"Previous stage: {self.state['current_stage']}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load state: {e}")
            return False
    
    def validate_setup(self) -> bool:
        """Validate setup and initialize components"""
        logger.info("Validating setup...")
        
        # Check NCBI API key
        ncbi_key = os.getenv('NCBI_API_KEY')
        if not ncbi_key:
            logger.error("Missing NCBI_API_KEY for PubMed access")
            return False
        
        # Initialize components
        try:
            self.pubmed_collector = PubMedCollector()
            self.db_manager = DatabaseManager()
            
            # Convert string model names to enum values
            primary_enum = OllamaModel(self.primary_model_str)
            secondary_enum = OllamaModel(self.secondary_model_str)
            
            # Create LLM configs
            primary_config = OllamaConfig.create_config(primary_enum, self.ollama_url)
            secondary_config = OllamaConfig.create_config(secondary_enum, self.ollama_url)
            
            # Check Ollama models availability
            if not OllamaConfig.check_ollama_server_and_model(primary_enum, self.ollama_url):
                logger.error(f"Primary model {self.primary_model_str} not available")
                return False
                
            if not OllamaConfig.check_ollama_server_and_model(secondary_enum, self.ollama_url):
                logger.error(f"Secondary model {self.secondary_model_str} not available")
                return False
            
            # Initialize analyzers
            self.primary_analyzer = ProbioticAnalyzer(primary_config, self.db_manager)
            self.secondary_analyzer = ProbioticAnalyzer(secondary_config, self.db_manager)
            
            logger.info("Setup validation passed!")
            return True
            
        except ValueError as e:
            logger.error(f"Invalid model name: {e}")
            logger.error(f"Available models: {[m.value for m in OllamaModel]}")
            return False
        except Exception as e:
            logger.error(f"Setup validation failed: {e}")
            return False
    
    def collect_papers_stage(self) -> bool:
        """Paper collection stage with resume capability"""
        if self.shutdown_requested:
            return False
            
        logger.info(f"Starting paper collection for condition: {self.condition}")
        self.state['current_stage'] = PipelineState.COLLECTING_PAPERS.value
        self._save_state()
        
        # Check if papers were already collected
        if self.state.get('papers_collected'):
            logger.info(f"Found {len(self.state['papers_collected'])} previously collected papers")
            return True
        
        try:
            collection_result = self.pubmed_collector.collect_probiotics_by_condition(
                condition=self.condition,
                max_results=self.max_papers
            )
            
            if collection_result.get('status') == 'success' and collection_result.get('paper_count', 0) > 0:
                # Fetch papers from database
                papers = self.db_manager.get_papers_by_condition(self.condition)
                self.state['papers_collected'] = papers[:self.max_papers]
                
                logger.info(f"Successfully collected {len(self.state['papers_collected'])} papers")
                self._save_state()
                return True
            else:
                logger.error("Paper collection failed or no papers found")
                return False
                
        except Exception as e:
            logger.error(f"Error during paper collection: {e}")
            self.state['errors'].append(f"Paper collection error: {str(e)}")
            self._save_state()
            return False
    
    def process_primary_model_stage(self) -> bool:
        """Primary model processing stage with resume capability"""
        if self.shutdown_requested:
            return False
            
        logger.info(f"Starting primary model ({self.primary_model_str}) processing...")
        self.state['current_stage'] = PipelineState.PROCESSING_PRIMARY.value
        self._save_state()
        
        # Check if primary processing was already completed
        if self.state.get('primary_results') and self.state['papers_processed_primary'] >= len(self.state['papers_collected']):
            logger.info("Primary model processing already completed")
            return True
        
        try:
            papers = self.state['papers_collected']
            start_idx = self.state['papers_processed_primary']
            
            logger.info(f"Resuming primary processing from paper {start_idx + 1}/{len(papers)}")
            
            # Process remaining papers
            for i in range(start_idx, len(papers)):
                if self.shutdown_requested:
                    logger.info("Shutdown requested during primary processing")
                    return False
                
                paper = papers[i]
                logger.info(f"Processing paper {i + 1}/{len(papers)} with primary model")
                
                try:
                    # Process single paper
                    result = self.primary_analyzer.process_papers([paper], save_to_db=True)
                    
                    # Update progress
                    self.state['papers_processed_primary'] = i + 1
                    if not self.state.get('primary_results'):
                        self.state['primary_results'] = result
                    else:
                        # Merge results
                        self.state['primary_results']['total_correlations'] += result.get('total_correlations', 0)
                        self.state['primary_results']['correlations'].extend(result.get('correlations', []))
                    
                    self._save_state()
                    
                except Exception as e:
                    logger.warning(f"Failed to process paper {i + 1} with primary model: {e}")
                    continue
            
            logger.info(f"Primary model processing completed. Processed {self.state['papers_processed_primary']} papers")
            return True
            
        except Exception as e:
            logger.error(f"Error during primary model processing: {e}")
            self.state['errors'].append(f"Primary processing error: {str(e)}")
            self._save_state()
            return False
    
    def process_secondary_model_stage(self) -> bool:
        """Secondary model processing stage with resume capability"""
        if self.shutdown_requested:
            return False
            
        logger.info(f"Starting secondary model ({self.secondary_model_str}) processing...")
        self.state['current_stage'] = PipelineState.PROCESSING_SECONDARY.value
        self._save_state()
        
        # Check if secondary processing was already completed
        if self.state.get('secondary_results') and self.state['papers_processed_secondary'] >= len(self.state['papers_collected']):
            logger.info("Secondary model processing already completed")
            return True
        
        try:
            papers = self.state['papers_collected']
            start_idx = self.state['papers_processed_secondary']
            
            logger.info(f"Resuming secondary processing from paper {start_idx + 1}/{len(papers)}")
            
            # Process remaining papers
            for i in range(start_idx, len(papers)):
                if self.shutdown_requested:
                    logger.info("Shutdown requested during secondary processing")
                    return False
                
                paper = papers[i]
                logger.info(f"Processing paper {i + 1}/{len(papers)} with secondary model")
                
                try:
                    # Process single paper
                    result = self.secondary_analyzer.process_papers([paper], save_to_db=True)
                    
                    # Update progress
                    self.state['papers_processed_secondary'] = i + 1
                    if not self.state.get('secondary_results'):
                        self.state['secondary_results'] = result
                    else:
                        # Merge results
                        self.state['secondary_results']['total_correlations'] += result.get('total_correlations', 0)
                        self.state['secondary_results']['correlations'].extend(result.get('correlations', []))
                    
                    self._save_state()
                    
                except Exception as e:
                    logger.warning(f"Failed to process paper {i + 1} with secondary model: {e}")
                    continue
            
            logger.info(f"Secondary model processing completed. Processed {self.state['papers_processed_secondary']} papers")
            return True
            
        except Exception as e:
            logger.error(f"Error during secondary model processing: {e}")
            self.state['errors'].append(f"Secondary processing error: {str(e)}")
            self._save_state()
            return False
    
    def finalize_results(self):
        """Finalize and save results"""
        logger.info("Finalizing results...")
        
        self.state['current_stage'] = PipelineState.COMPLETED.value
        self.state['completed'] = True
        self.state['end_time'] = time.time()
        
        # Create results summary
        results_summary = {
            'condition': self.condition,
            'max_papers': self.max_papers,
            'papers_collected': len(self.state['papers_collected']),
            'papers_processed_primary': self.state['papers_processed_primary'],
            'papers_processed_secondary': self.state['papers_processed_secondary'],
            'primary_model': self.primary_model_str,
            'secondary_model': self.secondary_model_str,
            'primary_correlations': len(self.state.get('primary_results', {}).get('correlations', [])),
            'secondary_correlations': len(self.state.get('secondary_results', {}).get('correlations', [])),
            'total_time': self.state.get('end_time', 0) - self.state.get('start_time', 0),
            'errors': self.state['errors']
        }
        
        # Save final results
        results_file = f"resumable_results_{self.condition.replace(' ', '_')}_{int(time.time())}.json"
        results_path = Path(results_file)
        
        final_results = {
            'summary': results_summary,
            'full_state': self.state
        }
        
        try:
            with open(results_path, 'w') as f:
                json.dump(final_results, f, indent=2, default=str)
            logger.info(f"Final results saved to {results_path}")
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
        
        self._save_state()
        
        # Print summary
        self.print_summary()
    
    def print_summary(self):
        """Print execution summary"""
        print("\n" + "="*70)
        print("RESUMABLE PIPELINE EXECUTION SUMMARY")
        print("="*70)
        
        print(f"\nCondition: {self.condition}")
        print(f"Max papers requested: {self.max_papers}")
        print(f"Papers collected: {len(self.state.get('papers_collected', []))}")
        print(f"Primary model: {self.primary_model_str}")
        print(f"Secondary model: {self.secondary_model_str}")
        
        print(f"\nProcessing Progress:")
        print(f"  Primary model papers processed: {self.state['papers_processed_primary']}")
        print(f"  Secondary model papers processed: {self.state['papers_processed_secondary']}")
        
        primary_correlations = len(self.state.get('primary_results', {}).get('correlations', []))
        secondary_correlations = len(self.state.get('secondary_results', {}).get('correlations', []))
        
        print(f"\nCorrelations Found:")
        print(f"  Primary model: {primary_correlations}")
        print(f"  Secondary model: {secondary_correlations}")
        
        if self.state.get('start_time') and self.state.get('end_time'):
            duration = self.state['end_time'] - self.state['start_time']
            print(f"\nTotal execution time: {duration:.2f} seconds")
        
        if self.state['errors']:
            print(f"\nErrors encountered: {len(self.state['errors'])}")
            for error in self.state['errors'][:3]:  # Show first 3 errors
                print(f"  - {error}")
            if len(self.state['errors']) > 3:
                print(f"  ... and {len(self.state['errors']) - 3} more")
        
        print(f"\nCurrent stage: {self.state['current_stage']}")
        print(f"Completed: {'Yes' if self.state['completed'] else 'No'}")
        print("="*70)
    
    def run(self) -> bool:
        """Run the complete pipeline with resume capability"""
        try:
            # Try to load previous state
            self._load_state()
            
            # Set start time if not already set
            if not self.state.get('start_time'):
                self.state['start_time'] = time.time()
            
            logger.info("Starting resumable pipeline execution...")
            
            # Validate setup
            if not self.validate_setup():
                logger.error("Setup validation failed")
                return False
            
            # Execute stages based on current state
            current_stage = self.state.get('current_stage', PipelineState.INITIALIZING.value)
            
            # Paper collection stage
            if current_stage in [PipelineState.INITIALIZING.value, PipelineState.COLLECTING_PAPERS.value]:
                if not self.collect_papers_stage():
                    logger.error("Paper collection stage failed")
                    return False
            
            # Primary model processing stage
            if current_stage in [PipelineState.INITIALIZING.value, PipelineState.COLLECTING_PAPERS.value, 
                               PipelineState.PROCESSING_PRIMARY.value]:
                if not self.process_primary_model_stage():
                    logger.error("Primary model processing stage failed")
                    return False
            
            # Secondary model processing stage
            if current_stage in [PipelineState.INITIALIZING.value, PipelineState.COLLECTING_PAPERS.value, 
                               PipelineState.PROCESSING_PRIMARY.value, PipelineState.PROCESSING_SECONDARY.value]:
                if not self.process_secondary_model_stage():
                    logger.error("Secondary model processing stage failed")
                    return False
            
            # Finalize results
            if not self.state.get('completed'):
                self.finalize_results()
            
            logger.info("Pipeline completed successfully!")
            
            # Clean up state file if completed
            if self.state.get('completed') and self.state_path.exists():
                try:
                    self.state_path.unlink()
                    logger.info("State file cleaned up after successful completion")
                except Exception as e:
                    logger.warning(f"Failed to clean up state file: {e}")
            
            return True
            
        except KeyboardInterrupt:
            logger.info("Pipeline interrupted by user")
            self._save_state()
            return False
        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}")
            self.state['current_stage'] = PipelineState.FAILED.value
            self.state['errors'].append(f"Pipeline execution error: {str(e)}")
            self._save_state()
            return False


def main():
    """Main function with argument parsing"""
    parser = argparse.ArgumentParser(description="Resumable Probiotic Research Pipeline")
    parser.add_argument("condition", help="Health condition to search for (e.g., 'IBS')")
    parser.add_argument("max_papers", type=int, help="Maximum number of papers to collect")
    parser.add_argument("--primary-model", default="deepseek-llm:7b-chat", 
                       help="Primary Ollama model (default: deepseek-llm:7b-chat)")
    parser.add_argument("--secondary-model", default="llama3.1:8b",
                       help="Secondary Ollama model (default: llama3.1:8b)")
    parser.add_argument("--ollama-url", default="http://localhost:11434/v1",
                       help="Ollama server URL (default: http://localhost:11434/v1)")
    parser.add_argument("--state-file", help="Custom state file path")
    parser.add_argument("--clean", action="store_true", 
                       help="Remove existing state file and start fresh")
    
    args = parser.parse_args()
    
    # Clean up previous state if requested
    if args.clean:
        if args.state_file:
            state_path = Path(args.state_file)
        else:
            safe_condition = "".join(c for c in args.condition if c.isalnum() or c in (' ', '-', '_')).rstrip()
            safe_condition = safe_condition.replace(' ', '_')
            state_file = f"pipeline_state_{safe_condition}_{args.max_papers}papers.json"
            state_path = Path(state_file)
        
        if state_path.exists():
            state_path.unlink()
            print(f"Removed existing state file: {state_path}")
    
    # Create and run pipeline
    pipeline = ResumablePipeline(
        condition=args.condition,
        max_papers=args.max_papers,
        primary_model=args.primary_model,
        secondary_model=args.secondary_model,
        ollama_url=args.ollama_url,
        state_file=args.state_file
    )
    
    success = pipeline.run()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()