# constipation_probiotics_study.py
import os
import sys
import logging
import json
import time
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from dotenv import load_dotenv
import pandas as pd

# Load environment variables
load_dotenv()

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from src.data.data_collector import PubMedCollector
from src.data.probiotic_analyzer import CorrelationExtractor, LLMConfig
from src.data.database_manager import DatabaseManager


class ConstipationProbioticsStudy:
    """Comprehensive study of probiotics for constipation."""
    
    def __init__(self):
        self.setup_logging()
        self.db_manager = DatabaseManager()
        self.collector = PubMedCollector()
        self.setup_extractor()
        
    def setup_logging(self):
        """Configure logging with both file and console output."""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"constipation_study_{timestamp}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Logging to {log_file}")
        
    def setup_extractor(self):
        """Configure the Groq-based correlation extractor."""
        api_key = os.getenv('GROQ_API_KEY')
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in .env file")
        
        config = LLMConfig(
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1",
            model_name="llama-3.3-70b-versatile",
            temperature=0.1,  # Low for consistent JSON
            max_tokens=4000,
            cost_per_1k_input_tokens=0,
            cost_per_1k_output_tokens=0
        )
        
        self.extractor = CorrelationExtractor(config, self.db_manager)
        
    def collect_constipation_papers(self, max_papers=100):
        """Collect papers about probiotics and constipation."""
        self.logger.info("=" * 60)
        self.logger.info("PHASE 1: COLLECTING PAPERS FROM PUBMED")
        self.logger.info("=" * 60)
        
        # Comprehensive search query for constipation and probiotics
        queries = [
            # Broad search
            '"probiotics"[Title/Abstract] AND ("constipation"[Title/Abstract] OR "bowel movement"[Title/Abstract] OR "defecation"[Title/Abstract])',
            
            # Specific strains often studied for constipation
            '"Bifidobacterium"[Title/Abstract] AND "constipation"[Title/Abstract]',
            '"Lactobacillus"[Title/Abstract] AND "constipation"[Title/Abstract]',
            
            # Clinical trials specifically
            '"probiotics"[Title/Abstract] AND "constipation"[Title/Abstract] AND ("randomized controlled trial"[Publication Type] OR "clinical trial"[Publication Type])',
        ]
        
        all_paper_ids = set()
        papers_per_query = max_papers // len(queries) + 10  # Slight overfetch to ensure we get enough unique papers
        
        for i, query in enumerate(queries, 1):
            self.logger.info(f"\n[Query {i}/{len(queries)}]")
            self.logger.info(f"Searching: {query[:100]}...")
            
            paper_ids = self.collector.search_papers(
                query=query,
                max_results=papers_per_query,
                min_year=2010  # Focus on recent research
            )
            
            if paper_ids:
                all_paper_ids.update(paper_ids)
                self.logger.info(f"Found {len(paper_ids)} papers (Total unique: {len(all_paper_ids)})")
            
            time.sleep(1)  # Be respectful to PubMed API
        
        # Convert to list and limit to max_papers
        paper_ids_list = list(all_paper_ids)[:max_papers]
        
        self.logger.info(f"\nTotal unique papers found: {len(paper_ids_list)}")
        
        # Fetch and store paper details in batches
        batch_size = 20
        for i in range(0, len(paper_ids_list), batch_size):
            batch = paper_ids_list[i:i+batch_size]
            self.logger.info(f"Fetching batch {i//batch_size + 1}/{(len(paper_ids_list)-1)//batch_size + 1}")
            
            metadata_file = self.collector.fetch_paper_details(batch)
            if metadata_file:
                papers = self.collector.parser.parse_metadata_file(metadata_file)
                self.logger.info(f"Parsed {len(papers)} papers from batch")
            
            time.sleep(1)
        
        return len(paper_ids_list)
    
    def extract_correlations(self, batch_size=10):
        """Extract correlations from collected papers."""
        self.logger.info("\n" + "=" * 60)
        self.logger.info("PHASE 2: EXTRACTING CORRELATIONS WITH LLM")
        self.logger.info("=" * 60)
        
        # Get unprocessed papers
        unprocessed = self.db_manager.get_unprocessed_papers(
            extraction_model="llama-3.3-70b-versatile",
            limit=None  # Get all unprocessed
        )
        
        self.logger.info(f"Found {len(unprocessed)} unprocessed papers")
        
        if not unprocessed:
            self.logger.warning("No unprocessed papers found. Papers may have been processed already.")
            return 0
        
        # Process in batches to avoid overwhelming the API and track progress
        total_correlations = 0
        
        for i in range(0, len(unprocessed), batch_size):
            batch = unprocessed[i:i+batch_size]
            batch_num = i // batch_size + 1
            total_batches = (len(unprocessed) - 1) // batch_size + 1
            
            self.logger.info(f"\nProcessing batch {batch_num}/{total_batches} ({len(batch)} papers)")
            
            results = self.extractor.process_papers(batch, save_to_db=True)
            total_correlations += results['total_correlations']
            
            self.logger.info(f"Batch {batch_num} complete: {results['total_correlations']} correlations found")
            
            # Add delay between batches to respect rate limits
            if i + batch_size < len(unprocessed):
                self.logger.info("Waiting 5 seconds before next batch...")
                time.sleep(5)
        
        return total_correlations
    
    def analyze_results(self):
        """Analyze extracted correlations to identify best probiotics for constipation."""
        self.logger.info("\n" + "=" * 60)
        self.logger.info("PHASE 3: ANALYZING RESULTS")
        self.logger.info("=" * 60)
        
        # Get all correlations related to constipation
        constipation_keywords = [
            'constipation', 'bowel movement', 'defecation', 'stool frequency',
            'intestinal transit', 'colonic transit', 'functional constipation',
            'chronic constipation', 'bowel function'
        ]
        
        # Query database for relevant correlations
        query = """
        SELECT 
            c.*,
            p.title,
            p.publication_date
        FROM correlations c
        JOIN papers p ON c.paper_id = p.pmid
        WHERE 1=1
        """
        
        # Build WHERE clause for constipation-related conditions
        conditions = " OR ".join([f"LOWER(c.health_condition) LIKE '%{kw}%'" for kw in constipation_keywords])
        query += f" AND ({conditions})"
        
        import sqlite3
        conn = sqlite3.connect(self.db_manager.db_path)
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        if df.empty:
            self.logger.warning("No constipation-related correlations found")
            return None
        
        self.logger.info(f"Found {len(df)} constipation-related correlations")
        
        # Analysis 1: Probiotic strain frequency and effectiveness
        strain_analysis = df.groupby('probiotic_strain').agg({
            'correlation_strength': ['mean', 'std', 'count'],
            'correlation_type': lambda x: (x == 'positive').sum() / len(x) * 100,  # % positive
            'sample_size': 'mean',
            'confidence_score': 'mean'
        }).round(3)
        
        strain_analysis.columns = ['avg_strength', 'std_strength', 'num_studies', 
                                   'percent_positive', 'avg_sample_size', 'avg_confidence']
        strain_analysis = strain_analysis.sort_values('num_studies', ascending=False)
        
        # Analysis 2: Top performers (high strength + positive correlations)
        positive_df = df[df['correlation_type'] == 'positive']
        if not positive_df.empty:
            top_performers = positive_df.groupby('probiotic_strain').agg({
                'correlation_strength': 'mean',
                'confidence_score': 'mean',
                'paper_id': 'count'
            }).round(3)
            top_performers.columns = ['avg_strength', 'avg_confidence', 'num_positive_studies']
            top_performers = top_performers.sort_values('avg_strength', ascending=False).head(10)
        else:
            top_performers = None
        
        # Analysis 3: Study type breakdown
        study_type_analysis = df.groupby('study_type').agg({
            'correlation_strength': 'mean',
            'paper_id': 'count'
        }).round(3)
        study_type_analysis.columns = ['avg_strength', 'num_studies']
        
        # Save results
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save to CSV files (avoiding Excel dependency)
        csv_base = results_dir / f"constipation_analysis_{timestamp}"
        strain_analysis.to_csv(f"{csv_base}_strain_analysis.csv")
        if top_performers is not None:
            top_performers.to_csv(f"{csv_base}_top_performers.csv")
        study_type_analysis.to_csv(f"{csv_base}_study_types.csv")
        df.to_csv(f"{csv_base}_raw_data.csv", index=False)
        
        self.logger.info(f"Results saved to {csv_base}_*.csv files")
        
        # Print summary
        self.logger.info("\n" + "=" * 60)
        self.logger.info("TOP PROBIOTIC STRAINS FOR CONSTIPATION")
        self.logger.info("=" * 60)
        print("\n", strain_analysis.head(10))
        
        if top_performers is not None:
            self.logger.info("\n" + "=" * 60)
            self.logger.info("HIGHEST EFFECTIVENESS (Positive Correlations Only)")
            self.logger.info("=" * 60)
            print("\n", top_performers)
        
        return {
            'strain_analysis': strain_analysis,
            'top_performers': top_performers,
            'study_types': study_type_analysis,
            'raw_data': df
        }
    
    def run_full_pipeline(self, collect_new=True, max_papers=100):
        """Run the complete pipeline."""
        start_time = time.time()
        
        self.logger.info("=" * 60)
        self.logger.info("CONSTIPATION-PROBIOTICS RESEARCH PIPELINE")
        self.logger.info(f"Target: {max_papers} papers")
        self.logger.info("=" * 60)
        
        # Phase 1: Collection (optional)
        if collect_new:
            papers_collected = self.collect_constipation_papers(max_papers)
            self.logger.info(f"\n[SUCCESS] Collected {papers_collected} papers")
        else:
            self.logger.info("\nSkipping collection phase (using existing papers)")
        
        # Phase 2: Extraction
        correlations_extracted = self.extract_correlations(batch_size=10)
        self.logger.info(f"\n[SUCCESS] Extracted {correlations_extracted} new correlations")
        
        # Phase 3: Analysis
        analysis_results = self.analyze_results()
        
        # Final summary
        elapsed_time = time.time() - start_time
        stats = self.db_manager.get_database_stats()
        
        self.logger.info("\n" + "=" * 60)
        self.logger.info("PIPELINE COMPLETE")
        self.logger.info("=" * 60)
        self.logger.info(f"Total papers in database: {stats['total_papers']}")
        self.logger.info(f"Total correlations: {stats['total_correlations']}")
        self.logger.info(f"Execution time: {elapsed_time/60:.1f} minutes")
        
        return analysis_results


def main():
    """Main execution function."""
    study = ConstipationProbioticsStudy()
    
    # Run the full pipeline
    # Set collect_new=False if you want to just process existing papers
    results = study.run_full_pipeline(
        collect_new=True,  # Set to False to skip collection and just process existing papers
        max_papers=10
    )
    
    if results and 'strain_analysis' in results:
        print("\n" + "=" * 60)
        print("STUDY COMPLETE - Check 'results' folder for detailed analysis")
        print("=" * 60)
        
        # Quick summary of top 5 strains
        top_5 = results['strain_analysis'].head(5)
        print("\nTop 5 Most Studied Probiotic Strains for Constipation:")
        for strain in top_5.index:
            data = top_5.loc[strain]
            print(f"\n• {strain}")
            print(f"  Studies: {int(data['num_studies'])}")
            print(f"  Avg Strength: {data['avg_strength']:.2f}")
            print(f"  Positive Rate: {data['percent_positive']:.1f}%")


if __name__ == "__main__":
    # Ensure you have GROQ_API_KEY in your .env file
    # Run with: python constipation_probiotics_study.py
    main()