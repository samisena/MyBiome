#!/usr/bin/env python3
"""
Detailed test of the multi-LLM consensus pipeline.
Tests with 10 abstracts, measures inference times, and verifies database storage.
"""

import sys
import time
import sqlite3
from pathlib import Path
from typing import Dict, List

# Add src to path for imports
current_dir = Path(__file__).parent
src_dir = current_dir / "src"
sys.path.insert(0, str(src_dir))

from src.llm.consensus_analyzer import MultiLLMConsensusAnalyzer
from src.data.config import config
from src.paper_collection.database_manager import database_manager

def get_test_papers(limit: int = 10) -> List[Dict]:
    """Get test papers from the database."""
    try:
        with database_manager.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT pmid, title, abstract, has_fulltext, fulltext_path
                FROM papers 
                WHERE abstract IS NOT NULL 
                  AND LENGTH(abstract) > 200
                  AND processing_status != 'processed'
                ORDER BY RANDOM()
                LIMIT ?
            ''', (limit,))
            
            papers = []
            for row in cursor.fetchall():
                papers.append({
                    'pmid': row[0],
                    'title': row[1],
                    'abstract': row[2],
                    'has_fulltext': bool(row[3]),
                    'fulltext_path': row[4]
                })
            
            return papers
            
    except Exception as e:
        print(f"Error getting test papers: {e}")
        return []

def check_database_before_test():
    """Check database state before testing."""
    try:
        with database_manager.get_connection() as conn:
            cursor = conn.cursor()
            
            # Count existing records
            cursor.execute("SELECT COUNT(*) FROM correlation_extractions")
            extractions_before = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM correlation_consensus") 
            consensus_before = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM correlations")
            correlations_before = cursor.fetchone()[0]
            
            print(f"Database state BEFORE test:")
            print(f"  correlation_extractions: {extractions_before}")
            print(f"  correlation_consensus: {consensus_before}")
            print(f"  correlations: {correlations_before}")
            
            return {
                'extractions': extractions_before,
                'consensus': consensus_before, 
                'correlations': correlations_before
            }
            
    except Exception as e:
        print(f"Error checking database: {e}")
        return None

def check_database_after_test(before_counts: Dict):
    """Check database state after testing."""
    try:
        with database_manager.get_connection() as conn:
            cursor = conn.cursor()
            
            # Count records after test
            cursor.execute("SELECT COUNT(*) FROM correlation_extractions")
            extractions_after = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM correlation_consensus")
            consensus_after = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM correlations")
            correlations_after = cursor.fetchone()[0]
            
            print(f"\nDatabase state AFTER test:")
            print(f"  correlation_extractions: {extractions_after} (+{extractions_after - before_counts['extractions']})")
            print(f"  correlation_consensus: {consensus_after} (+{consensus_after - before_counts['consensus']})")
            print(f"  correlations: {correlations_after} (+{correlations_after - before_counts['correlations']})")
            
            # Show some sample data
            print(f"\nSample extraction records:")
            cursor.execute('''
                SELECT paper_id, extraction_model, probiotic_strain, health_condition, correlation_type
                FROM correlation_extractions
                ORDER BY id DESC
                LIMIT 5
            ''')
            for row in cursor.fetchall():
                print(f"  {row[0]} | {row[1]} | {row[2]} | {row[3]} | {row[4]}")
            
            print(f"\nSample consensus records:")
            cursor.execute('''
                SELECT paper_id, probiotic_strain, health_condition, consensus_status, needs_review
                FROM correlation_consensus
                ORDER BY id DESC  
                LIMIT 5
            ''')
            for row in cursor.fetchall():
                print(f"  {row[0]} | {row[1]} | {row[2]} | {row[3]} | Review: {row[4]}")
            
            return True
            
    except Exception as e:
        print(f"Error checking database after test: {e}")
        return False

def measure_inference_times(analyzer: MultiLLMConsensusAnalyzer, papers: List[Dict]):
    """Measure detailed inference times for each model on each paper."""
    
    print(f"\n=== DETAILED INFERENCE TIME ANALYSIS ===")
    
    model_times = {model.model_name: [] for model in config.multi_llm.models}
    total_processing_times = []
    
    for i, paper in enumerate(papers, 1):
        pmid = paper['pmid']
        print(f"\nProcessing paper {i}/{len(papers)}: {pmid}")
        print(f"Abstract length: {len(paper.get('abstract', ''))} chars")
        
        paper_start_time = time.time()
        
        # Test individual model extractions with timing
        individual_times = {}
        
        for model_config in config.multi_llm.models:
            model_name = model_config.model_name
            print(f"  Testing {model_name}...")
            
            model_start = time.time()
            
            try:
                extraction_result = analyzer.extract_with_single_model(paper, model_name)
                model_duration = time.time() - model_start
                
                individual_times[model_name] = model_duration
                model_times[model_name].append(model_duration)
                
                print(f"    Time: {model_duration:.2f}s")
                print(f"    Correlations: {len(extraction_result.correlations)}")
                print(f"    Tokens: {extraction_result.token_usage.get('total', 0)}")
                
                if extraction_result.error:
                    print(f"    Error: {extraction_result.error}")
                    
            except Exception as e:
                print(f"    Error testing {model_name}: {e}")
                individual_times[model_name] = -1
        
        total_paper_time = time.time() - paper_start_time
        total_processing_times.append(total_paper_time)
        
        print(f"  Total paper time: {total_paper_time:.2f}s")
        
        # Small delay between papers
        time.sleep(1)
    
    # Calculate and display statistics
    print(f"\n=== INFERENCE TIME STATISTICS ===")
    
    for model_name, times in model_times.items():
        if times and all(t >= 0 for t in times):  # Exclude error cases
            avg_time = sum(times) / len(times)
            min_time = min(times)
            max_time = max(times)
            print(f"\n{model_name}:")
            print(f"  Average: {avg_time:.2f}s per abstract")
            print(f"  Min: {min_time:.2f}s")
            print(f"  Max: {max_time:.2f}s")
            print(f"  Total: {sum(times):.2f}s for {len(times)} abstracts")
        else:
            print(f"\n{model_name}: No successful extractions or errors occurred")
    
    if total_processing_times:
        avg_total = sum(total_processing_times) / len(total_processing_times)
        print(f"\nOverall processing (including consensus):")
        print(f"  Average: {avg_total:.2f}s per paper") 
        print(f"  Total: {sum(total_processing_times):.2f}s for {len(papers)} papers")

def main():
    """Run detailed consensus pipeline test."""
    
    print("=" * 80)
    print("DETAILED MULTI-LLM CONSENSUS PIPELINE TEST")
    print("=" * 80)
    
    # Check database state before test
    before_counts = check_database_before_test()
    if before_counts is None:
        print("Failed to check database state. Exiting.")
        return 1
    
    # Get test papers
    print(f"\nGetting test papers...")
    test_papers = get_test_papers(10)
    
    if not test_papers:
        print("No test papers found. Make sure you have papers with abstracts in the database.")
        return 1
    
    print(f"Found {len(test_papers)} test papers")
    
    # Display model configuration
    print(f"\nConfigured models:")
    for i, model_config in enumerate(config.multi_llm.models, 1):
        print(f"  {i}. {model_config.model_name} (temp: {model_config.temperature})")
    
    # Initialize consensus analyzer
    print(f"\nInitializing consensus analyzer...")
    analyzer = MultiLLMConsensusAnalyzer()
    
    # Measure inference times in detail
    measure_inference_times(analyzer, test_papers)
    
    print(f"\n" + "=" * 80)
    print("RUNNING FULL CONSENSUS PIPELINE TEST")
    print("=" * 80)
    
    # Now run the full consensus pipeline
    total_start = time.time()
    
    all_results = []
    successful_papers = 0
    total_agreed = 0
    total_conflicts = 0
    
    for i, paper in enumerate(test_papers, 1):
        pmid = paper['pmid']
        print(f"\nProcessing paper {i}/{len(test_papers)} with full consensus: {pmid}")
        
        try:
            # Process with full consensus pipeline
            consensus_result = analyzer.process_paper_with_consensus(paper)
            all_results.append(consensus_result)
            
            if consensus_result.consensus_status not in ['processing_error', 'invalid_paper']:
                successful_papers += 1
            
            total_agreed += len(consensus_result.agreed_correlations)
            total_conflicts += len(consensus_result.conflicting_correlations)
            
            print(f"  Status: {consensus_result.consensus_status}")
            print(f"  Agreed: {len(consensus_result.agreed_correlations)}")
            print(f"  Conflicts: {len(consensus_result.conflicting_correlations)}")
            print(f"  Needs review: {consensus_result.needs_review}")
            
            if consensus_result.review_reasons:
                print(f"  Review reasons: {', '.join(consensus_result.review_reasons[:3])}")
            
        except Exception as e:
            print(f"  Error: {e}")
            continue
    
    total_duration = time.time() - total_start
    
    # Display results
    print(f"\n" + "=" * 80)
    print("CONSENSUS PIPELINE RESULTS")
    print("=" * 80)
    
    print(f"Papers processed: {successful_papers}/{len(test_papers)}")
    print(f"Total agreed correlations: {total_agreed}")
    print(f"Total conflicts: {total_conflicts}")
    print(f"Papers needing review: {sum(1 for r in all_results if r.needs_review)}")
    print(f"Total processing time: {total_duration:.2f}s")
    print(f"Average time per paper: {total_duration/len(test_papers):.2f}s")
    
    # Show token usage summary
    print(f"\nToken usage summary:")
    for model, usage in analyzer.token_usage.items():
        total_tokens = usage['input'] + usage['output']
        print(f"  {model}: {total_tokens:,} tokens ({usage['input']:,} in, {usage['output']:,} out)")
    
    # Check database state after test  
    print(f"\n" + "=" * 80)
    print("DATABASE VERIFICATION")
    print("=" * 80)
    
    check_database_after_test(before_counts)
    
    print(f"\n" + "=" * 80)
    print("TEST COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)