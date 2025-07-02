""" This script orchestrates the collection of about 200-300 papers"""

import time
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple

from src.data.data_collector import PubMedCollector
from src.data.database_manager import DatabaseManager

project_root = Path(__file__).parent.parent.parent


DEFAULT_STRAINS = [
    "Lactobacillus rhamnosus GG",  # Most studied probiotic
    "Saccharomyces boulardii",      # Well-studied yeast probiotic
    "Lactobacillus acidophilus",    # Common in supplements
    "Bifidobacterium longum",       # Important for gut health
    "Lactobacillus plantarum"       # Versatile strain
]

DEFAULT_CONDITIONS = [
    "irritable bowel syndrome",      
    "constipation",                  
    "inflammatory bowel disease",     
    "acne",             
    "immune function",               
    "bloating",                    
    "acid reflux"                        
]


class CollectionRunner:
    """_summary_
    """
    
    def __init__(self):
        """_summary_
        """
        self.collector = PubMedCollector()
        self.db_manager = DatabaseManager()

        
    def collect_papers(self, strains: List[str] = DEFAULT_STRAINS, 
                       conditions :List[str] = DEFAULT_CONDITIONS,
                       papers_per_query: int =10,
                       delay_between_queries: float = 1.0
                       ) -> int:
        """
        
        """
        
        strains = strains
        conditions = conditions 
        
        total_new_papers = 0
        total_queries = len(strains) * len(conditions)
        query_count = 0 
        
        print(f'''Starting collection: {len(strains)} strains and
              {len(conditions)}conditions = {total_queries} queries''')
        
        #* Every strain with every condition
        for strain in strains:
            for condition in conditions:
                query_count += 1
                print(f"[{query_count}/{total_queries}] {strain} + {condition}")
                
                try:
                    #* Calls PubMedCollector() method
                    result = self.collector.collect_by_strain_and_condition(
                        strain, condition, papers_per_query)
                    
                    if result['status'] == 'success':
                        papers_found = result['paper_count']
                        print(f"Found {papers_found} papers")
                        total_new_papers += papers_found
                    else:
                        print(f"No results found")
                    
                except Exception as e:
                    print(f"Error: {e}")
                    
                time.sleep(delay_between_queries)
                
        print(f"\n Collection run complete: {total_new_papers} new papers added to the database")
        return total_new_papers
    
    
    def get_collection_summary(self) -> Dict:
        """_summary_
        """
        return self.db_manager.get_database_stats()
        
        
def main():
        runner = CollectionRunner()
        
        print("\n Starting paper collection...")
        new_papers = runner.collect_papers(
            strains=["Lactobacillus rhamnosus GG", "Bifidobacterium longum"], 
            conditions=["irritable bowel syndrome", "depression"],
            papers_per_query=15,
            delay_between_queries=1.5
            )
    
        print("\n📊 Collection Summary:")
        summary = runner.get_collection_summary()
        for key, value in summary.items():
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
        
