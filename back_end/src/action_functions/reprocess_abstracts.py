#!/usr/bin/env python3
"""
Script to re-process existing XML files with the fixed abstract parser
to get complete abstracts instead of truncated ones.
"""

import sys
from pathlib import Path
from typing import Dict, List

# Add the src directory to Python path
src_dir = Path(__file__).parent.parent
sys.path.insert(0, str(src_dir))

try:
    from src.paper_collection.database_manager import database_manager
    from src.paper_collection.paper_parser import EnhancedPubmedParser
    from src.data.config import config, setup_logging
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running this from the back_end directory")
    sys.exit(1)

logger = setup_logging(__name__, 'reprocess_abstracts.log')

class AbstractReprocessor:
    """Re-process existing XML files to get complete abstracts."""
    
    def __init__(self):
        self.parser = EnhancedPubmedParser()
        self.metadata_dir = config.paths.metadata_dir
        self.updated_count = 0
        self.skipped_count = 0
        self.error_count = 0
    
    def find_xml_files(self) -> List[Path]:
        """Find all XML metadata files."""
        xml_files = list(self.metadata_dir.glob("pubmed_batch_*.xml"))
        print(f"Found {len(xml_files)} XML files to reprocess")
        return xml_files
    
    def reprocess_xml_file(self, xml_file: Path) -> Dict[str, int]:
        """Reprocess a single XML file and update abstracts."""
        print(f"\nProcessing: {xml_file.name}")
        
        # Parse the XML file with fixed parser
        papers = self.parser.parse_metadata_file(str(xml_file))
        
        if not papers:
            print(f"  No papers found in {xml_file.name}")
            return {'updated': 0, 'skipped': 0, 'errors': 0}
        
        print(f"  Found {len(papers)} papers")
        
        updated = 0
        skipped = 0
        errors = 0
        
        # Update abstracts in database
        for paper in papers:
            try:
                pmid = paper['pmid']
                new_abstract = paper['abstract']
                
                if not new_abstract:
                    skipped += 1
                    continue
                
                # Get current abstract from database
                with database_manager.get_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute('SELECT abstract FROM papers WHERE pmid = ?', (pmid,))
                    result = cursor.fetchone()
                    
                    if result:
                        current_abstract = result[0] or ''
                        
                        # Only update if new abstract is longer (indicating it's more complete)
                        if len(new_abstract) > len(current_abstract):
                            cursor.execute('''
                                UPDATE papers 
                                SET abstract = ?, updated_at = CURRENT_TIMESTAMP
                                WHERE pmid = ?
                            ''', (new_abstract, pmid))
                            
                            conn.commit()
                            updated += 1
                            
                            print(f"    Updated PMID {pmid}: {len(current_abstract)} -> {len(new_abstract)} chars")
                        else:
                            skipped += 1
                    else:
                        # Paper not in database, insert it
                        if database_manager.insert_paper(paper):
                            updated += 1
                            print(f"    Inserted new paper PMID {pmid}")
                        else:
                            errors += 1
                            
            except Exception as e:
                logger.error(f"Error processing paper {paper.get('pmid', 'unknown')}: {e}")
                errors += 1
        
        print(f"  Results: {updated} updated, {skipped} skipped, {errors} errors")
        return {'updated': updated, 'skipped': skipped, 'errors': errors}
    
    def reprocess_all_files(self) -> None:
        """Reprocess all XML files."""
        print("MyBiome Abstract Reprocessor")
        print("============================")
        print("Re-processing XML files with fixed abstract parser")
        print()
        
        xml_files = self.find_xml_files()
        
        if not xml_files:
            print("No XML files found to process!")
            return
        
        total_stats = {'updated': 0, 'skipped': 0, 'errors': 0}
        
        for i, xml_file in enumerate(xml_files, 1):
            print(f"\nFile {i}/{len(xml_files)}")
            stats = self.reprocess_xml_file(xml_file)
            
            # Accumulate stats
            for key in total_stats:
                total_stats[key] += stats[key]
        
        self.print_final_results(total_stats)
    
    def print_final_results(self, stats: Dict[str, int]) -> None:
        """Print final processing results."""
        print("\n" + "=" * 50)
        print("ABSTRACT REPROCESSING RESULTS")
        print("=" * 50)
        print(f"Abstracts updated: {stats['updated']}")
        print(f"Abstracts skipped: {stats['skipped']}")
        print(f"Errors encountered: {stats['errors']}")
        print()
        
        if stats['updated'] > 0:
            print("SUCCESS: Abstract reprocessing completed successfully!")
            print("   Truncated abstracts have been replaced with complete versions")
            print("   The correlation review tool will now show complete abstracts")
        else:
            print("INFO: No abstracts needed updating")
        
        print()
        print("Next steps:")
        print("1. Run the correlation review tool to see complete abstracts")
        print("2. Consider re-running correlation extraction on papers with updated abstracts")
    
    def check_sample_improvements(self) -> None:
        """Check a few sample papers to show improvements."""
        print("\nSample Abstract Improvements:")
        print("-" * 40)
        
        # Check the problematic PMID we tested
        test_pmids = ['38999862', '36362056', '38331656']
        
        with database_manager.get_connection() as conn:
            cursor = conn.cursor()
            
            for pmid in test_pmids:
                cursor.execute('SELECT LENGTH(abstract), abstract FROM papers WHERE pmid = ? LIMIT 1', (pmid,))
                result = cursor.fetchone()
                
                if result:
                    length, abstract = result
                    # Clean abstract text to avoid encoding issues
                    clean_abstract = abstract.encode('ascii', 'ignore').decode('ascii') if abstract else ''
                    ending = clean_abstract[-50:] if len(clean_abstract) > 50 else clean_abstract
                    print(f"PMID {pmid}: {length} chars, ends with: '...{ending}'")
                else:
                    print(f"PMID {pmid}: Not found in database")

def main():
    """Main function to run the reprocessor."""
    reprocessor = AbstractReprocessor()
    
    # First show current state
    print("Checking current abstract state...")
    reprocessor.check_sample_improvements()
    
    # Ask for confirmation
    print("\nThis will re-process all XML files and update abstracts in the database.")
    confirm = input("Continue? (y/n): ").lower().strip()
    
    if confirm not in ['y', 'yes']:
        print("Cancelled.")
        return
    
    # Run reprocessing
    reprocessor.reprocess_all_files()
    
    # Show improvements
    print("\nAfter reprocessing:")
    reprocessor.check_sample_improvements()

if __name__ == "__main__":
    main()