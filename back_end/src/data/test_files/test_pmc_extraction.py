"""
Test PMC ID extraction from existing XML files.
"""

import sys
from pathlib import Path

# Add the back_end directory to Python path  
back_end_root = Path(__file__).parent.parent.parent  # Go up 3 levels: file -> data -> src -> back_end
sys.path.append(str(back_end_root))
project_root = back_end_root.parent  # My_Biome directory

from src.data.paper_parser import PubmedParser

def main():
    """Test PMC extraction from actual XML files."""
    print("Testing PMC ID extraction from existing XML files...")
    
    # Get an existing XML file
    metadata_dir = project_root / "data" / "raw" / "metadata"
    xml_files = list(metadata_dir.glob("pubmed_batch_*.xml"))
    
    if not xml_files:
        print("No XML files found!")
        return
    
    # Test with a recent file
    test_file = xml_files[-1]  # Use the most recent one
    print(f"Testing with: {test_file.name}")
    
    # Parse the file
    parser = PubmedParser()
    papers = parser.parse_metadata_file(str(test_file))
    
    if not papers:
        print("Failed to parse papers!")
        return
    
    print(f"Parsed {len(papers)} papers")
    
    # Check PMC IDs
    papers_with_pmc = [p for p in papers if p.get('pmc_id')]
    papers_with_doi = [p for p in papers if p.get('doi')]
    
    print(f"Papers with PMC IDs: {len(papers_with_pmc)} ({len(papers_with_pmc)/len(papers)*100:.1f}%)")
    print(f"Papers with DOIs: {len(papers_with_doi)} ({len(papers_with_doi)/len(papers)*100:.1f}%)")
    
    # Show samples
    if papers_with_pmc:
        print("\nSample papers with PMC IDs:")
        for i, paper in enumerate(papers_with_pmc[:3]):
            print(f"{i+1}. PMID: {paper['pmid']}")
            print(f"   PMC ID: {paper['pmc_id']}")
            print(f"   Title: {paper['title'][:60]}...")
            print()
    
    if papers_with_doi and not papers_with_pmc:
        print("\nSample papers with DOIs (no PMC):")
        for i, paper in enumerate(papers_with_doi[:3]):
            print(f"{i+1}. PMID: {paper['pmid']}")
            print(f"   DOI: {paper['doi']}")
            print(f"   Title: {paper['title'][:60]}...")
            print()
    
    # Test a few more files to get a better sample
    print("\nTesting multiple files for better statistics...")
    
    total_papers = 0
    total_pmc = 0
    total_doi = 0
    
    for xml_file in xml_files[-3:]:  # Test last 3 files
        papers = parser.parse_metadata_file(str(xml_file))
        if papers:
            total_papers += len(papers)
            total_pmc += len([p for p in papers if p.get('pmc_id')])
            total_doi += len([p for p in papers if p.get('doi')])
    
    print(f"\nOverall statistics from {min(3, len(xml_files))} files:")
    print(f"Total papers: {total_papers}")
    print(f"Papers with PMC IDs: {total_pmc} ({total_pmc/total_papers*100:.1f}%)")
    print(f"Papers with DOIs: {total_doi} ({total_doi/total_papers*100:.1f}%)")
    print(f"Papers suitable for full-text retrieval: {total_pmc + total_doi} ({(total_pmc + total_doi)/total_papers*100:.1f}%)")

if __name__ == "__main__":
    main()