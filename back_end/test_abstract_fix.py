#!/usr/bin/env python3
"""
Test script to verify the abstract parsing fix.
"""

import sys
from pathlib import Path
import xml.etree.ElementTree as ET

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.paper_collection.paper_parser import EnhancedPubmedParser

def test_abstract_parsing():
    """Test the abstract parsing fix on the problematic PMID."""
    
    # Load the XML file that contains the problematic abstract
    xml_file = Path("data/raw/metadata/pubmed_batch_1757519759.xml")
    
    if not xml_file.exists():
        print(f"XML file not found: {xml_file}")
        return
    
    # Parse the XML
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    # Find the specific article with PMID 38999862
    target_pmid = "38999862"
    article = None
    
    for pubmed_article in root.findall(".//PubmedArticle"):
        pmid_elem = pubmed_article.find(".//PMID")
        if pmid_elem is not None and pmid_elem.text == target_pmid:
            article = pubmed_article
            break
    
    if article is None:
        print(f"Article with PMID {target_pmid} not found")
        return
    
    # Extract abstract using the parser
    parser = EnhancedPubmedParser()
    article_meta = article.find(".//Article")
    
    if article_meta is not None:
        abstract = parser._extract_abstract(article_meta)
        
        print(f"PMID: {target_pmid}")
        print(f"Abstract length: {len(abstract)} characters")
        print(f"Abstract ends with: '{abstract[-50:]}'" if len(abstract) > 50 else f"Full abstract: '{abstract}'")
        print()
        print("Full abstract:")
        print(abstract)
        
        # Check if it contains the expected content
        if "Bifidobacterium" in abstract and "Lactobacillus" in abstract:
            print("\n✅ SUCCESS: Abstract contains expected content!")
        else:
            print("\n❌ ISSUE: Abstract may still be truncated")
    else:
        print("Article metadata not found")

if __name__ == "__main__":
    test_abstract_parsing()