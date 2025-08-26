import os 
import json 
import xml.etree.ElementTree as ET  # Module for processing XML data
from pathlib import Path
from dataclasses import dataclass, asdict 
from typing import List, Dict, Optional, Any
from src.data.database_manager import DatabaseManager, Paper  # Import our new database manager

project_root = Path(__file__).parent.parent.parent

class PubmedParser:
    """ This class parses the XML files from the pubmed API into a tree structure, 
    then searches for relevant information within the structure using methods 
    from xml.etree and stores it within a SQLite database.
    """
    
    def __init__(self):
        self.metadata_dir = project_root / "data" / "raw" / "metadata"
        self.processed_dir = project_root / "data" / "processed"
        self.ns = {"": "http://www.ncbi.nlm.nih.gov/pubmed"}
        self.db_manager = DatabaseManager()
        
    def parse_metadata_file(self, file_path: str) -> List['Paper']:
        """Parses a single XML metadata file and returns a list of Paper objects."""
        
        print(f"Starting to parse: {file_path}")  # Debug line
        
        try:
            tree = ET.parse(file_path)
            root = tree.getroot()
            papers = []
            papers_inserted = 0
            papers_skipped = 0
            
            # Count articles for debugging
            articles = root.findall(".//PubmedArticle")
            print(f"Found {len(articles)} articles in XML")  # Debug line
            
            for i, article in enumerate(articles):
                print(f"\nProcessing article {i+1}...")  # Debug line
                
                try:
                    # Extract PMID
                    print("  - Extracting PMID...")
                    pmid_elem = article.find(".//PMID")
                    if pmid_elem is None:   
                        print("Warning: Article missing PMID, skipping...")
                        continue
                    pmid = pmid_elem.text
                    
                    #* Extract article metadata
                    article_meta = article.find(".//Article")
                    if article_meta is None:
                        print(f"Warning: Article {pmid} missing Article element, skipping...")
                        continue
                    
                    #* Extract title
                    title_elem = article_meta.find(".//ArticleTitle")
                    title = title_elem.text if title_elem is not None else "No title available"
                    
                    #* Extract abstract
                    abstract_elem = article_meta.find(".//Abstract/AbstractText")
                    abstract = abstract_elem.text if abstract_elem is not None else ""
                    
                    #* Extract journal
                    journal_elem = article_meta.find(".//Journal/Title")
                    journal = journal_elem.text if journal_elem is not None else "Unknown journal"
                    
                    #* Extract publication date
                    pub_date = article_meta.find(".//PubDate")
                    year = month = day = ""
                    if pub_date is not None:
                        year_elem = pub_date.find(".//Year")
                        month_elem = pub_date.find(".//Month")
                        day_elem = pub_date.find(".//Day")
                        year = year_elem.text if year_elem is not None else ""
                        month = month_elem.text if month_elem is not None else ""
                        day = day_elem.text if day_elem is not None else ""
                    
                    pub_date_str = f"{year}-{month}-{day}" if day else f"{year}-{month}" if month else year
                    
                    #* Extract DOI
                    doi_elem = article.find(".//ArticleId[@IdType='doi']")
                    doi = doi_elem.text if doi_elem is not None else None
                    
                    #* Extract keywords        
                    keywords = []
                    keyword_list = article.find(".//KeywordList")
                    if keyword_list is not None:
                        for keyword_elem in keyword_list.findall(".//Keyword"):
                            if keyword_elem.text:
                                keywords.append(keyword_elem.text)
                    
                    #* Create Paper object
                    paper = {
                        'pmid':pmid,
                        'title':title,
                        'abstract':abstract,
                        'journal':journal,
                        'publication_date':pub_date_str,
                        'doi':doi,
                        'keywords':keywords if keywords else None
                     }
                    
                    #* Insert into database:
                    was_new = self.db_manager.insert_paper(paper)
                    if was_new:
                        papers_inserted += 1
                    else:
                        papers_skipped += 1
                    
                    papers.append(paper)
                    
                except Exception as e:
                    print(f"Error parsing article: {e}")
                    continue  # doesn't break the code - continues with next article
            
            print(f"Successfully parsed {len(papers)} papers from {file_path}")
            print(f"  - New papers inserted: {papers_inserted}")
            print(f"  - Existing papers skipped: {papers_skipped}")
            return papers  
            
        except Exception as e:
            print(f"Error parsing file {file_path}: {e}")
            return []
        
    def parse_all_metadata(self):
        """Parse all XML files and save them to the database"""
        
        # Create directories if they don't exist
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        
        xml_files = list(self.metadata_dir.glob("pubmed_batch_*.xml"))
        
        if not xml_files:
            print("No XML files found to parse")
            return []
            
        total_papers = 0
        for file_path in xml_files:  # find all files that match the pattern 
                                     # "pubmed_batch_.xml" in the metadata directory
            print(f"\nParsing {file_path}...")    
            papers = self.parse_metadata_file(file_path)  # calls previous method
            total_papers += len(papers)
            
        #* database statistics
        stats = self.db_manager.get_database_stats()
        print(f"\n=== Database Statistics ===")
        print(f"Total papers in database: {stats['total_papers']}")
        print(f"Date range: {stats['date_range']}")
        
        if stats['top_journals']:
            print("\nTop Journals:")
            for journal, count in stats['top_journals'][:5]:
                print(f"  - {journal}: {count} papers")

        
        
