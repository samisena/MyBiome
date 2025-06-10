import os 
import json 
import xml.etree.ElementTree as ET  #? Module for processing XML data
from pathlib import Path
from dataclasses import dataclass, asdict 
from typing import List, Dict, Optional, Any

project_root = Path(__file__).parent.parent.parent


@dataclass  #* generates several methods for the class - including __init__()
class Author: 
    last_name: str    #* Type annotations - data type of inputs
    first_name: str
    initials: str
    affiliations: Optional[str] = None
    
@dataclass
class Paper:
    pmid: str  # ID
    title: str
    abstract: str
    authors: List[Author]  # should be a list w/ the elements of the previous class
    journal: str
    publication_date: str
    doi: Optional[str] = None
    keywords: Optional[List[str]] = None  # Fixed: should be List[str], not str

class PubmedParser:
    """This class parses the XML files from the pubmed API into a tree structure, then searches 
    for relevant information within the structure using methods from xml.etree
    
    Returns:
        papers(List): a list of Paper object containing metadata info parsed 
    
    """
    def __init__(self):
        self.metadata_dir = project_root / "data" / "raw" / "metadata"
        self.processed_dir = project_root / "data" / "processed"
        self.ns = {"": "http://www.ncbi.nlm.nih.gov/pubmed"}
        
    def parse_metadata_file(self, file_path):
        """Parse a single XML metadata file and return Paper objects"""
        try:
            tree = ET.parse(file_path)   # Uses ElementTree (ET) to parse the XML file into a tree structure
            root = tree.getroot()  # Gets the root element of the tree
            papers = []  # empty list to store the parsed paper objects
            
            # searching the XML tree for all elements with the tag "PubmedArticle" at any level of the hierarchy
            for article in root.findall(".//PubmedArticle"):  # powerful way to search for matching elements 
                                                            # in an XML structure    
                try:
                    # Extract PMID
                    pmid_elem = article.find(".//PMID")
                    if pmid_elem is None:
                        print("Warning: Article missing PMID, skipping...")
                        continue
                    pmid = pmid_elem.text
                    
                    # Extract article metadata
                    article_meta = article.find(".//Article")
                    if article_meta is None:
                        print(f"Warning: Article {pmid} missing Article element, skipping...")
                        continue
                    
                    # Extract title
                    title_elem = article_meta.find(".//ArticleTitle")
                    title = title_elem.text if title_elem is not None else "No title available"
                    
                    # Extract abstract
                    abstract_elem = article_meta.find(".//Abstract/AbstractText")
                    abstract = abstract_elem.text if abstract_elem is not None else ""
                    
                    # Extract journal
                    journal_elem = article_meta.find(".//Journal/Title")
                    journal = journal_elem.text if journal_elem is not None else "Unknown journal"
                    
                    # Extract publication date
                    pub_date = article_meta.find(".//PubDate")
                    year = ""
                    month = ""
                    day = ""
                    if pub_date is not None:
                        year_elem = pub_date.find(".//Year")
                        month_elem = pub_date.find(".//Month")
                        day_elem = pub_date.find(".//Day")
                        year = year_elem.text if year_elem is not None else ""
                        month = month_elem.text if month_elem is not None else ""
                        day = day_elem.text if day_elem is not None else ""
                    
                    pub_date_str = f"{year}-{month}-{day}" if day else f"{year}-{month}" if month else year
                    
                    # Extract DOI
                    doi_elem = article.find(".//ArticleId[@IdType='doi']")
                    doi = doi_elem.text if doi_elem is not None else None
                    
                    # Extract authors
                    authors = []
                    author_list = article_meta.find(".//AuthorList")
                    if author_list is not None:  # check if parent class exists
                        for author_elem in author_list.findall(".//Author"):
                            last_name_elem = author_elem.find(".//LastName")
                            first_name_elem = author_elem.find(".//ForeName")
                            initials_elem = author_elem.find(".//Initials")
                            affiliation_elem = author_elem.find(".//Affiliation")
                            
                            last_name = last_name_elem.text if last_name_elem is not None else ""
                            first_name = first_name_elem.text if first_name_elem is not None else ""
                            initials = initials_elem.text if initials_elem is not None else ""
                            affiliation = affiliation_elem.text if affiliation_elem is not None else None
                            
                            authors.append(Author(   # Creating an Author object (previously defined)
                                last_name=last_name,
                                first_name=first_name,
                                initials=initials,
                                affiliations=affiliation  # Fixed: should be affiliations not affiliation
                            ))
                    
                    # Extract keywords        
                    keywords = []
                    keyword_list = article.find(".//KeywordList")
                    if keyword_list is not None:
                        for keyword_elem in keyword_list.findall(".//Keyword"):
                            if keyword_elem.text:
                                keywords.append(keyword_elem.text)
                    
                    # Create Paper object
                    paper = Paper(
                        pmid=pmid,
                        title=title,
                        abstract=abstract,
                        authors=authors,  # takes the authors list with Author Objects
                        journal=journal,
                        publication_date=pub_date_str,
                        doi=doi,
                        keywords=keywords if keywords else None
                    )

                    papers.append(paper)
                    
                except Exception as e:
                    print(f"Error parsing article: {e}")
                    continue  # doesn't break the code - continues with next article
                    
            print(f"Successfully parsed {len(papers)} papers from {file_path}")
            return papers  # Fixed indentation: moved outside the for loop
            
        except Exception as e:
            print(f"Error parsing file {file_path}: {e}")
            return []
        
    def parse_all_metadata(self):
        """Saves the results as a JSON file"""
        all_papers = []
        
        # Create directories if they doesn't exist
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        
        # Fixed: self.metadata.dir -> self.metadata_dir
        xml_files = list(self.metadata_dir.glob("pubmed_batch_*.xml"))
        
        if not xml_files:
            print("No XML files found to parse")
            return []
            
        for file_path in xml_files:  # find all files that match the 
                                    # pattern "pubmed_batch_.xml" in the metadata directory
            print(f"Parsing {file_path}...")    
            papers = self.parse_metadata_file(file_path)  # calls previous method
            all_papers.extend(papers)  # using extend instead of append not to end up with nested list 
                                        # structure rather than a flat list of papers because papers is itself a list
        
        if not all_papers:
            print("No papers were successfully parsed")
            return []
            
        # Convert Paper objects to dictionaries
        papers_dict = [asdict(paper) for paper in all_papers]  # converts each Paper object to a dictionary 
                                                                # using the asdict() function.
        output_file = self.processed_dir / "all_papers.json" 
        
        # Writes the parsed data to a file as JSON
        try:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(papers_dict, f, indent=2, ensure_ascii=False) 
            print(f"Parsed {len(all_papers)} papers and saved to {output_file}")
        except Exception as e:
            print(f"Error saving JSON file: {e}")
            return []
            
        return all_papers
                
            
            
        
                
                    

