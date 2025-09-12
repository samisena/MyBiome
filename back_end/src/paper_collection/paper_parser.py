"""
Paper parser with improved efficiency and error handling.
"""

import xml.etree.ElementTree as ET
from typing import List, Dict, Optional
from pathlib import Path
import sys

# Add the current directory to sys.path for imports
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from ..data.config import config, setup_logging
from .database_manager import database_manager
from ..data.utils import log_execution_time, batch_process

logger = setup_logging(__name__, 'paper_parser.log')


class PubmedParser:
    """
    Enhanced PubMed XML parser with improved error handling and batch processing.
    Uses dependency injection for database management.
    """
    
    def __init__(self, db_manager=None):
        """
        Initialize parser with dependency injection.
        
        Args:
            db_manager: Database manager instance (optional, uses global if None)
        """
        self.db_manager = db_manager or database_manager
        self.metadata_dir = config.paths.metadata_dir
        self.processed_dir = config.paths.processed_data
        
        logger.info("Enhanced PubMed parser initialized")
    
    @log_execution_time
    def parse_metadata_file(self, file_path: str, batch_size: int = 50) -> List[Dict]:
        """
        Parse a single XML metadata file with batch processing.
        
        Args:
            file_path: Path to XML file
            batch_size: Number of papers to process in each batch
            
        Returns:
            List of parsed paper dictionaries
        """
        file_path = Path(file_path)
        logger.info(f"Starting to parse: {file_path}")
        
        try:
            tree = ET.parse(file_path)
            root = tree.getroot()
            articles = root.findall(".//PubmedArticle")
            
            logger.info(f"Found {len(articles)} articles in XML")
            
            if not articles:
                logger.warning("No articles found in XML file")
                return []
            
            # Process articles in batches for better memory management
            all_papers = []
            batches = batch_process(articles, batch_size)
            
            total_inserted = 0
            total_skipped = 0
            
            for i, batch in enumerate(batches, 1):
                logger.info(f"Processing batch {i}/{len(batches)} ({len(batch)} articles)")
                
                batch_papers = []
                for article in batch:
                    try:
                        paper = self._parse_single_article(article)
                        if paper:
                            batch_papers.append(paper)
                    except Exception as e:
                        logger.warning(f"Error parsing article: {e}")
                        continue
                
                # Insert batch to database
                batch_inserted, batch_skipped = self._insert_papers_batch(batch_papers)
                total_inserted += batch_inserted
                total_skipped += batch_skipped
                
                all_papers.extend(batch_papers)
            
            logger.info(f"Parsing completed: {len(all_papers)} papers parsed")
            logger.info(f"Database operations: {total_inserted} new, {total_skipped} existing")
            
            return all_papers
            
        except ET.ParseError as e:
            logger.error(f"XML parsing error for {file_path}: {e}")
            return []
        except Exception as e:
            logger.error(f"Error parsing file {file_path}: {e}")
            return []
    
    def _parse_single_article(self, article: ET.Element) -> Optional[Dict]:
        """
        Parse a single PubmedArticle XML element.
        
        Args:
            article: XML element representing a single article
            
        Returns:
            Paper dictionary or None if parsing failed
        """
        try:
            # Extract PMID (required)
            pmid_elem = article.find(".//PMID")
            if pmid_elem is None or not pmid_elem.text:
                logger.warning("Article missing PMID, skipping...")
                return None
            
            pmid = pmid_elem.text.strip()
            
            # Extract article metadata
            article_meta = article.find(".//Article")
            if article_meta is None:
                logger.warning(f"Article {pmid} missing Article element, skipping...")
                return None
            
            # Extract title (required)
            title_elem = article_meta.find(".//ArticleTitle")
            title = title_elem.text.strip() if title_elem is not None and title_elem.text else "No title available"
            
            # Extract abstract (with structured abstract support)
            abstract = self._extract_abstract(article_meta)
            
            # Extract journal information
            journal = self._extract_journal_info(article_meta)
            
            # Extract publication date
            publication_date = self._extract_publication_date(article_meta)
            
            # Extract DOI
            doi = self._extract_doi(article)
            
            # Extract PMC ID
            pmc_id = self._extract_pmc_id(article)
            
            # Extract keywords
            keywords = self._extract_keywords(article)
            
            # Create paper dictionary
            paper = {
                'pmid': pmid,
                'title': title,
                'abstract': abstract,
                'journal': journal,
                'publication_date': publication_date,
                'doi': doi,
                'pmc_id': pmc_id,
                'keywords': keywords,
                'has_fulltext': False,  # Will be updated by fulltext retriever
                'fulltext_source': None,
                'fulltext_path': None
            }
            
            return paper
            
        except Exception as e:
            logger.error(f"Error parsing individual article: {e}")
            return None
    
    def _extract_abstract(self, article_meta: ET.Element) -> str:
        """Extract abstract text, handling structured abstracts and HTML tags."""
        abstract_parts = []
        
        # Try to find structured abstract first
        abstract_elem = article_meta.find(".//Abstract")
        if abstract_elem is not None:
            # Check for structured abstract with multiple AbstractText elements
            abstract_texts = abstract_elem.findall(".//AbstractText")
            
            if len(abstract_texts) > 1:
                # Structured abstract
                for text_elem in abstract_texts:
                    label = text_elem.get('Label', '').strip()
                    # Use itertext() to get all text including text after child elements
                    content_parts = list(text_elem.itertext())
                    content = ''.join(content_parts).strip()
                    
                    if content:
                        if label:
                            abstract_parts.append(f"{label}: {content}")
                        else:
                            abstract_parts.append(content)
            elif len(abstract_texts) == 1:
                # Simple abstract - use itertext() to get complete text including after HTML tags
                content_parts = list(abstract_texts[0].itertext())
                content = ''.join(content_parts).strip()
                if content:
                    abstract_parts.append(content)
        
        return ' '.join(abstract_parts) if abstract_parts else ''
    
    def _extract_journal_info(self, article_meta: ET.Element) -> str:
        """Extract journal information."""
        # Try journal title first
        journal_elem = article_meta.find(".//Journal/Title")
        if journal_elem is not None and journal_elem.text:
            return journal_elem.text.strip()
        
        # Try ISOAbbreviation as fallback
        iso_elem = article_meta.find(".//Journal/ISOAbbreviation")
        if iso_elem is not None and iso_elem.text:
            return iso_elem.text.strip()
        
        return "Unknown journal"
    
    def _extract_publication_date(self, article_meta: ET.Element) -> str:
        """Extract publication date in YYYY-MM-DD format."""
        pub_date = article_meta.find(".//PubDate")
        if pub_date is None:
            return ""
        
        # Extract year, month, day
        year_elem = pub_date.find(".//Year")
        month_elem = pub_date.find(".//Month")
        day_elem = pub_date.find(".//Day")
        
        year = year_elem.text.strip() if year_elem is not None and year_elem.text else ""
        month = month_elem.text.strip() if month_elem is not None and month_elem.text else ""
        day = day_elem.text.strip() if day_elem is not None and day_elem.text else ""
        
        # Handle month names
        if month and not month.isdigit():
            month_names = {
                'Jan': '01', 'Feb': '02', 'Mar': '03', 'Apr': '04',
                'May': '05', 'Jun': '06', 'Jul': '07', 'Aug': '08',
                'Sep': '09', 'Oct': '10', 'Nov': '11', 'Dec': '12'
            }
            month = month_names.get(month, month)
        
        # Format date
        if year:
            if month:
                if day:
                    return f"{year}-{month.zfill(2)}-{day.zfill(2)}"
                else:
                    return f"{year}-{month.zfill(2)}"
            else:
                return year
        
        return ""
    
    def _extract_doi(self, article: ET.Element) -> Optional[str]:
        """Extract DOI."""
        doi_elem = article.find(".//ArticleId[@IdType='doi']")
        if doi_elem is not None and doi_elem.text:
            return doi_elem.text.strip()
        return None
    
    def _extract_pmc_id(self, article: ET.Element) -> Optional[str]:
        """Extract PMC ID."""
        pmc_elem = article.find(".//ArticleId[@IdType='pmc']")
        if pmc_elem is not None and pmc_elem.text:
            return pmc_elem.text.strip()
        return None
    
    def _extract_keywords(self, article: ET.Element) -> Optional[List[str]]:
        """Extract keywords as a list."""
        keywords = []
        
        # Extract MeSH terms
        mesh_list = article.find(".//MeshHeadingList")
        if mesh_list is not None:
            for mesh_heading in mesh_list.findall(".//MeshHeading"):
                descriptor = mesh_heading.find(".//DescriptorName")
                if descriptor is not None and descriptor.text:
                    keywords.append(descriptor.text.strip())
        
        # Extract author keywords
        keyword_list = article.find(".//KeywordList")
        if keyword_list is not None:
            for keyword_elem in keyword_list.findall(".//Keyword"):
                if keyword_elem.text:
                    keyword = keyword_elem.text.strip()
                    if keyword not in keywords:  # Avoid duplicates
                        keywords.append(keyword)
        
        return keywords if keywords else None
    
    def _insert_papers_batch(self, papers: List[Dict]) -> tuple[int, int]:
        """
        Insert a batch of papers to database.
        
        Args:
            papers: List of paper dictionaries
            
        Returns:
            Tuple of (inserted_count, skipped_count)
        """
        inserted = 0
        skipped = 0
        
        for paper in papers:
            try:
                if self.db_manager.insert_paper(paper):
                    inserted += 1
                else:
                    skipped += 1
            except Exception as e:
                logger.error(f"Error inserting paper {paper.get('pmid', 'unknown')}: {e}")
                skipped += 1
        
        return inserted, skipped
    
    @log_execution_time
    def parse_all_metadata_files(self, pattern: str = "pubmed_batch_*.xml") -> List[Dict]:
        """
        Parse all XML files matching the pattern.
        
        Args:
            pattern: Glob pattern for XML files
            
        Returns:
            List of all parsed papers
        """
        xml_files = list(self.metadata_dir.glob(pattern))
        
        if not xml_files:
            logger.warning(f"No XML files found matching pattern: {pattern}")
            return []
        
        logger.info(f"Found {len(xml_files)} XML files to parse")
        
        all_papers = []
        for i, file_path in enumerate(xml_files, 1):
            logger.info(f"Parsing file {i}/{len(xml_files)}: {file_path.name}")
            
            papers = self.parse_metadata_file(str(file_path))
            all_papers.extend(papers)
        
        # Log final statistics
        stats = self.db_manager.get_database_stats()
        logger.info(f"\n=== Final Database Statistics ===")
        logger.info(f"Total papers in database: {stats['total_papers']}")
        logger.info(f"Processing status breakdown: {stats.get('processing_status', {})}")
        
        return all_papers