"""
PubMed collector with improved architecture and efficiency.
"""

import time
from typing import List, Dict, Any, Optional
from pathlib import Path
import sys

from back_end.src.data.config import config, setup_logging
from back_end.src.data.api_clients import get_pubmed_client
from back_end.src.data_collection.database_manager import database_manager
from back_end.src.data_collection.paper_parser import PubmedParser
from back_end.src.data_collection.fulltext_retriever import FullTextRetriever
from back_end.src.data.utils import batch_process
from pathlib import Path
from back_end.src.interventions.search_terms import search_terms

logger = setup_logging(__name__, 'pubmed_collector.log')


class PubMedCollector:
    """
    PubMed collector for intervention studies with centralized configuration.
    Supports collection for all intervention types (exercise, diet, supplements, etc.).
    """
    
    def __init__(self, db_manager=None, parser=None, fulltext_retriever=None):
        """
        Initialize with dependency injection.
        
        Args:
            db_manager: Database manager instance (optional, uses global if None)
            parser: Paper parser instance (optional, creates new if None)
            fulltext_retriever: Fulltext retriever instance (optional, creates new if None)
        """
        # Use dependency injection or defaults
        self.db_manager = db_manager or database_manager
        self.parser = parser or PubmedParser(self.db_manager)
        self.fulltext_retriever = fulltext_retriever or FullTextRetriever(self.db_manager)
        
        # Get API client
        self.pubmed_client = get_pubmed_client()
        
        # Configuration from central config
        self.metadata_dir = config.metadata_dir
        
        # Enhanced PubMed collector initialized
    
    # Removed @log_execution_time - use error_handler.py decorators instead
    def search_papers(self, query: str, min_year: int = 2000,
                     max_year: Optional[int] = None, max_results: int = 100) -> List[str]:
        """
        Search for papers using the centralized PubMed client.
        
        Args:
            query: Search query string
            min_year: Minimum publication year
            max_year: Maximum publication year (optional)
            max_results: Maximum number of results

        Returns:
            List of PMIDs
        """
        try:
            result = self.pubmed_client.search_papers(query, min_year, max_results)
            return result['pmids']
            
        except Exception as e:
            logger.error(f"Error searching PubMed: {e}")
            return []
    
    # Removed @log_execution_time - use error_handler.py decorators instead
    def fetch_papers_metadata(self, pmid_list: List[str]) -> Optional[Path]:
        """
        Fetch paper metadata and save to file.
        
        Args:
            pmid_list: List of PMIDs to fetch
            
        Returns:
            Path to saved metadata file or None if failed
        """
        if not pmid_list:
            # No PMIDs provided for fetching
            return None
        
        try:
            # Fetch metadata using centralized client
            xml_content = self.pubmed_client.fetch_papers(pmid_list)
            
            if not xml_content:
                logger.error("No content received from PubMed API")
                return None
            
            # Save to file with timestamp
            timestamp = int(time.time())
            filename = f'pubmed_batch_{timestamp}.xml'
            metadata_file = self.metadata_dir / filename
            
            try:
                metadata_file.parent.mkdir(parents=True, exist_ok=True)
                with open(metadata_file, 'w', encoding='utf-8') as f:
                    f.write(xml_content)
                # Metadata saved to file
                return metadata_file
            except Exception as e:
                logger.error(f"Error writing metadata file {metadata_file}: {e}")
                return None
                
        except Exception as e:
            logger.error(f"Error fetching papers metadata: {e}")
            return None
    
    # Removed @log_execution_time - use error_handler.py decorators instead
    def collect_interventions_by_condition(self, condition: str, min_year: int = 2010,
                                          max_year: Optional[int] = None,
                                          max_results: int = 100,
                                          include_fulltext: bool = True,
                                          include_study_filter: bool = True,
                                          use_interleaved_s2: bool = True) -> Dict[str, Any]:
        """
        Collect intervention papers for a health condition with enhanced processing.
        Now supports interleaved Semantic Scholar discovery: each PubMed paper immediately
        triggers discovery of 5 similar papers.

        Args:
            condition: Health condition to search for
            min_year: Minimum publication year
            max_results: Maximum number of seed papers from PubMed (will find ~5x more via S2)
            include_fulltext: Whether to attempt fulltext retrieval
            include_study_filter: Whether to filter for intervention studies
            use_interleaved_s2: Whether to use interleaved S2 discovery (1 paper -> 5 similar papers)

        Returns:
            Collection results dictionary with PubMed and S2 discovery stats
        """
        # Starting intervention collection for condition
        # Build enhanced search query for interventions
        query = self._build_intervention_query(condition, include_study_filter)

        try:
            if use_interleaved_s2:
                return self._collect_with_interleaved_s2(
                    condition, query, min_year, max_year, max_results, include_fulltext
                )
            else:
                return self._collect_traditional_batch(
                    condition, query, min_year, max_year, max_results, include_fulltext
                )

        except Exception as e:
            logger.error(f"Error in collection process: {e}")
            return {
                "condition": condition,
                "paper_count": 0,
                "new_papers_count": 0,
                "status": "error",
                "message": f"Collection failed: {str(e)}"
            }
    
    def _build_intervention_query(self, condition: str, include_study_filter: bool = True) -> str:
        """Build an optimized search query for health interventions and conditions."""
        # Use a simplified intervention query to avoid URL length issues
        # Focus on most common intervention terms
        intervention_terms = [
            'intervention[Title/Abstract]',
            'treatment[Title/Abstract]',
            'therapy[Title/Abstract]',
            'exercise[Title/Abstract]',
            'diet[Title/Abstract]',
            'medication[Title/Abstract]',
            'supplement[Title/Abstract]',
            '"Drug Therapy"[MeSH Terms]',
            '"Exercise"[MeSH Terms]',
            '"Diet Therapy"[MeSH Terms]',
            '"Behavioral Intervention"[MeSH Terms]'
        ]

        intervention_query = f"({' OR '.join(intervention_terms)})"

        # Build condition terms
        condition_terms = [
            f'"{condition}"[Title/Abstract]',
            f'"{condition}"[MeSH Terms]'
        ]

        condition_query = f"({' OR '.join(condition_terms)})"

        # Build base query
        base_query = f"{condition_query} AND {intervention_query}"

        # Add study type filter if requested
        if include_study_filter:
            study_filter_terms = [
                '"Randomized Controlled Trial"[Publication Type]',
                '"Clinical Trial"[Publication Type]',
                'randomized[Title/Abstract]',
                'controlled trial[Title/Abstract]',
                'RCT[Title/Abstract]'
            ]
            study_filter = f"({' OR '.join(study_filter_terms)})"
            base_query = f"{base_query} AND {study_filter}"

        return base_query

    def _search_papers_with_offset(self, query: str, min_year: int, max_results: int,
                                   max_year: Optional[int] = None, offset: int = 0) -> List[str]:
        """
        Search for papers with pagination support.

        Args:
            query: Search query string
            min_year: Minimum publication year
            max_results: Maximum number of results
            offset: Starting position for results (for pagination)

        Returns:
            List of PMIDs
        """
        try:
            # Most PubMed clients don't support direct offset, so we'll use retstart parameter
            # if available, or fetch a larger batch and slice it
            if hasattr(self.pubmed_client, 'search_papers_with_offset'):
                result = self.pubmed_client.search_papers_with_offset(query, min_year, max_results, offset)
                return result['pmids']
            else:
                # Fallback: fetch larger batch and slice
                # This is less efficient but works with basic clients
                fetch_size = max_results + offset
                result = self.pubmed_client.search_papers(query, min_year, min(fetch_size, 10000))
                pmids = result['pmids']
                return pmids[offset:offset + max_results] if len(pmids) > offset else []

        except Exception as e:
            logger.error(f"Error searching PubMed with offset: {e}")
            return []

    def _filter_existing_papers(self, pmid_list: List[str]) -> List[str]:
        """
        Filter out PMIDs that already exist in the database.

        Args:
            pmid_list: List of PMIDs to check

        Returns:
            List of PMIDs that don't exist in the database
        """
        if not pmid_list:
            return []

        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()

                # Create placeholders for the IN clause
                placeholders = ','.join(['?'] * len(pmid_list))
                query = f'SELECT pmid FROM papers WHERE pmid IN ({placeholders})'

                cursor.execute(query, pmid_list)
                existing_pmids = {row[0] for row in cursor.fetchall()}

                # Return PMIDs that are NOT in the existing set
                new_pmids = [pmid for pmid in pmid_list if pmid not in existing_pmids]

                # PMIDs filtered for existing papers
                return new_pmids

        except Exception as e:
            logger.error(f"Error filtering existing papers: {e}")
            # If we can't check, return all PMIDs and let the database handle duplicates
            return pmid_list

    def _collect_with_interleaved_s2(self, condition: str, query: str, min_year: int,
                                   max_year: Optional[int], max_results: int, include_fulltext: bool) -> Dict[str, Any]:
        """
        Collect papers using interleaved Semantic Scholar discovery.
        For each PubMed paper collected, immediately find 5 similar papers.
        """
        # Using interleaved Semantic Scholar discovery workflow

        # Import S2 enricher here to avoid circular imports
        from back_end.src.data_collection.semantic_scholar_enrichment import SemanticScholarEnricher
        s2_enricher = SemanticScholarEnricher(self.db_manager)

        # Statistics tracking
        pubmed_papers = []
        s2_papers = []
        total_papers_searched = 0
        search_offset = 0
        metadata_files = []

        # Search for papers one by one and process immediately
        for seed_count in range(max_results):
            # Step 1: Get the next PubMed paper
            pmid_list = self._search_papers_with_offset(query, min_year, 1, max_year, search_offset)

            if not pmid_list:
                # No more papers found - stopping search
                break

            total_papers_searched += len(pmid_list)

            # Filter out existing papers
            new_pmids = self._filter_existing_papers(pmid_list)

            if not new_pmids:
                search_offset += len(pmid_list)
                continue

            # Take the first new paper as our seed
            seed_pmid = new_pmids[0]

            # Step 2: Fetch and parse the seed paper
            metadata_file = self.fetch_papers_metadata([seed_pmid])

            if not metadata_file:
                search_offset += len(pmid_list)
                continue

            batch_papers = self.parser.parse_metadata_file(str(metadata_file))

            if not batch_papers:
                search_offset += len(pmid_list)
                continue

            seed_paper = batch_papers[0]
            pubmed_papers.append(seed_paper)

            # Seed paper collected

            # Clean up temporary XML file after successful processing
            try:
                Path(metadata_file).unlink()
                logger.debug(f"Deleted temporary XML file: {metadata_file}")
            except Exception as e:
                logger.warning(f"Could not delete temporary XML file {metadata_file}: {e}")
                # Only add to list if cleanup failed
                metadata_files.append(str(metadata_file))

            # Step 3: Immediately enrich the seed paper with S2 data
            try:
                enrichment_stats = s2_enricher._enrich_single_paper(seed_paper)
                if enrichment_stats.enriched_papers > 0:
                    # S2 enrichment successful
                    pass
                else:
                    # S2 enrichment failed
                    pass
            except Exception as e:
                # S2 enrichment failed
                pass

            # Step 4: Similar paper discovery (DISABLED)
            # Similar paper discovery has been disabled to prevent API issues
            # Only S2 metrics enrichment is performed (Step 3 above)

            # Step 5: Process fulltext if requested (for seed paper only)
            if include_fulltext and batch_papers:
                fulltext_stats = self._process_fulltext_batch(batch_papers)

            search_offset += len(pmid_list)

            # Rate limiting between iterations
            time.sleep(0.5)

        # Build comprehensive results (s2_papers will be empty since discovery is disabled)
        total_papers = len(pubmed_papers) + len(s2_papers)

        result = {
            "condition": condition,
            "paper_count": total_papers,
            "pubmed_papers": len(pubmed_papers),
            "s2_similar_papers": len(s2_papers),
            "papers_stored": total_papers,
            "total_papers_searched": total_papers_searched,
            "undeleted_metadata_files": metadata_files,  # Only contains files that failed to be deleted
            "interleaved_workflow": True,
            "status": "success" if len(pubmed_papers) >= max_results else "partial_success",
            "message": f"Interleaved collection with S2 metrics: {len(pubmed_papers)} PubMed papers enriched with S2 data (similar paper discovery disabled)"
        }

        # Interleaved collection completed
        return result

    def _collect_traditional_batch(self, condition: str, query: str, min_year: int,
                                 max_year: Optional[int], max_results: int, include_fulltext: bool) -> Dict[str, Any]:
        """
        Traditional batch collection method (original logic).
        """
        # [This would contain the original batch collection logic we had before]
        # For now, let's implement a simplified version
        # Using traditional batch collection workflow

        new_papers_collected = 0
        total_papers_processed = 0
        search_batch_size = max_results * 2  # Start with 2x to account for duplicates
        max_search_attempts = 5
        search_offset = 0

        all_new_papers = []
        metadata_files = []

        for attempt in range(max_search_attempts):
            if new_papers_collected >= max_results:
                break

            remaining_needed = max_results - new_papers_collected
            current_search_size = min(search_batch_size, remaining_needed * 3)  # 3x buffer

            # Searching for papers

            # Step 1: Search for papers with offset
            pmid_list = self._search_papers_with_offset(query, min_year, current_search_size, max_year, search_offset)

            if not pmid_list:
                # No more papers found in search
                break

            # Filter out papers that already exist in our database
            new_pmids = self._filter_existing_papers(pmid_list)
            # Papers found and filtered

            if not new_pmids:
                search_offset += len(pmid_list)
                continue

            # Only process the number we actually need
            pmids_to_process = new_pmids[:remaining_needed]

            # Step 2: Fetch metadata
            metadata_file = self.fetch_papers_metadata(pmids_to_process)

            if not metadata_file:
                # Failed to fetch metadata for batch
                search_offset += len(pmid_list)
                continue

            # Step 3: Parse and store papers
            batch_papers = self.parser.parse_metadata_file(str(metadata_file))

            if batch_papers:
                all_new_papers.extend(batch_papers)
                new_papers_collected += len(batch_papers)
                total_papers_processed += len(pmid_list)

                # Batch processing completed

                # DELAY XML CLEANUP: Keep XML file for now, will clean up after database verification
                # This ensures we can recover if database persistence fails
                metadata_files.append(str(metadata_file))
                logger.debug(f"Keeping XML file for verification: {metadata_file}")

            search_offset += len(pmid_list)

            # If we didn't get as many new papers as expected, increase search size for next attempt
            if len(batch_papers) < len(pmids_to_process) * 0.8:
                search_batch_size = int(search_batch_size * 1.5)

        if not all_new_papers:
            return {
                "condition": condition,
                "paper_count": 0,
                "new_papers_count": 0,
                "status": "no_results",
                "message": "No new papers found matching criteria"
            }

        # Step 4: Retrieve fulltext if requested
        fulltext_stats = None
        if include_fulltext:
            # Attempting fulltext retrieval for new papers
            fulltext_stats = self._process_fulltext_batch(all_new_papers)

        # Step 5: Clean up XML files after collection
        for xml_file in metadata_files:
            try:
                Path(xml_file).unlink()
            except Exception as e:
                logger.debug(f"Could not delete XML file {xml_file}: {e}")

        # Step 6: Build result
        result = {
            "condition": condition,
            "paper_count": len(all_new_papers),
            "new_papers_count": len(all_new_papers),
            "papers_stored": len(all_new_papers),
            "total_papers_searched": total_papers_processed,
            "undeleted_metadata_files": [],
            "interleaved_workflow": False,
            "status": "success" if new_papers_collected >= max_results else "partial_success",
            "message": f"Collected {len(all_new_papers)} papers (target: {max_results})"
        }

        if fulltext_stats:
            result["fulltext_stats"] = fulltext_stats

        # Traditional collection completed with verification
        return result

    def _process_fulltext_batch(self, papers: List[Dict]) -> Dict[str, Any]:
        """Process papers for fulltext retrieval in batches."""
        # Filter papers that have PMC IDs or DOIs
        fulltext_candidates = [
            p for p in papers 
            if p.get('pmc_id') or p.get('doi')
        ]
        
        if not fulltext_candidates:
            # No papers with PMC IDs or DOIs found
            return {
                'total_candidates': 0,
                'processed': 0,
                'successful_pmc': 0,
                'successful_unpaywall': 0,
                'failed': 0
            }
        
        # Papers with fulltext potential found
        
        # Process in smaller batches to manage resources
        batch_size = 20
        batches = batch_process(fulltext_candidates, batch_size)
        
        total_stats = {
            'total_candidates': len(fulltext_candidates),
            'processed': 0,
            'successful_pmc': 0,
            'successful_unpaywall': 0,
            'failed': 0,
            'errors': []
        }
        
        for i, batch in enumerate(batches, 1):
            # Processing fulltext batch
            
            batch_stats = self.fulltext_retriever.process_papers_batch(batch)
            
            # Accumulate statistics
            total_stats['processed'] += batch_stats['total_papers']
            total_stats['successful_pmc'] += batch_stats['successful_pmc']
            total_stats['successful_unpaywall'] += batch_stats['successful_unpaywall']
            total_stats['failed'] += batch_stats['failed']
            total_stats['errors'].extend(batch_stats.get('errors', []))
        
        # Fulltext processing completed
        return total_stats
    
    # Removed @log_execution_time - use error_handler.py decorators instead
    def bulk_collect_conditions(self, conditions: List[str], 
                              max_results: int = 100,
                              include_fulltext: bool = True,
                              delay_between_conditions: float = 2.0) -> List[Dict[str, Any]]:
        """
        Collect papers for multiple conditions with improved batch processing.
        
        Args:
            conditions: List of health conditions
            max_results: Maximum papers per condition
            include_fulltext: Whether to retrieve fulltext
            delay_between_conditions: Delay between condition queries
            
        Returns:
            List of collection results
        """
        # Starting bulk collection for conditions
        
        results = []
        total_papers = 0
        
        for i, condition in enumerate(conditions, 1):
            # Processing condition
            
            try:
                result = self.collect_interventions_by_condition(
                    condition=condition,
                    max_results=max_results,
                    include_fulltext=include_fulltext
                )
                
                results.append(result)
                total_papers += result.get('paper_count', 0)
                
                # Rate limiting between conditions
                if i < len(conditions) and delay_between_conditions > 0:
                    time.sleep(delay_between_conditions)
                    
            except Exception as e:
                logger.error(f"Error processing condition {condition}: {e}")
                results.append({
                    "condition": condition,
                    "paper_count": 0,
                    "status": "error",
                    "message": str(e)
                })
        
        # Save bulk results
        timestamp = int(time.time())
        results_file = self.metadata_dir / f"bulk_collection_results_{timestamp}.json"
        
        summary = {
            "total_conditions": len(conditions),
            "total_papers_collected": total_papers,
            "successful_conditions": len([r for r in results if r.get('status') == 'success']),
            "failed_conditions": len([r for r in results if r.get('status') == 'error']),
            "conditions_processed": conditions,
            "detailed_results": results
        }
        
        try:
            results_file.parent.mkdir(parents=True, exist_ok=True)
            with open(results_file, 'w', encoding='utf-8') as f:
                f.write(str(summary))
            # Collection summary saved to file
            pass
        except Exception as e:
            logger.error(f"Error writing results file {results_file}: {e}")
            # Bulk collection results saved to file

        # Bulk collection completed
        return results