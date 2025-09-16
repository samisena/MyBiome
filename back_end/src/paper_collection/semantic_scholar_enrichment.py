"""
Semantic Scholar enrichment module for paper collection pipeline.
Adds influence scores, AI summaries, and discovers similar papers.
"""

import json
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict

from src.data.config import setup_logging
from src.data.api_clients import get_semantic_scholar_client
from src.paper_collection.database_manager import database_manager

logger = setup_logging(__name__, 'semantic_scholar.log')


@dataclass
class EnrichmentStats:
    """Statistics for tracking enrichment progress."""
    total_papers: int = 0
    enriched_papers: int = 0
    failed_papers: int = 0
    new_papers_found: int = 0
    duplicate_papers: int = 0
    errors: List[str] = None

    def __post_init__(self):
        if self.errors is None:
            self.errors = []


class SemanticScholarEnricher:
    """Handles Semantic Scholar enrichment and paper discovery."""

    def __init__(self, db_manager=None):
        self.db_manager = db_manager or database_manager
        self.s2_client = get_semantic_scholar_client()
        self.batch_size = 500  # S2 API limit

    def enrich_existing_papers(self, limit: Optional[int] = None) -> EnrichmentStats:
        """
        Enrich existing papers with Semantic Scholar data.

        Args:
            limit: Maximum number of papers to process (None = all)

        Returns:
            EnrichmentStats with processing results
        """
        logger.info("Starting Semantic Scholar enrichment of existing papers")
        stats = EnrichmentStats()

        try:
            # Get papers that haven't been processed by S2 yet
            unprocessed_papers = self._get_unprocessed_papers(limit)
            stats.total_papers = len(unprocessed_papers)

            logger.info(f"Found {stats.total_papers} papers to enrich with Semantic Scholar data")

            if not unprocessed_papers:
                logger.info("No papers need S2 enrichment")
                return stats

            # Process papers in batches
            for batch in self._batch_papers(unprocessed_papers, self.batch_size):
                batch_stats = self._enrich_paper_batch(batch)
                stats.enriched_papers += batch_stats.enriched_papers
                stats.failed_papers += batch_stats.failed_papers
                stats.errors.extend(batch_stats.errors)

                # Progress update
                processed = stats.enriched_papers + stats.failed_papers
                logger.info(f"Progress: {processed}/{stats.total_papers} papers processed")

                # Rate limiting - S2 recommends no more than 100 requests per second
                time.sleep(1.0)

            logger.info(f"S2 enrichment completed: {stats.enriched_papers} enriched, {stats.failed_papers} failed")
            return stats

        except Exception as e:
            logger.error(f"Error during S2 enrichment: {e}")
            stats.errors.append(f"Enrichment failed: {str(e)}")
            return stats

    def discover_similar_papers(self, seed_papers: Optional[List[str]] = None,
                               limit_per_paper: int = 5) -> EnrichmentStats:
        """
        Discover similar papers using Semantic Scholar recommendations.

        Args:
            seed_papers: List of PMIDs to use as seeds (None = use enriched papers)
            limit_per_paper: Max similar papers to find per seed paper

        Returns:
            EnrichmentStats with discovery results
        """
        logger.info("Starting similar paper discovery via Semantic Scholar")
        stats = EnrichmentStats()

        try:
            # Get seed papers (papers that have been S2 enriched)
            if seed_papers is None:
                seed_papers = self._get_enriched_papers_for_similarity()

            logger.info(f"Using {len(seed_papers)} papers as seeds for similarity search")

            discovered_papers = []
            processed_seeds = 0

            for pmid in seed_papers:
                try:
                    # Find similar papers for this seed
                    similar_papers = self.s2_client.get_similar_papers(
                        pmid, limit=limit_per_paper
                    )

                    # Convert S2 format to our paper format and check for duplicates
                    for s2_paper in similar_papers:
                        converted_paper = self._convert_s2_to_paper_format(s2_paper)
                        if converted_paper and not self._is_duplicate_paper(converted_paper):
                            discovered_papers.append(converted_paper)
                        else:
                            stats.duplicate_papers += 1

                    processed_seeds += 1
                    if processed_seeds % 10 == 0:
                        logger.info(f"Processed {processed_seeds}/{len(seed_papers)} seed papers")

                    # Rate limiting
                    time.sleep(0.1)

                except Exception as e:
                    logger.warning(f"Failed to get similar papers for PMID {pmid}: {e}")
                    stats.errors.append(f"Similar papers failed for {pmid}: {str(e)}")

            # Insert discovered papers
            if discovered_papers:
                logger.info(f"Inserting {len(discovered_papers)} newly discovered papers")
                inserted_count, failed_count = self.db_manager.insert_papers_batch(discovered_papers)
                stats.new_papers_found = inserted_count
                stats.failed_papers = failed_count

            logger.info(f"Similar paper discovery completed: {stats.new_papers_found} new papers found, "
                       f"{stats.duplicate_papers} duplicates skipped")
            return stats

        except Exception as e:
            logger.error(f"Error during similar paper discovery: {e}")
            stats.errors.append(f"Discovery failed: {str(e)}")
            return stats

    def run_full_enrichment_pipeline(self, enrich_limit: Optional[int] = None) -> Dict:
        """
        Run the complete S2 enrichment pipeline:
        1. Enrich existing papers with S2 data
        2. Discover similar papers (one iteration only)

        Args:
            enrich_limit: Limit for paper enrichment (None = all)

        Returns:
            Combined results from both steps
        """
        logger.info("Starting full Semantic Scholar enrichment pipeline")
        pipeline_start = time.time()

        # Step 1: Enrich existing papers
        logger.info("=== Step 1: Enriching existing papers ===")
        enrichment_stats = self.enrich_existing_papers(limit=enrich_limit)

        # Step 2: Discover similar papers (only if enrichment was successful)
        logger.info("=== Step 2: Discovering similar papers ===")
        discovery_stats = EnrichmentStats()
        if enrichment_stats.enriched_papers > 0:
            discovery_stats = self.discover_similar_papers(limit_per_paper=5)
        else:
            logger.warning("Skipping similar paper discovery - no papers were enriched")

        # Combine results
        total_time = time.time() - pipeline_start

        results = {
            'enrichment': {
                'total_papers': enrichment_stats.total_papers,
                'enriched_papers': enrichment_stats.enriched_papers,
                'failed_papers': enrichment_stats.failed_papers,
                'errors': enrichment_stats.errors
            },
            'discovery': {
                'new_papers_found': discovery_stats.new_papers_found,
                'duplicate_papers': discovery_stats.duplicate_papers,
                'failed_papers': discovery_stats.failed_papers,
                'errors': discovery_stats.errors
            },
            'pipeline': {
                'total_time_seconds': round(total_time, 2),
                'status': 'success' if not (enrichment_stats.errors or discovery_stats.errors) else 'partial_success'
            }
        }

        logger.info(f"Full S2 enrichment pipeline completed in {total_time:.1f}s")
        logger.info(f"Results: {enrichment_stats.enriched_papers} enriched, "
                   f"{discovery_stats.new_papers_found} new papers discovered")

        return results

    # Private helper methods

    def _get_unprocessed_papers(self, limit: Optional[int] = None) -> List[Dict]:
        """Get papers that haven't been processed by Semantic Scholar yet."""
        with self.db_manager.get_connection() as conn:
            cursor = conn.cursor()

            query = '''
                SELECT pmid, title, abstract, doi, journal, publication_date
                FROM papers
                WHERE (s2_processed = 0 OR s2_processed IS NULL OR s2_processed = 'false')
                ORDER BY created_at DESC
            '''

            if limit:
                query += f' LIMIT {limit}'

            cursor.execute(query)
            return [dict(row) for row in cursor.fetchall()]

    def _get_enriched_papers_for_similarity(self) -> List[str]:
        """Get PMIDs of papers that have been enriched and can be used for similarity search."""
        with self.db_manager.get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute('''
                SELECT pmid FROM papers
                WHERE s2_processed = 1 AND s2_paper_id IS NOT NULL
                ORDER BY influence_score DESC
                LIMIT 50
            ''')

            return [row[0] for row in cursor.fetchall()]

    def _batch_papers(self, papers: List[Dict], batch_size: int) -> List[List[Dict]]:
        """Split papers into batches for API processing."""
        for i in range(0, len(papers), batch_size):
            yield papers[i:i + batch_size]

    def _enrich_paper_batch(self, papers: List[Dict]) -> EnrichmentStats:
        """Enrich a batch of papers with S2 data."""
        batch_stats = EnrichmentStats()

        try:
            # Extract identifiers for S2 API
            paper_ids = []
            pmid_to_paper = {}

            for paper in papers:
                pmid = paper['pmid']
                pmid_to_paper[pmid] = paper

                # Prefer DOI, fallback to PMID
                if paper.get('doi'):
                    paper_ids.append(paper['doi'])
                else:
                    paper_ids.append(pmid)

            # Get S2 data for batch
            s2_papers = self.s2_client.get_papers_batch(paper_ids)

            # Match S2 results back to our papers and update database
            for s2_paper in s2_papers:
                if s2_paper is None:
                    continue

                # Find matching PMID
                matching_pmid = self._find_matching_pmid(s2_paper, pmid_to_paper)
                if matching_pmid:
                    success = self._update_paper_with_s2_data(matching_pmid, s2_paper)
                    if success:
                        batch_stats.enriched_papers += 1
                    else:
                        batch_stats.failed_papers += 1

            # Mark remaining papers as processed (even if no S2 data found)
            for pmid in pmid_to_paper.keys():
                if not self._was_paper_enriched(pmid):
                    self._mark_paper_s2_processed(pmid)

        except Exception as e:
            logger.error(f"Error enriching paper batch: {e}")
            batch_stats.errors.append(f"Batch enrichment failed: {str(e)}")
            batch_stats.failed_papers = len(papers)

        return batch_stats

    def _find_matching_pmid(self, s2_paper: Dict, pmid_to_paper: Dict) -> Optional[str]:
        """Find which PMID matches the S2 paper result."""
        # Try to match by external IDs first
        external_ids = s2_paper.get('externalIds', {})
        if external_ids.get('PubMed'):
            pmid = external_ids['PubMed']
            if pmid in pmid_to_paper:
                return pmid

        # Try to match by DOI
        if external_ids.get('DOI'):
            doi = external_ids['DOI']
            for pmid, paper in pmid_to_paper.items():
                if paper.get('doi') == doi:
                    return pmid

        # Try to match by title similarity as last resort
        s2_title = s2_paper.get('title', '').lower().strip()
        if s2_title:
            for pmid, paper in pmid_to_paper.items():
                paper_title = paper.get('title', '').lower().strip()
                if paper_title and s2_title == paper_title:
                    return pmid

        return None

    def _update_paper_with_s2_data(self, pmid: str, s2_paper: Dict) -> bool:
        """Update a paper record with Semantic Scholar data."""
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()

                # Extract S2 data
                s2_paper_id = s2_paper.get('paperId')
                citation_count = s2_paper.get('citationCount', 0)
                influence_score = s2_paper.get('influentialCitationCount', 0)
                tldr_data = s2_paper.get('tldr')
                tldr_text = tldr_data.get('text') if tldr_data else None
                embedding = s2_paper.get('embedding')
                embedding_json = json.dumps(embedding) if embedding else None

                cursor.execute('''
                    UPDATE papers SET
                        s2_paper_id = ?,
                        citation_count = ?,
                        influence_score = ?,
                        tldr = ?,
                        s2_embedding = ?,
                        s2_processed = 1,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE pmid = ?
                ''', (s2_paper_id, citation_count, influence_score,
                      tldr_text, embedding_json, pmid))

                conn.commit()
                logger.debug(f"Enriched paper {pmid} with S2 data")
                return True

        except Exception as e:
            logger.error(f"Failed to update paper {pmid} with S2 data: {e}")
            return False

    def _mark_paper_s2_processed(self, pmid: str) -> bool:
        """Mark a paper as processed by S2 (even if no data was found)."""
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    UPDATE papers SET s2_processed = 1, updated_at = CURRENT_TIMESTAMP
                    WHERE pmid = ?
                ''', (pmid,))
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Failed to mark paper {pmid} as S2 processed: {e}")
            return False

    def _was_paper_enriched(self, pmid: str) -> bool:
        """Check if a paper was successfully enriched with S2 data."""
        with self.db_manager.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT s2_paper_id FROM papers WHERE pmid = ? AND s2_paper_id IS NOT NULL
            ''', (pmid,))
            return cursor.fetchone() is not None

    def _enrich_single_paper(self, paper: Dict) -> EnrichmentStats:
        """Enrich a single paper with S2 data."""
        stats = EnrichmentStats()
        stats.total_papers = 1

        try:
            pmid = paper['pmid']

            # Use DOI if available, otherwise PMID
            paper_id = paper.get('doi', pmid)

            # Get S2 data for this paper
            s2_paper = self.s2_client.get_paper(paper_id)

            if s2_paper:
                success = self._update_paper_with_s2_data(pmid, s2_paper)
                if success:
                    stats.enriched_papers = 1
                else:
                    stats.failed_papers = 1
                    stats.errors.append(f"Failed to update {pmid} with S2 data")
            else:
                # Mark as processed even if no S2 data found
                self._mark_paper_s2_processed(pmid)
                stats.failed_papers = 1
                stats.errors.append(f"No S2 data found for {pmid}")

        except Exception as e:
            stats.failed_papers = 1
            stats.errors.append(f"Error enriching {paper.get('pmid', 'unknown')}: {str(e)}")

        return stats

    def _convert_s2_to_paper_format(self, s2_paper: Dict) -> Optional[Dict]:
        """Convert Semantic Scholar paper format to our paper format."""
        try:
            external_ids = s2_paper.get('externalIds', {})

            # We need at least a PMID or DOI to be useful
            pmid = external_ids.get('PubMed')
            doi = external_ids.get('DOI')

            if not pmid and not doi:
                return None

            # Extract authors
            authors_data = s2_paper.get('authors', [])
            authors = [author.get('name', '') for author in authors_data if author.get('name')]
            authors_str = ', '.join(authors) if authors else None

            # Extract journal info
            journal_data = s2_paper.get('journal', {})
            journal = journal_data.get('name') if journal_data else None

            paper = {
                'pmid': pmid if pmid else f"S2_{s2_paper.get('paperId', '')}",
                'title': s2_paper.get('title', ''),
                'abstract': s2_paper.get('abstract', ''),
                'doi': doi,
                'journal': journal,
                'publication_date': str(s2_paper.get('year', '')),
                'keywords': json.dumps([]) if not s2_paper.get('fieldsOfStudy') else json.dumps(s2_paper.get('fieldsOfStudy', [])),

                # S2 specific fields
                's2_paper_id': s2_paper.get('paperId'),
                'citation_count': s2_paper.get('citationCount', 0),
                'influence_score': s2_paper.get('influentialCitationCount', 0),
                'tldr': s2_paper.get('tldr', {}).get('text') if s2_paper.get('tldr') else None,
                's2_processed': True,

                # Default values
                'has_fulltext': False,
                'processing_status': 'pending',
                'discovery_source': 'semantic_scholar'
            }

            return paper

        except Exception as e:
            logger.error(f"Failed to convert S2 paper format: {e}")
            return None

    def _is_duplicate_paper(self, paper: Dict) -> bool:
        """Check if a paper already exists in our database."""
        pmid = paper.get('pmid')
        doi = paper.get('doi')
        title = paper.get('title', '').lower().strip()

        with self.db_manager.get_connection() as conn:
            cursor = conn.cursor()

            # Check by PMID
            if pmid and not pmid.startswith('S2_'):
                cursor.execute('SELECT 1 FROM papers WHERE pmid = ?', (pmid,))
                if cursor.fetchone():
                    return True

            # Check by DOI
            if doi:
                cursor.execute('SELECT 1 FROM papers WHERE doi = ?', (doi,))
                if cursor.fetchone():
                    return True

            # Check by title similarity (exact match)
            if title:
                cursor.execute('SELECT 1 FROM papers WHERE LOWER(TRIM(title)) = ?', (title,))
                if cursor.fetchone():
                    return True

            return False


# Convenience function for external use
def run_semantic_scholar_enrichment(limit: Optional[int] = None) -> Dict:
    """
    Run the complete Semantic Scholar enrichment pipeline.

    Args:
        limit: Maximum number of papers to enrich (None = all)

    Returns:
        Dictionary with enrichment results
    """
    enricher = SemanticScholarEnricher()
    return enricher.run_full_enrichment_pipeline(enrich_limit=limit)