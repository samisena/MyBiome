"""
Semantic Scholar enrichment module for paper collection pipeline.
Adds influence scores, citation counts, and AI summaries (TL;DR).
Note: Similar paper discovery functionality has been disabled.
"""

import json
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict

from back_end.src.data.config import setup_logging
from back_end.src.data.api_clients import get_semantic_scholar_client
from back_end.src.data_collection.database_manager import database_manager

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
        DEPRECATED: Similar paper discovery has been disabled.
        This method now returns empty stats without performing any API calls.

        Args:
            seed_papers: List of PMIDs to use as seeds (ignored)
            limit_per_paper: Max similar papers to find per seed paper (ignored)

        Returns:
            EnrichmentStats with empty results
        """
        logger.warning("discover_similar_papers() called but similar paper discovery is disabled")
        stats = EnrichmentStats()
        stats.errors.append("Similar paper discovery is disabled")
        return stats

    def run_full_enrichment_pipeline(self, enrich_limit: Optional[int] = None) -> Dict:
        """
        Run the S2 enrichment pipeline (metrics only).
        Only enriches existing papers with S2 data - no similar paper discovery.

        Args:
            enrich_limit: Limit for paper enrichment (None = all)

        Returns:
            Results from enrichment step
        """
        logger.info("Starting Semantic Scholar enrichment pipeline (metrics only)")
        pipeline_start = time.time()

        # Enrich existing papers with S2 metrics
        logger.info("=== Enriching existing papers with S2 metrics ===")
        enrichment_stats = self.enrich_existing_papers(limit=enrich_limit)

        # Combine results
        total_time = time.time() - pipeline_start

        results = {
            'enrichment': {
                'total_papers': enrichment_stats.total_papers,
                'enriched_papers': enrichment_stats.enriched_papers,
                'failed_papers': enrichment_stats.failed_papers,
                'errors': enrichment_stats.errors
            },
            'pipeline': {
                'total_time_seconds': round(total_time, 2),
                'status': 'success' if not enrichment_stats.errors else 'partial_success'
            }
        }

        logger.info(f"S2 enrichment pipeline completed in {total_time:.1f}s")
        logger.info(f"Results: {enrichment_stats.enriched_papers} papers enriched with S2 metrics")

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