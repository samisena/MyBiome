"""
PapersDAO - Data access object for paper operations.

Handles all CRUD operations for research papers.
"""

import json
from typing import List, Dict, Optional
from back_end.src.data.config import setup_logging
from back_end.src.data.validators import validation_manager
from .base_dao import BaseDAO

logger = setup_logging(__name__, 'database.log')


class PapersDAO(BaseDAO):
    """Data Access Object for paper operations."""

    def insert_paper(self, paper: Dict) -> bool:
        """
        Insert a paper with validation.

        Args:
            paper: Paper dictionary with metadata

        Returns:
            True if paper was newly inserted, False if already exists
        """
        try:
            # Validate paper data
            validation_result = validation_manager.validate_paper(paper)
            if not validation_result.is_valid:
                error_messages = [issue.message for issue in validation_result.errors]
                raise ValueError(f"Paper validation failed: {'; '.join(error_messages)}")
            validated_paper = validation_result.cleaned_data

            with self.get_connection() as conn:
                cursor = conn.cursor()

                cursor.execute('''
                    INSERT OR IGNORE INTO papers
                    (pmid, title, abstract, journal, publication_date, doi, pmc_id,
                     keywords, has_fulltext, fulltext_source, fulltext_path, processing_status, discovery_source)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    validated_paper['pmid'],
                    validated_paper['title'],
                    validated_paper['abstract'],
                    validated_paper['journal'],
                    validated_paper['publication_date'],
                    validated_paper['doi'],
                    validated_paper['pmc_id'],
                    json.dumps(validated_paper['keywords']) if validated_paper['keywords'] else None,
                    validated_paper['has_fulltext'],
                    validated_paper['fulltext_source'],
                    validated_paper['fulltext_path'],
                    'pending',
                    validated_paper.get('discovery_source', 'pubmed')
                ))

                was_new = cursor.rowcount > 0

                if was_new:
                    # Paper successfully inserted - perform cleanup
                    self._cleanup_paper_files(validated_paper['pmid'])

                return was_new

        except ValueError as e:
            logger.error(f"Validation error for paper {paper.get('pmid', 'unknown')}: {e}")
            return False
        except Exception as e:
            logger.error(f"Error inserting paper {paper.get('pmid', 'unknown')}: {e}")
            return False

    def _cleanup_paper_files(self, pmid: str):
        """Clean up temporary files for a successfully processed paper."""
        try:
            from back_end.src.utils.batch_file_operations import cleanup_xml_files_for_papers
            cleanup_xml_files_for_papers([pmid])
        except Exception as e:
            logger.error(f"Error during cleanup for paper {pmid}: {e}")

    def insert_papers_batch(self, papers: List[Dict]) -> tuple[int, int]:
        """
        Insert multiple papers efficiently using executemany().

        Args:
            papers: List of paper dictionaries

        Returns:
            Tuple of (inserted_count, failed_count)
        """
        if not papers:
            return 0, 0

        inserted_count = 0
        failed_count = 0

        # Validate all papers first
        validated_papers = []
        for paper in papers:
            try:
                validation_result = validation_manager.validate_paper(paper)
                if not validation_result.is_valid:
                    logger.warning(f"Paper {paper.get('pmid', 'unknown')} validation failed")
                    failed_count += 1
                    continue
                validated_papers.append(validation_result.cleaned_data)
            except Exception as e:
                logger.error(f"Error validating paper {paper.get('pmid', 'unknown')}: {e}")
                failed_count += 1

        if not validated_papers:
            return 0, failed_count

        # Batch insert using executemany (5x faster)
        try:
            # Prepare data tuples for executemany
            data_tuples = [
                (
                    p['pmid'],
                    p['title'],
                    p['abstract'],
                    p['journal'],
                    p['publication_date'],
                    p['doi'],
                    p['pmc_id'],
                    json.dumps(p['keywords']) if p['keywords'] else None,
                    p['has_fulltext'],
                    p['fulltext_source'],
                    p['fulltext_path'],
                    'pending',
                    p.get('discovery_source', 'pubmed')
                )
                for p in validated_papers
            ]

            # Execute batch insert
            rows_affected = self.execute_batch('''
                INSERT OR IGNORE INTO papers
                (pmid, title, abstract, journal, publication_date, doi, pmc_id,
                 keywords, has_fulltext, fulltext_source, fulltext_path, processing_status, discovery_source)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', data_tuples)

            inserted_count = len(validated_papers)

        except Exception as e:
            logger.error(f"Batch insert failed: {e}")
            failed_count += len(validated_papers)
            inserted_count = 0

        # Flush any pending file operations after batch completion
        try:
            from back_end.src.utils.batch_file_operations import flush_pending_operations
            flush_pending_operations()
        except Exception as e:
            logger.error(f"Error flushing batch file operations: {e}")

        return inserted_count, failed_count

    def get_paper_by_pmid(self, pmid: str) -> Optional[Dict]:
        """Get a paper by PMID."""
        return self.execute_single('SELECT * FROM papers WHERE pmid = ?', (pmid,))

    def get_all_papers(self, limit: Optional[int] = None,
                      processing_status: Optional[str] = None) -> List[Dict]:
        """Get all papers with optional filtering."""
        query = 'SELECT * FROM papers'
        params = []

        if processing_status:
            query += ' WHERE processing_status = ?'
            params.append(processing_status)

        query += ' ORDER BY publication_date DESC'

        if limit:
            query += ' LIMIT ?'
            params.append(limit)

        return self.execute_query(query, tuple(params))

    def get_papers_for_processing(self, extraction_model: str,
                                  limit: Optional[int] = None) -> List[Dict]:
        """
        Get papers that need LLM processing (intervention extraction).

        Papers are prioritized by influence score, then citation count, then publication date.
        """
        query = '''
            SELECT *
            FROM papers
            WHERE llm_processed = FALSE
              AND abstract IS NOT NULL
              AND abstract != ''
              AND (processing_status IS NULL OR processing_status != 'failed')
            ORDER BY
                COALESCE(influence_score, 0) DESC,
                COALESCE(citation_count, 0) DESC,
                publication_date DESC
        '''

        params = []
        if limit:
            query += ' LIMIT ?'
            params.append(limit)

        return self.execute_query(query, tuple(params))

    def mark_paper_llm_processed(self, pmid: str) -> bool:
        """Mark a paper as LLM processed."""
        try:
            rows_affected = self.execute_update('''
                UPDATE papers
                SET llm_processed = TRUE,
                    processing_status = 'processed'
                WHERE pmid = ?
            ''', (pmid,))
            return rows_affected > 0
        except Exception as e:
            logger.error(f"Error marking paper {pmid} as LLM processed: {e}")
            return False

    def update_paper_processing_status(self, pmid: str, status: str) -> bool:
        """Update paper processing status."""
        valid_statuses = ['pending', 'processing', 'processed', 'failed', 'needs_review']
        if status not in valid_statuses:
            raise ValueError(f"Invalid status: {status}")

        rows_affected = self.execute_update('''
            UPDATE papers
            SET processing_status = ?, updated_at = CURRENT_TIMESTAMP
            WHERE pmid = ?
        ''', (status, pmid))

        return rows_affected > 0

    def get_papers_by_condition(self, condition: str, limit: Optional[int] = None) -> List[Dict]:
        """Get papers related to a specific health condition."""
        query = '''
            SELECT * FROM papers
            WHERE (LOWER(title) LIKE LOWER(?)
                   OR LOWER(abstract) LIKE LOWER(?)
                   OR LOWER(keywords) LIKE LOWER(?))
              AND abstract IS NOT NULL
              AND abstract != ''
            ORDER BY publication_date DESC
        '''

        condition_pattern = f"%{condition}%"
        params = [condition_pattern, condition_pattern, condition_pattern]

        if limit:
            query += ' LIMIT ?'
            params.append(limit)

        return self.execute_query(query, tuple(params))
