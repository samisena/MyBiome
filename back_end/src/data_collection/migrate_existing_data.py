"""
Migration script to reprocess existing papers through the new semantic merger system.
This will eliminate existing duplicates and add semantic fields to existing interventions.
"""

import sys
import os
import shutil
import sqlite3
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
from collections import defaultdict
import json
import time

from back_end.src.data_collection.database_manager import DatabaseManager
from back_end.src.llm_processing.semantic_merger import SemanticMerger, InterventionExtraction
from back_end.src.data.config import setup_logging

logger = setup_logging(__name__, 'migration.log')

class DataMigration:
    """
    Handles migration of existing intervention data to the new semantic system.
    """

    def __init__(self, batch_size: int = 10, dry_run: bool = False):
        """
        Initialize migration system.

        Args:
            batch_size: Number of papers to process in each batch
            dry_run: If True, don't actually modify the database
        """
        self.batch_size = batch_size
        self.dry_run = dry_run
        self.db_manager = DatabaseManager()

        # Initialize semantic merger (without LLM calls for existing data)
        self.semantic_merger = SemanticMerger()

        # Migration statistics
        self.stats = {
            'papers_processed': 0,
            'interventions_before': 0,
            'interventions_after': 0,
            'duplicates_merged': 0,
            'semantic_groups_created': 0,
            'errors': 0
        }

        # Database paths
        self.db_path = Path("data/processed/intervention_research.db")
        self.backup_path = Path(f"data/processed/intervention_research_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db")

    def create_backup(self) -> bool:
        """Create a backup of the current database."""
        try:
            if self.db_path.exists():
                shutil.copy2(self.db_path, self.backup_path)
                logger.info(f"Database backup created: {self.backup_path}")
                return True
            else:
                logger.error(f"Database not found: {self.db_path}")
                return False
        except Exception as e:
            logger.error(f"Failed to create backup: {e}")
            return False

    def get_existing_papers_with_interventions(self) -> List[Dict]:
        """Get all papers that have interventions in the database."""
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()

                # Get papers with their interventions
                cursor.execute('''
                    SELECT DISTINCT p.pmid, p.title, p.abstract,
                           COUNT(i.id) as intervention_count
                    FROM papers p
                    INNER JOIN interventions i ON p.pmid = i.paper_id
                    WHERE i.semantic_group_id IS NULL OR i.semantic_group_id = ''
                    GROUP BY p.pmid, p.title, p.abstract
                    ORDER BY intervention_count DESC
                ''')

                papers = []
                for row in cursor.fetchall():
                    papers.append({
                        'pmid': row[0],
                        'title': row[1],
                        'abstract': row[2],
                        'intervention_count': row[3]
                    })

                logger.info(f"Found {len(papers)} papers with interventions needing migration")
                return papers

        except Exception as e:
            logger.error(f"Error getting papers: {e}")
            return []

    def get_interventions_for_paper(self, pmid: str) -> List[Dict]:
        """Get all interventions for a specific paper."""
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()

                cursor.execute('''
                    SELECT id, intervention_name, intervention_category, health_condition,
                           correlation_type, confidence_score, correlation_strength,
                           supporting_quote, extraction_model, consensus_confidence,
                           model_agreement, models_used, raw_extraction_count,
                           models_contributing
                    FROM interventions
                    WHERE paper_id = ?
                    AND (semantic_group_id IS NULL OR semantic_group_id = '')
                ''', (pmid,))

                interventions = []
                for row in cursor.fetchall():
                    interventions.append({
                        'id': row[0],
                        'intervention_name': row[1],
                        'intervention_category': row[2],
                        'health_condition': row[3],
                        'correlation_type': row[4],
                        'confidence_score': row[5] or 0.5,
                        'correlation_strength': row[6] or 0.5,
                        'supporting_quote': row[7] or '',
                        'extraction_model': row[8] or 'unknown',
                        'consensus_confidence': row[9],
                        'model_agreement': row[10],
                        'models_used': row[11],
                        'raw_extraction_count': row[12] or 1,
                        'models_contributing': row[13]
                    })

                return interventions

        except Exception as e:
            logger.error(f"Error getting interventions for paper {pmid}: {e}")
            return []

    def create_semantic_groups(self, interventions: List[Dict]) -> List[Dict]:
        """
        Create semantic groups from existing interventions without LLM calls.
        Uses rule-based matching for common duplicates.
        """
        if not interventions:
            return []

        # Convert to extraction objects
        extractions = []
        for intervention in interventions:
            extraction = InterventionExtraction(
                model_name=intervention.get('extraction_model', 'unknown'),
                intervention_name=intervention.get('intervention_name', ''),
                health_condition=intervention.get('health_condition', ''),
                intervention_category=intervention.get('intervention_category', ''),
                correlation_type=intervention.get('correlation_type', ''),
                confidence_score=intervention.get('confidence_score', 0.5),
                correlation_strength=intervention.get('correlation_strength', 0.5),
                supporting_quote=intervention.get('supporting_quote', ''),
                raw_data=intervention
            )
            extractions.append(extraction)

        # Group similar interventions using rule-based approach
        semantic_groups = []
        processed_indices = set()

        for i, extraction1 in enumerate(extractions):
            if i in processed_indices:
                continue

            # Find all similar interventions for this one
            similar_extractions = [extraction1]
            matched_indices = {i}

            for j, extraction2 in enumerate(extractions[i+1:], start=i+1):
                if j in processed_indices:
                    continue

                # Use rule-based duplicate detection
                if self._are_rule_based_duplicates(extraction1, extraction2):
                    similar_extractions.append(extraction2)
                    matched_indices.add(j)
                    logger.debug(f"Rule-based match: {extraction1.intervention_name} + {extraction2.intervention_name}")

            # Create merged intervention
            if len(similar_extractions) > 1:
                # Multiple extractions to merge
                merged = self._create_rule_based_merge(similar_extractions)
                self.stats['duplicates_merged'] += len(similar_extractions) - 1
            else:
                # Single extraction, just add semantic fields
                merged = self._enhance_single_intervention(extraction1)

            semantic_groups.append(merged)
            processed_indices.update(matched_indices)

        self.stats['semantic_groups_created'] += len(semantic_groups)
        return semantic_groups

    def _are_rule_based_duplicates(self, extract1: InterventionExtraction, extract2: InterventionExtraction) -> bool:
        """
        Rule-based duplicate detection without LLM calls.
        Handles common medical intervention variations.
        """
        # Must be same condition
        if not self.semantic_merger._are_same_condition(extract1, extract2):
            return False

        # Must be same category
        if extract1.intervention_category.lower() != extract2.intervention_category.lower():
            return False

        name1 = extract1.intervention_name.lower().strip()
        name2 = extract2.intervention_name.lower().strip()

        # Exact match
        if name1 == name2:
            return True

        # Common medication abbreviations
        medication_patterns = [
            (['proton pump inhibitors', 'ppi', 'ppis'], 'ppi_group'),
            (['anti-reflux mucosal ablation', 'arma'], 'arma_group'),
            (['irritable bowel syndrome', 'ibs'], 'ibs_group'),
            (['probiotics', 'probiotic therapy', 'probiotic treatment'], 'probiotics_group'),
            (['dietary modification', 'diet modification', 'dietary changes'], 'diet_group'),
            (['fundoplication', 'nissen procedure', 'anti-reflux surgery'], 'surgery_group'),
            (['h2 blockers', 'h2 receptor blockers', 'histamine-2 blockers'], 'h2_group'),
            (['calcium channel blockers', 'ccbs'], 'ccb_group'),
            (['ace inhibitors', 'acei'], 'ace_group'),
            (['angiotensin receptor blockers', 'arbs'], 'arb_group')
        ]

        # Check for pattern matches
        for patterns, group_name in medication_patterns:
            if any(pattern in name1 for pattern in patterns) and any(pattern in name2 for pattern in patterns):
                return True

        # Substring matching for similar names (conservative)
        if len(name1) > 5 and len(name2) > 5:
            # Check if one is contained in the other
            if name1 in name2 or name2 in name1:
                return True

            # Check for common word overlap
            words1 = set(name1.split())
            words2 = set(name2.split())
            common_words = words1.intersection(words2)

            # If they share most significant words, likely duplicates
            if len(common_words) >= 2 and len(common_words) >= min(len(words1), len(words2)) * 0.6:
                return True

        return False

    def _create_rule_based_merge(self, extractions: List[InterventionExtraction]) -> Dict:
        """Create a merged intervention from multiple extractions using rules."""
        # Use the highest confidence extraction as base
        base_extraction = max(extractions, key=lambda x: x.confidence_score)

        # Collect all intervention names as alternatives
        all_names = list(set(e.intervention_name for e in extractions))
        canonical_name = self._select_canonical_name(all_names)
        alternative_names = [name for name in all_names if name != canonical_name]

        # Create search terms
        search_terms = []
        for name in all_names:
            search_terms.extend(name.lower().split())
            # Add common abbreviations
            if 'proton pump inhibitors' in name.lower():
                search_terms.extend(['ppi', 'ppis'])
            elif 'irritable bowel syndrome' in name.lower():
                search_terms.extend(['ibs'])

        search_terms = list(set(search_terms))

        # Generate semantic group ID
        group_id = self.semantic_merger._generate_semantic_group_id(canonical_name, base_extraction.health_condition)

        # Calculate consensus values
        avg_confidence = sum(e.confidence_score for e in extractions) / len(extractions)
        contributing_models = list(set(e.model_name for e in extractions))

        # Create merged intervention
        merged = base_extraction.raw_data.copy()
        merged.update({
            'canonical_name': canonical_name,
            'alternative_names': alternative_names,
            'search_terms': search_terms,
            'semantic_group_id': group_id,
            'semantic_confidence': 0.85,  # High confidence for rule-based matching
            'merge_source': 'rule_based_migration',
            'consensus_confidence': avg_confidence,
            'model_agreement': 'migrated',
            'models_used': ','.join(contributing_models),
            'raw_extraction_count': len(extractions),
            'merge_decision_log': {
                'timestamp': datetime.now().isoformat(),
                'decision_method': 'rule_based_migration',
                'reasoning': f'Merged {len(extractions)} similar interventions during data migration',
                'extractions_merged': len(extractions),
                'original_names': all_names
            },
            'validator_agreement': True,
            'needs_human_review': False
        })

        return merged

    def _enhance_single_intervention(self, extraction: InterventionExtraction) -> Dict:
        """Add semantic fields to a single intervention."""
        result = extraction.raw_data.copy()

        # Add semantic fields
        result.update({
            'canonical_name': extraction.intervention_name,
            'alternative_names': [],
            'search_terms': [extraction.intervention_name.lower()],
            'semantic_group_id': self.semantic_merger._generate_semantic_group_id(
                extraction.intervention_name, extraction.health_condition
            ),
            'semantic_confidence': 1.0,
            'merge_source': 'single_migration',
            'merge_decision_log': {
                'timestamp': datetime.now().isoformat(),
                'decision_method': 'single_migration',
                'reasoning': 'Single intervention, no duplicates found',
                'extractions_merged': 1
            },
            'validator_agreement': True,
            'needs_human_review': False
        })

        return result

    def _select_canonical_name(self, names: List[str]) -> str:
        """Select the best canonical name from a list of alternatives."""
        # Prefer longer, more descriptive names
        # Prefer names that are not abbreviations
        names_with_scores = []

        for name in names:
            score = len(name)  # Longer names get higher scores

            # Penalize obvious abbreviations
            if len(name) <= 4 and name.isupper():
                score -= 10

            # Bonus for common medical terms
            bonus_terms = ['inhibitors', 'blockers', 'therapy', 'treatment', 'modification', 'surgery']
            for term in bonus_terms:
                if term in name.lower():
                    score += 5

            names_with_scores.append((name, score))

        # Return the name with the highest score
        return max(names_with_scores, key=lambda x: x[1])[0]

    def update_interventions_in_database(self, paper_pmid: str, merged_interventions: List[Dict]) -> bool:
        """Update interventions in the database."""
        if self.dry_run:
            logger.info(f"DRY RUN: Would update {len(merged_interventions)} interventions for paper {paper_pmid}")
            return True

        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()

                # First, delete existing interventions for this paper
                cursor.execute('DELETE FROM interventions WHERE paper_id = ?', (paper_pmid,))
                deleted_count = cursor.rowcount

                # Insert the merged interventions
                for intervention in merged_interventions:
                    # Ensure paper_id is set
                    intervention['paper_id'] = paper_pmid

                    # Insert the intervention
                    success = self.db_manager.insert_intervention(intervention)
                    if not success:
                        logger.error(f"Failed to insert intervention: {intervention.get('intervention_name')}")
                        return False

                conn.commit()
                logger.info(f"Updated paper {paper_pmid}: {deleted_count} → {len(merged_interventions)} interventions")
                return True

        except Exception as e:
            logger.error(f"Error updating interventions for paper {paper_pmid}: {e}")
            return False

    def migrate_paper_batch(self, papers: List[Dict]) -> Dict:
        """Migrate a batch of papers."""
        batch_stats = {'processed': 0, 'errors': 0, 'interventions_before': 0, 'interventions_after': 0}

        for paper in papers:
            try:
                pmid = paper['pmid']
                logger.info(f"Processing paper {pmid} ({paper['intervention_count']} interventions)")

                # Get existing interventions
                existing_interventions = self.get_interventions_for_paper(pmid)
                batch_stats['interventions_before'] += len(existing_interventions)

                if not existing_interventions:
                    continue

                # Create semantic groups
                merged_interventions = self.create_semantic_groups(existing_interventions)
                batch_stats['interventions_after'] += len(merged_interventions)

                # Update in database
                if self.update_interventions_in_database(pmid, merged_interventions):
                    batch_stats['processed'] += 1
                else:
                    batch_stats['errors'] += 1

            except Exception as e:
                logger.error(f"Error processing paper {paper.get('pmid', 'unknown')}: {e}")
                batch_stats['errors'] += 1

        return batch_stats

    def run_migration(self, max_papers: Optional[int] = None) -> Dict:
        """Run the complete migration process."""
        logger.info("=" * 50)
        logger.info("STARTING DATA MIGRATION")
        logger.info("=" * 50)

        start_time = time.time()

        # Create backup
        if not self.create_backup():
            logger.error("Failed to create backup. Aborting migration.")
            return {'success': False, 'error': 'Backup failed'}

        # Get papers to migrate
        papers = self.get_existing_papers_with_interventions()
        if not papers:
            logger.info("No papers found that need migration")
            return {'success': True, 'stats': self.stats}

        if max_papers:
            papers = papers[:max_papers]
            logger.info(f"Limited to {max_papers} papers for testing")

        # Process in batches
        total_batches = (len(papers) + self.batch_size - 1) // self.batch_size

        for batch_idx in range(total_batches):
            start_idx = batch_idx * self.batch_size
            end_idx = min(start_idx + self.batch_size, len(papers))
            batch = papers[start_idx:end_idx]

            logger.info(f"Processing batch {batch_idx + 1}/{total_batches} ({len(batch)} papers)")

            batch_stats = self.migrate_paper_batch(batch)

            # Update overall stats
            self.stats['papers_processed'] += batch_stats['processed']
            self.stats['interventions_before'] += batch_stats['interventions_before']
            self.stats['interventions_after'] += batch_stats['interventions_after']
            self.stats['errors'] += batch_stats['errors']

            logger.info(f"Batch {batch_idx + 1} completed: {batch_stats}")

        # Final statistics
        duration = time.time() - start_time

        logger.info("=" * 50)
        logger.info("MIGRATION COMPLETED")
        logger.info("=" * 50)
        logger.info(f"Duration: {duration:.2f} seconds")
        logger.info(f"Papers processed: {self.stats['papers_processed']}")
        logger.info(f"Interventions before: {self.stats['interventions_before']}")
        logger.info(f"Interventions after: {self.stats['interventions_after']}")
        logger.info(f"Duplicates merged: {self.stats['duplicates_merged']}")
        logger.info(f"Semantic groups created: {self.stats['semantic_groups_created']}")
        logger.info(f"Errors: {self.stats['errors']}")

        if self.stats['interventions_before'] > 0:
            reduction_pct = ((self.stats['interventions_before'] - self.stats['interventions_after']) /
                           self.stats['interventions_before'] * 100)
            logger.info(f"Data reduction: {reduction_pct:.1f}%")

        return {'success': True, 'stats': self.stats, 'backup_path': str(self.backup_path)}


def main():
    """Main migration function."""
    import argparse

    parser = argparse.ArgumentParser(description='Migrate existing data to semantic system')
    parser.add_argument('--dry-run', action='store_true', help='Run without making changes')
    parser.add_argument('--batch-size', type=int, default=10, help='Batch size for processing')
    parser.add_argument('--max-papers', type=int, help='Maximum papers to process (for testing)')

    args = parser.parse_args()

    # Initialize migration
    migration = DataMigration(batch_size=args.batch_size, dry_run=args.dry_run)

    if args.dry_run:
        print("[DRY RUN] No changes will be made to the database")

    # Run migration
    result = migration.run_migration(max_papers=args.max_papers)

    if result['success']:
        print("\n[SUCCESS] Migration completed successfully!")
        if 'backup_path' in result:
            print(f"[BACKUP] Database backup saved to: {result['backup_path']}")
    else:
        print(f"\n[ERROR] Migration failed: {result.get('error')}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())