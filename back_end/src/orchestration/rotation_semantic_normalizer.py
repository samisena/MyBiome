"""
Semantic Normalization Orchestrator

Runs hierarchical semantic normalization for interventions after LLM processing.

Flow:
1. Load interventions for condition from database
2. Generate embeddings (cached)
3. Extract canonical groups via LLM (cached)
4. Find similar interventions and classify relationships
5. Populate semantic_hierarchy tables
6. Update canonical_groups aggregations

Features:
- Resumable (checks what's already normalized)
- Incremental (only processes new interventions)
- Batch-aware (processes in chunks of 50)
- Caching (embeddings + LLM decisions)

Usage:
    # Normalize single condition
    python -m back_end.src.orchestration.rotation_semantic_normalizer diabetes

    # Normalize all conditions
    python -m back_end.src.orchestration.rotation_semantic_normalizer --all

    # Resume from specific condition
    python -m back_end.src.orchestration.rotation_semantic_normalizer --resume

    # Check status
    python -m back_end.src.orchestration.rotation_semantic_normalizer --status
"""

import argparse
import json
import logging
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Set
from collections import defaultdict

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from back_end.src.data.config import config
from back_end.src.semantic_normalization import SemanticNormalizer
from back_end.src.semantic_normalization.config import (
    DB_PATH,
    RESULTS_DIR,
    CACHE_DIR
)

# Setup logging
logging.basicConfig(
    level=logging.INFO if not config.fast_mode else logging.ERROR,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SemanticNormalizationOrchestrator:
    """Orchestrate semantic normalization across conditions."""

    def __init__(self, db_path: str = None):
        """Initialize orchestrator."""
        self.db_path = db_path or str(DB_PATH)
        self.results_dir = RESULTS_DIR
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.session_file = None
        self.session_data = {}

        # Initialize normalizer
        self.normalizer = SemanticNormalizer(db_path=self.db_path)

        logger.info(f"Initialized SemanticNormalizationOrchestrator with DB: {self.db_path}")

    def load_or_create_session(self, session_file: Path = None, force: bool = False) -> Dict:
        """Load existing session or create new one."""
        if session_file is None:
            # Check for existing session
            session_files = list(self.results_dir.glob("semantic_norm_session_*.json"))
            if session_files:
                latest_session = max(session_files, key=lambda p: p.stat().st_mtime)

                print(f"\nFound existing session: {latest_session.name}")

                if force:
                    response = 'y'
                    print("Auto-resuming (force=True)")
                else:
                    response = input("Resume this session? (y/n): ").strip().lower()

                if response == 'y':
                    with open(latest_session, 'r', encoding='utf-8') as f:
                        self.session_data = json.load(f)

                    self.session_file = latest_session
                    logger.info(f"Resumed session from {latest_session}")
                    return self.session_data

        # Create new session
        session_id = f"semantic_norm_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.session_file = self.results_dir / f"{session_id}.json"

        self.session_data = {
            "session_id": session_id,
            "created_at": datetime.now().isoformat(),
            "conditions_processed": [],
            "conditions_pending": [],
            "total_interventions": 0,
            "canonical_groups": 0,
            "relationships": 0,
            "errors": []
        }

        logger.info(f"Created new session {session_id}")
        return self.session_data

    def save_session(self):
        """Save current session state."""
        if self.session_file:
            with open(self.session_file, 'w', encoding='utf-8') as f:
                json.dump(self.session_data, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved session to {self.session_file}")

    def get_conditions_from_db(self) -> List[str]:
        """Get list of unique health conditions from interventions table."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT DISTINCT health_condition
            FROM interventions
            WHERE health_condition IS NOT NULL
            AND health_condition != ''
            ORDER BY health_condition
        """)

        conditions = [row[0] for row in cursor.fetchall()]
        conn.close()

        return conditions

    def get_interventions_for_condition(self, condition: str) -> List[Dict]:
        """Get all interventions for a specific condition."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute("""
            SELECT DISTINCT
                intervention_name,
                health_condition,
                intervention_category,
                correlation_type,
                sample_size,
                study_type
            FROM interventions
            WHERE health_condition = ?
            ORDER BY intervention_name
        """, (condition,))

        interventions = [dict(row) for row in cursor.fetchall()]
        conn.close()

        return interventions

    def check_already_normalized(self, condition: str) -> int:
        """Check how many interventions for this condition are already normalized."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Check if semantic_hierarchy table exists
        cursor.execute("""
            SELECT COUNT(*)
            FROM sqlite_master
            WHERE type='table' AND name='semantic_hierarchy'
        """)

        if cursor.fetchone()[0] == 0:
            conn.close()
            return 0

        # Count normalized interventions for this condition
        cursor.execute("""
            SELECT COUNT(*)
            FROM semantic_hierarchy
            WHERE entity_type = 'intervention'
            AND source_table = 'interventions'
            AND entity_name IN (
                SELECT intervention_name
                FROM interventions
                WHERE health_condition = ?
            )
        """, (condition,))

        count = cursor.fetchone()[0]
        conn.close()

        return count

    def normalize_condition(self, condition: str, batch_size: int = 50, force: bool = False) -> Dict:
        """
        Run semantic normalization for a single condition.

        Args:
            condition: Health condition name
            batch_size: Number of interventions to process per batch
            force: Skip confirmation prompts

        Returns:
            Dictionary with normalization results
        """
        try:
            print(f"\n{'='*80}")
            print(f"NORMALIZING: {condition}")
            print(f"{'='*80}")
        except UnicodeEncodeError:
            safe_condition = condition.encode('ascii', errors='replace').decode('ascii')
            print(f"\n{'='*80}")
            print(f"NORMALIZING: {safe_condition}")
            print(f"{'='*80}")

        # Load interventions
        interventions = self.get_interventions_for_condition(condition)
        print(f"Found {len(interventions)} interventions")

        if not interventions:
            print("No interventions found, skipping")
            return {"condition": condition, "interventions": 0, "error": "No interventions"}

        # Check if already normalized
        already_normalized = self.check_already_normalized(condition)
        if already_normalized > 0 and not force:
            print(f"[WARNING]  {already_normalized} interventions already normalized")
            response = input("Re-normalize anyway? (y/n): ").strip().lower()
            if response != 'y':
                print("Skipping...")
                return {"condition": condition, "interventions": len(interventions), "skipped": True}

        # Run normalization
        try:
            print(f"\nRunning normalization (batch size: {batch_size})...")
            results = self.normalizer.normalize_interventions(
                interventions=[i['intervention_name'] for i in interventions],
                entity_type='intervention',
                source_table='interventions',
                batch_size=batch_size
            )

            print(f"\n[OK] Normalization complete!")
            print(f"  - Processed: {results['total_processed']}")
            print(f"  - Canonical groups: {results['canonical_groups_created']}")
            print(f"  - Relationships: {results['relationships_created']}")

            return {
                "condition": condition,
                "interventions": len(interventions),
                "processed": results['total_processed'],
                "canonical_groups": results['canonical_groups_created'],
                "relationships": results['relationships_created'],
                "completed_at": datetime.now().isoformat()
            }

        except Exception as e:
            safe_condition = condition.encode('ascii', errors='replace').decode('ascii')
            logger.error(f"Error normalizing {safe_condition}: {e}", exc_info=True)
            print(f"\n[ERROR] Error: {e}")

            return {
                "condition": condition,
                "interventions": len(interventions),
                "error": str(e),
                "failed_at": datetime.now().isoformat()
            }

    def normalize_all_conditions(self, batch_size: int = 50, force: bool = False):
        """Normalize all conditions in database."""
        # Load or create session
        self.load_or_create_session(force=force)

        # Get all conditions
        all_conditions = self.get_conditions_from_db()
        print(f"\nFound {len(all_conditions)} unique conditions in database")

        # Determine which conditions to process
        processed_set = set(self.session_data.get('conditions_processed', []))
        pending_conditions = [c for c in all_conditions if c not in processed_set]

        if not pending_conditions:
            print("All conditions already processed!")
            return

        print(f"Pending: {len(pending_conditions)} conditions")
        print(f"Already processed: {len(processed_set)} conditions")

        # Process each condition
        for i, condition in enumerate(pending_conditions, 1):
            try:
                print(f"\n[{i}/{len(pending_conditions)}] Processing: {condition}")
            except UnicodeEncodeError:
                # Handle Unicode characters that can't be printed to console
                safe_condition = condition.encode('ascii', errors='replace').decode('ascii')
                print(f"\n[{i}/{len(pending_conditions)}] Processing: {safe_condition}")

            result = self.normalize_condition(condition, batch_size=batch_size, force=force)

            # Update session
            if 'error' not in result:
                self.session_data['conditions_processed'].append(condition)
                self.session_data['total_interventions'] += result.get('processed', 0)
                self.session_data['canonical_groups'] += result.get('canonical_groups', 0)
                self.session_data['relationships'] += result.get('relationships', 0)
            else:
                self.session_data['errors'].append(result)

            # Save progress
            self.save_session()

        print(f"\n{'='*80}")
        print("NORMALIZATION COMPLETE")
        print(f"{'='*80}")
        print(f"Conditions processed: {len(self.session_data['conditions_processed'])}")
        print(f"Total interventions: {self.session_data['total_interventions']}")
        print(f"Canonical groups: {self.session_data['canonical_groups']}")
        print(f"Relationships: {self.session_data['relationships']}")
        if self.session_data['errors']:
            print(f"\n[WARNING]  Errors: {len(self.session_data['errors'])}")
        print(f"{'='*80}\n")

    def display_status(self):
        """Display normalization status."""
        print(f"\n{'='*80}")
        print("SEMANTIC NORMALIZATION STATUS")
        print(f"{'='*80}")

        # Check if tables exist
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT COUNT(*)
            FROM sqlite_master
            WHERE type='table' AND name='semantic_hierarchy'
        """)

        if cursor.fetchone()[0] == 0:
            print("\n[WARNING]  semantic_hierarchy table does not exist")
            print("Run migration first:")
            print("  python -m back_end.src.migrations.add_semantic_normalization_tables")
            print(f"{'='*80}\n")
            conn.close()
            return

        # Get counts
        cursor.execute("SELECT COUNT(*) FROM semantic_hierarchy WHERE entity_type = 'intervention'")
        intervention_count = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(DISTINCT layer_1_canonical) FROM semantic_hierarchy WHERE entity_type = 'intervention' AND layer_1_canonical IS NOT NULL")
        canonical_count = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM entity_relationships")
        relationship_count = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM canonical_groups WHERE entity_type = 'intervention'")
        group_count = cursor.fetchone()[0]

        conn.close()

        print(f"\nDatabase: {self.db_path}")
        print(f"\nSemantic Hierarchy:")
        print(f"  - Interventions normalized: {intervention_count}")
        print(f"  - Canonical groups (Layer 1): {canonical_count}")
        print(f"  - Relationships tracked: {relationship_count}")
        print(f"  - Canonical group records: {group_count}")

        # Check session
        session_files = list(self.results_dir.glob("semantic_norm_session_*.json"))
        if session_files:
            latest_session = max(session_files, key=lambda p: p.stat().st_mtime)
            with open(latest_session, 'r', encoding='utf-8') as f:
                session_data = json.load(f)

            print(f"\nLatest Session: {session_data['session_id']}")
            print(f"  - Created: {session_data.get('created_at', 'Unknown')}")
            print(f"  - Conditions processed: {len(session_data.get('conditions_processed', []))}")
            print(f"  - Errors: {len(session_data.get('errors', []))}")

        print(f"{'='*80}\n")


def main():
    """CLI entry point for semantic normalization orchestrator."""
    parser = argparse.ArgumentParser(
        description="Semantic normalization orchestrator for interventions"
    )

    parser.add_argument(
        'condition',
        nargs='?',
        help='Health condition to normalize (e.g., diabetes, hypertension)'
    )

    parser.add_argument(
        '--all',
        action='store_true',
        help='Normalize all conditions in database'
    )

    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume from latest session'
    )

    parser.add_argument(
        '--status',
        action='store_true',
        help='Show normalization status'
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=50,
        help='Batch size for processing (default: 50)'
    )

    parser.add_argument(
        '--db-path',
        type=str,
        default=None,
        help='Override database path'
    )

    parser.add_argument(
        '--force',
        action='store_true',
        help='Force re-normalization without prompting'
    )

    args = parser.parse_args()

    # Initialize orchestrator
    orchestrator = SemanticNormalizationOrchestrator(db_path=args.db_path)

    # Handle commands
    if args.status:
        orchestrator.display_status()

    elif args.all or args.resume:
        orchestrator.normalize_all_conditions(batch_size=args.batch_size, force=args.force)

    elif args.condition:
        result = orchestrator.normalize_condition(args.condition, batch_size=args.batch_size, force=args.force)
        orchestrator.save_session()

    else:
        parser.print_help()
        print("\nExamples:")
        print("  # Normalize single condition")
        print("  python -m back_end.src.orchestration.rotation_semantic_normalizer diabetes")
        print()
        print("  # Normalize all conditions")
        print("  python -m back_end.src.orchestration.rotation_semantic_normalizer --all")
        print()
        print("  # Check status")
        print("  python -m back_end.src.orchestration.rotation_semantic_normalizer --status")


if __name__ == "__main__":
    main()
