"""
Test Runner for Hierarchical Semantic Normalization

Runs the normalization system on sample interventions and generates:
- Timestamped results for comparison across runs
- Clustering analysis (canonical groups with members)
- Summary statistics (LLM usage, cache hits, relationship distribution)
- Progress tracking for iterative testing

Designed for multiple test iterations with threshold tuning.
"""

import os
import json
import sqlite3
import logging
from typing import List, Dict, Optional
from datetime import datetime
from pathlib import Path
from collections import defaultdict, Counter

# Import local modules
from embedding_engine import EmbeddingEngine
from llm_classifier import LLMClassifier
from hierarchy_manager import HierarchyManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TestRunner:
    """
    Run normalization tests with detailed analysis and timestamped results.
    """

    def __init__(
        self,
        source_db_path: str,
        config_path: Optional[str] = None,
        output_dir: Optional[str] = None
    ):
        """
        Initialize the test runner.

        Args:
            source_db_path: Path to source database (intervention_research.db)
            config_path: Path to YAML config file (optional)
            output_dir: Directory for test results (default: results/)
        """
        self.source_db_path = source_db_path
        self.config_path = config_path
        self.output_dir = output_dir or "c:/Users/samis/Desktop/MyBiome/back_end/experiments/semantic_normalization/results"

        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)

        # Initialize components
        self.embedding_engine = self._load_embedding_engine()
        self.llm_classifier = self._load_llm_classifier()

        # Test results
        self.results = []
        self.clusters = defaultdict(list)
        self.stats = {
            'total_interventions': 0,
            'unique_canonicals': 0,
            'llm_canonical_extractions': 0,
            'llm_relationship_classifications': 0,
            'cache_hits_canonical': 0,
            'cache_hits_relationship': 0,
            'relationship_type_counts': Counter()
        }

        # Timestamp for this run
        self.run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        logger.info("TestRunner initialized")

    def _load_embedding_engine(self) -> EmbeddingEngine:
        """Load embedding engine with config."""
        cache_path = "c:/Users/samis/Desktop/MyBiome/back_end/experiments/semantic_normalization/data/cache/embeddings.pkl"

        if self.config_path and os.path.exists(self.config_path):
            import yaml
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
                emb_config = config.get('embedding', {})
                cache_path = config.get('cache', {}).get('embedding_cache_path', cache_path)

                return EmbeddingEngine(
                    model=emb_config.get('model', 'nomic-embed-text'),
                    cache_path=cache_path,
                    batch_size=emb_config.get('batch_size', 32)
                )

        return EmbeddingEngine(cache_path=cache_path)

    def _load_llm_classifier(self) -> LLMClassifier:
        """Load LLM classifier with config."""
        canonical_cache = "c:/Users/samis/Desktop/MyBiome/back_end/experiments/semantic_normalization/data/cache/canonicals.pkl"
        relationship_cache = "c:/Users/samis/Desktop/MyBiome/back_end/experiments/semantic_normalization/data/cache/llm_decisions.pkl"

        if self.config_path and os.path.exists(self.config_path):
            import yaml
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
                llm_config = config.get('llm', {})
                cache_config = config.get('cache', {})

                return LLMClassifier(
                    model=llm_config.get('model', 'qwen3:14b'),
                    temperature=llm_config.get('temperature', 0.0),
                    timeout=llm_config.get('timeout', 60),
                    max_retries=llm_config.get('max_retries', 3),
                    strip_think_tags=llm_config.get('strip_think_tags', True),
                    canonical_cache_path=cache_config.get('canonical_cache_path', canonical_cache),
                    relationship_cache_path=cache_config.get('llm_cache_path', relationship_cache)
                )

        return LLMClassifier(
            canonical_cache_path=canonical_cache,
            relationship_cache_path=relationship_cache
        )

    def load_sample_interventions(self, limit: Optional[int] = None) -> List[str]:
        """
        Load intervention names from source database.

        Args:
            limit: Maximum number of interventions to load (optional)

        Returns:
            List of intervention names
        """
        conn = sqlite3.connect(self.source_db_path)
        cursor = conn.cursor()

        query = """
        SELECT DISTINCT intervention_name
        FROM interventions
        WHERE intervention_name IS NOT NULL
        ORDER BY intervention_name
        """

        if limit:
            query += f" LIMIT {limit}"

        cursor.execute(query)
        interventions = [row[0] for row in cursor.fetchall()]

        conn.close()

        logger.info(f"Loaded {len(interventions)} interventions from database")
        return interventions

    def process_intervention(self, intervention_name: str) -> Dict:
        """
        Process a single intervention through normalization.

        Args:
            intervention_name: Intervention name to process

        Returns:
            Result dict with canonical, reasoning, etc.
        """
        # Extract canonical
        canonical_result = self.llm_classifier.extract_canonical(intervention_name)

        result = {
            'original_name': intervention_name,
            'canonical_group': canonical_result['canonical_group'],
            'reasoning': canonical_result['reasoning'],
            'source': canonical_result['source']  # 'llm' or 'fallback'
        }

        # Update stats
        if canonical_result['source'] == 'llm':
            self.stats['llm_canonical_extractions'] += 1

        # Add to cluster
        self.clusters[canonical_result['canonical_group']].append(intervention_name)

        return result

    def find_relationships(
        self,
        intervention_names: List[str],
        top_k: int = 5
    ) -> List[Dict]:
        """
        Find and classify relationships between interventions.

        Args:
            intervention_names: List of all intervention names
            top_k: Number of similar interventions to find per intervention

        Returns:
            List of relationship dicts
        """
        relationships = []

        logger.info(f"Finding relationships for {len(intervention_names)} interventions...")

        for i, intervention in enumerate(intervention_names):
            if (i + 1) % 50 == 0:
                logger.info(f"Processing relationships: {i + 1}/{len(intervention_names)}")

            # Find similar interventions
            similar = self.embedding_engine.find_similar(
                query_text=intervention,
                candidate_texts=intervention_names,
                top_k=top_k,
                min_similarity=0.70
            )

            # Classify relationships
            for similar_name, similarity in similar:
                rel_result = self.llm_classifier.classify_relationship(
                    intervention,
                    similar_name,
                    similarity
                )

                relationships.append({
                    'intervention_1': intervention,
                    'intervention_2': similar_name,
                    'similarity': similarity,
                    'relationship_type': rel_result['relationship_type'],
                    'canonical': rel_result.get('layer_1_canonical'),
                    'same_variant': rel_result.get('layer_2_same_variant', False),
                    'source': rel_result['source']  # 'llm', 'auto_threshold', 'fallback'
                })

                # Update stats
                self.stats['relationship_type_counts'][rel_result['relationship_type']] += 1
                if rel_result['source'] == 'llm':
                    self.stats['llm_relationship_classifications'] += 1

        return relationships

    def run_test(
        self,
        limit: Optional[int] = None,
        find_relationships: bool = True,
        top_k_similar: int = 5
    ) -> Dict:
        """
        Run complete test workflow.

        Args:
            limit: Maximum interventions to process (optional)
            find_relationships: Whether to find and classify relationships
            top_k_similar: Number of similar interventions per intervention

        Returns:
            Complete test results dict
        """
        logger.info(f"Starting test run: {self.run_timestamp}")

        # Step 1: Load interventions
        intervention_names = self.load_sample_interventions(limit)
        self.stats['total_interventions'] = len(intervention_names)

        # Step 2: Process each intervention
        logger.info("Processing interventions...")
        for i, intervention_name in enumerate(intervention_names):
            if (i + 1) % 50 == 0:
                logger.info(f"Processed {i + 1}/{len(intervention_names)} interventions")

            result = self.process_intervention(intervention_name)
            self.results.append(result)

        # Step 3: Count unique canonicals
        self.stats['unique_canonicals'] = len(self.clusters)

        # Step 4: Find relationships (optional)
        relationships = []
        if find_relationships:
            relationships = self.find_relationships(intervention_names, top_k_similar)

        # Step 5: Get cache stats
        llm_stats = self.llm_classifier.get_stats()
        self.stats['cache_hits_canonical'] = llm_stats['canonical_cache_hits']
        self.stats['cache_hits_relationship'] = llm_stats['relationship_cache_hits']

        # Step 6: Build final results
        test_results = {
            'run_timestamp': self.run_timestamp,
            'config_path': self.config_path,
            'source_db': self.source_db_path,
            'stats': dict(self.stats),
            'stats_summary': {
                'total_interventions': self.stats['total_interventions'],
                'unique_canonicals': self.stats['unique_canonicals'],
                'llm_canonical_extractions': self.stats['llm_canonical_extractions'],
                'llm_relationship_classifications': self.stats['llm_relationship_classifications'],
                'canonical_cache_hit_rate': llm_stats['canonical_hit_rate'],
                'relationship_cache_hit_rate': llm_stats['relationship_hit_rate']
            },
            'interventions': self.results,
            'clusters': dict(self.clusters),
            'relationships': relationships,
            'relationship_type_distribution': dict(self.stats['relationship_type_counts'])
        }

        return test_results

    def save_results(self, results: Dict):
        """
        Save test results to timestamped JSON file.

        Args:
            results: Test results dict
        """
        output_file = os.path.join(
            self.output_dir,
            f"test_run_{self.run_timestamp}.json"
        )

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        logger.info(f"Test results saved to: {output_file}")

    def print_summary(self, results: Dict):
        """Print test summary to console."""
        print("\n" + "="*80)
        print(f"TEST RUN SUMMARY - {results['run_timestamp']}")
        print("="*80)

        stats = results['stats_summary']

        print(f"\n--- PROCESSING STATS ---")
        print(f"Total interventions: {stats['total_interventions']}")
        print(f"Unique canonical groups: {stats['unique_canonicals']}")
        print(f"Average cluster size: {stats['total_interventions'] / stats['unique_canonicals']:.2f}")

        print(f"\n--- LLM USAGE ---")
        print(f"Canonical extractions (LLM): {stats['llm_canonical_extractions']}")
        print(f"Relationship classifications (LLM): {stats['llm_relationship_classifications']}")
        print(f"Canonical cache hit rate: {stats['canonical_cache_hit_rate']:.2%}")
        print(f"Relationship cache hit rate: {stats['relationship_cache_hit_rate']:.2%}")

        # Relationship type distribution
        if results.get('relationship_type_distribution'):
            print(f"\n--- RELATIONSHIP TYPE DISTRIBUTION ---")
            for rel_type, count in sorted(results['relationship_type_distribution'].items()):
                print(f"{rel_type}: {count}")

        # Top clusters (canonical groups with most members)
        print(f"\n--- TOP 20 CANONICAL GROUPS (by member count) ---")
        clusters = results['clusters']
        sorted_clusters = sorted(clusters.items(), key=lambda x: len(x[1]), reverse=True)

        for i, (canonical, members) in enumerate(sorted_clusters[:20], 1):
            print(f"\n{i}. {canonical} ({len(members)} members)")
            # Show first 5 members
            for member in members[:5]:
                print(f"   - {member}")
            if len(members) > 5:
                print(f"   ... and {len(members) - 5} more")

        print("\n" + "="*80)

    def compare_runs(self, previous_run_path: str) -> Dict:
        """
        Compare current run with a previous run.

        Args:
            previous_run_path: Path to previous run JSON file

        Returns:
            Comparison dict
        """
        with open(previous_run_path, 'r', encoding='utf-8') as f:
            previous_run = json.load(f)

        # Build comparison
        comparison = {
            'previous_timestamp': previous_run['run_timestamp'],
            'current_timestamp': self.run_timestamp,
            'stats_diff': {
                'total_interventions': self.stats['total_interventions'] - previous_run['stats']['total_interventions'],
                'unique_canonicals': self.stats['unique_canonicals'] - previous_run['stats']['unique_canonicals']
            },
            'cluster_changes': []
        }

        # Find clusters that changed
        prev_clusters = previous_run['clusters']
        curr_clusters = self.clusters

        for canonical in set(list(prev_clusters.keys()) + list(curr_clusters.keys())):
            prev_members = set(prev_clusters.get(canonical, []))
            curr_members = set(curr_clusters.get(canonical, []))

            if prev_members != curr_members:
                comparison['cluster_changes'].append({
                    'canonical': canonical,
                    'added_members': list(curr_members - prev_members),
                    'removed_members': list(prev_members - curr_members)
                })

        return comparison


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Test Hierarchical Normalization System")
    parser.add_argument(
        '--source-db',
        default='c:/Users/samis/Desktop/MyBiome/back_end/data/processed/intervention_research.db',
        help='Source database path'
    )
    parser.add_argument(
        '--config',
        default='c:/Users/samis/Desktop/MyBiome/back_end/experiments/semantic_normalization/config/config_phase2.yaml',
        help='Config YAML path'
    )
    parser.add_argument(
        '--output-dir',
        default='c:/Users/samis/Desktop/MyBiome/back_end/experiments/semantic_normalization/results',
        help='Output directory for test results'
    )
    parser.add_argument(
        '--limit',
        type=int,
        help='Limit number of interventions to process (for quick tests)'
    )
    parser.add_argument(
        '--no-relationships',
        action='store_true',
        help='Skip relationship finding (faster)'
    )
    parser.add_argument(
        '--top-k',
        type=int,
        default=5,
        help='Number of similar interventions to find per intervention'
    )

    args = parser.parse_args()

    # Create test runner
    runner = TestRunner(
        source_db_path=args.source_db,
        config_path=args.config if os.path.exists(args.config) else None,
        output_dir=args.output_dir
    )

    # Run test
    results = runner.run_test(
        limit=args.limit,
        find_relationships=not args.no_relationships,
        top_k_similar=args.top_k
    )

    # Save results
    runner.save_results(results)

    # Print summary
    runner.print_summary(results)

    print(f"\nTest complete. Results saved to: {args.output_dir}/test_run_{runner.run_timestamp}.json")


if __name__ == "__main__":
    main()
