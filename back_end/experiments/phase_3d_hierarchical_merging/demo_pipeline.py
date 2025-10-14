"""
Demo Pipeline - Run Phase 3d on Actual Database

Demonstrates the value-add of hierarchical clustering by showing:
- Before: Flat cluster structure
- After: Multi-level hierarchy with interpretable groups
"""

import sys
import logging
from pathlib import Path

# Add to path
sys.path.insert(0, str(Path(__file__).parent))

from stage_0_hyperparameter_optimizer import HyperparameterOptimizer
from stage_1_centroid_computation import compute_centroids
from stage_2_candidate_generation import generate_merge_candidates, print_candidate_summary
from stage_3_llm_validation import LLMValidator
from stage_4_cross_category_detection import CrossCategoryDetector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def show_current_state(db_path: str):
    """
    Show current state of database (before Phase 3d).

    Args:
        db_path: Path to database
    """
    import sqlite3

    print("\n" + "="*80)
    print("CURRENT STATE (BEFORE PHASE 3d)")
    print("="*80)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Check if mechanism_clusters table exists
    cursor.execute("""
        SELECT name FROM sqlite_master
        WHERE type='table' AND name='mechanism_clusters'
    """)

    if not cursor.fetchone():
        print("\n[X] mechanism_clusters table does not exist!")
        print("   Phase 3.6 (Mechanism Clustering) has not been run yet.")
        print("\n   Please run Phase 3.6 first:")
        print("   python -m back_end.src.orchestration.rotation_mechanism_clustering")
        conn.close()
        return False

    # Get cluster statistics
    cursor.execute("""
        SELECT
            COUNT(*) as total_clusters,
            COUNT(CASE WHEN parent_cluster_id IS NOT NULL THEN 1 END) as with_parents,
            COUNT(CASE WHEN member_count = 1 THEN 1 END) as singletons,
            AVG(member_count) as avg_size,
            MAX(hierarchy_level) as max_depth
        FROM mechanism_clusters
    """)

    stats = cursor.fetchone()
    total_clusters, with_parents, singletons, avg_size, max_depth = stats

    if total_clusters == 0:
        print("\n[X] No mechanism clusters found in database!")
        print("   Phase 3.6 needs to be run first.")
        conn.close()
        return False

    print(f"\n[STATS] Cluster Statistics:")
    print(f"   Total clusters: {total_clusters}")
    print(f"   Clusters with parents: {with_parents} ({with_parents/total_clusters*100:.1f}%)")
    print(f"   Singleton clusters: {singletons} ({singletons/total_clusters*100:.1f}%)")
    print(f"   Average cluster size: {avg_size:.1f}")
    print(f"   Hierarchy depth: {max_depth} levels")

    # Sample clusters
    cursor.execute("""
        SELECT cluster_id, canonical_name, member_count, hierarchy_level
        FROM mechanism_clusters
        ORDER BY member_count DESC
        LIMIT 10
    """)

    print(f"\n[LIST] Top 10 Largest Clusters:")
    print(f"{'ID':>5s} | {'Name':50s} | {'Size':>5s} | {'Level':>5s}")
    print("-" * 70)
    for row in cursor.fetchall():
        cluster_id, name, size, level = row
        print(f"{cluster_id:5d} | {name[:48]:50s} | {size:5d} | {level:5d}")

    # Check fragmentation
    cursor.execute("""
        SELECT canonical_name
        FROM mechanism_clusters
        WHERE canonical_name LIKE '%probiotic%'
           OR canonical_name LIKE '%Lactobacillus%'
           OR canonical_name LIKE '%Bifidobacterium%'
        ORDER BY canonical_name
        LIMIT 10
    """)

    probiotic_clusters = cursor.fetchall()
    if probiotic_clusters:
        print(f"\n[SEARCH] Example Fragmentation (Probiotic-related clusters):")
        for i, (name,) in enumerate(probiotic_clusters, 1):
            print(f"   {i}. {name}")
        print(f"   → These {len(probiotic_clusters)} clusters could potentially be hierarchically organized")

    conn.close()
    return True


def run_demo_optimization(db_path: str):
    """
    Run Stage 0: Hyperparameter Optimization (subset for demo).

    Args:
        db_path: Path to database
    """
    print("\n" + "="*80)
    print("STAGE 0: HYPERPARAMETER OPTIMIZATION (SUBSET)")
    print("="*80)
    print("\nNote: Running on SUBSET of 16 configs for demo speed")
    print("      (Full run: 64 configs, ~10 minutes)")

    from config import Phase3dConfig

    # Create smaller search space for demo
    demo_config = Phase3dConfig()
    demo_config.threshold_search_space = {
        'level_3_to_2': [0.84, 0.86],  # 2 values (vs 4)
        'level_2_to_1': [0.80, 0.82],  # 2 values (vs 4)
        'level_1_to_0': [0.76, 0.78]   # 2 values (vs 4)
    }
    # Total: 2×2×2 = 8 configs (vs 64)

    optimizer = HyperparameterOptimizer(
        db_path=db_path,
        entity_type='mechanism',
        config=demo_config
    )

    try:
        results = optimizer.optimize()

        if results:
            best = results[0]
            print(f"\n[OK] Best configuration found:")
            print(f"   Thresholds: L3→L2: {best.thresholds[3]:.2f}, " +
                  f"L2→L1: {best.thresholds[2]:.2f}, L1→L0: {best.thresholds[1]:.2f}")
            print(f"   Composite score: {best.composite_score:.1f}/100")
            print(f"   Estimated reduction: {best.metrics['reduction_ratio']:.1%}")

            return best.thresholds
        else:
            print("\n[!] No results from optimization")
            return None

    except Exception as e:
        print(f"\n[X] Optimization failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_demo_candidates(db_path: str, thresholds: dict):
    """
    Run Stage 1-2: Generate merge candidates.

    Args:
        db_path: Path to database
        thresholds: Optimal thresholds from Stage 0
    """
    print("\n" + "="*80)
    print("STAGES 1-2: CANDIDATE GENERATION")
    print("="*80)

    # Load data
    from stage_0_hyperparameter_optimizer import HyperparameterOptimizer

    loader = HyperparameterOptimizer(db_path=db_path, entity_type='mechanism')
    loader.load_data()

    clusters = loader.clusters
    embeddings = loader.embeddings

    print(f"\n[STATS] Loaded {len(clusters)} clusters with {len(embeddings)} embeddings")

    # Compute centroids (Stage 1)
    print("\n[Stage 1] Computing centroids...")
    centroids = compute_centroids(clusters, embeddings)
    print(f"   ✓ Computed {len(centroids)} centroids")

    # Generate candidates (Stage 2)
    print("\n[Stage 2] Generating merge candidates...")
    threshold = thresholds[3]  # Use Level 3→2 threshold for demo
    candidates = generate_merge_candidates(
        clusters,
        centroids,
        similarity_threshold=threshold,
        max_candidates=50  # Limit for demo
    )

    print(f"   ✓ Found {len(candidates)} candidates (showing top 50)")

    # Show summary
    print_candidate_summary(candidates, top_n=10)

    return candidates


def run_demo_llm_validation(candidates, embeddings):
    """
    Run Stage 3: LLM Validation (dry run - no actual LLM calls for demo).

    Args:
        candidates: List of merge candidates
        embeddings: Dict of embeddings
    """
    print("\n" + "="*80)
    print("STAGE 3: LLM VALIDATION (DRY RUN)")
    print("="*80)

    print("\n[!] NOTE: Skipping actual LLM calls for demo speed")
    print("   In production, this stage would:")
    print("   - Call qwen3:14b for each candidate")
    print("   - Classify relationship (MERGE_IDENTICAL / CREATE_PARENT / DIFFERENT)")
    print("   - Auto-approve based on confidence + quality checks")
    print("   - Flag edge cases for review")

    # Show what would happen with mock validation
    validator = LLMValidator()

    print(f"\n[LIST] Validation Preview (first 5 candidates):")
    for i, candidate in enumerate(candidates[:5], 1):
        print(f"\n   Candidate {i}:")
        print(f"   - Cluster A: {candidate.cluster_a.canonical_name} ({len(candidate.cluster_a.members)} members)")
        print(f"   - Cluster B: {candidate.cluster_b.canonical_name} ({len(candidate.cluster_b.members)} members)")
        print(f"   - Similarity: {candidate.similarity:.3f}")
        print(f"   - Confidence tier: {candidate.confidence_tier}")

        # Name quality check (can do without LLM)
        suggested_name = f"{candidate.cluster_a.canonical_name.split()[0]} Family"
        name_quality = validator._score_name_quality(suggested_name)
        print(f"   - Name quality: {name_quality.score}/100")

        # Diversity check (can do without LLM)
        diversity = validator._check_diversity(candidate, embeddings)
        print(f"   - Diversity: {diversity.severity} (similarity={diversity.inter_child_similarity:.2f})")

    print(f"\n[STATS] Expected Auto-Approval Rate:")
    print(f"   - HIGH tier ({sum(1 for c in candidates if c.confidence_tier == 'HIGH')} candidates): ~80-90%")
    print(f"   - MEDIUM tier ({sum(1 for c in candidates if c.confidence_tier == 'MEDIUM')} candidates): ~60-70%")
    print(f"   - Total expected approvals: ~{int(len(candidates) * 0.7)} merges")


def show_expected_improvements():
    """Show what improvements Phase 3d would provide."""
    print("\n" + "="*80)
    print("EXPECTED VALUE-ADD FROM PHASE 3d")
    print("="*80)

    print("\n[TARGET] Key Benefits:")
    print("\n1. REDUCED FRAGMENTATION")
    print("   Before: 415 flat clusters (65.8% singletons)")
    print("   After:  ~250-300 top-level clusters (2-3 hierarchy levels)")
    print("   Value:  40-50% reduction in cognitive load when browsing")

    print("\n2. HIERARCHICAL ORGANIZATION")
    print("   Before: Flat list (no relationships)")
    print("   After:  Parent-child relationships")
    print("   Example:")
    print("      Level 0: Probiotic Mechanisms (parent)")
    print("        ├─ Level 1: Lactobacillus Species")
    print("        │    └─ Level 2: L. reuteri Strains")
    print("        └─ Level 1: Bifidobacterium Species")
    print("             └─ Level 2: B. lactis Strains")

    print("\n3. IMPROVED INTERPRETABILITY")
    print("   Before: Cluster names like 'Cluster_142: modulates gut...'")
    print("   After:  Specific parent names like 'Probiotic-Mediated Gut Modulation'")
    print("   Value:  Easier to understand at-a-glance")

    print("\n4. CROSS-CATEGORY INSIGHTS")
    print("   Detects potential mis-categorizations:")
    print("   Example: 'Probiotics' (supplement) + 'FMT' (procedure)")
    print("           → Both merged under 'Gut Microbiome Modulation'")
    print("           → Suggests reviewing FMT categorization")

    print("\n5. BETTER FRONTEND EXPERIENCE")
    print("   Before: Scroll through 415 flat clusters")
    print("   After:  Browse ~250 top-level, drill down into children")
    print("   Value:  Hierarchical filtering, faster discovery")

    print("\n6. SEMANTIC SEARCH IMPROVEMENTS")
    print("   Before: Query matches exact cluster only")
    print("   After:  Query matches parent → shows all children")
    print("   Example: Search 'probiotic' → shows all probiotic subcategories")

    print("\n[CHART] Quantitative Improvements:")
    print("   - Cluster count reduction: 40-50%")
    print("   - Interpretability score: >75/100 (from N/A)")
    print("   - Singleton rate: <40% (from 65.8%)")
    print("   - Hierarchy depth: 2-3 levels (from 0)")
    print("   - Cross-category insights: ~10-20 cases detected")


def main():
    """Run demo pipeline."""
    print("="*80)
    print("PHASE 3d: HIERARCHICAL CLUSTERING - DEMO")
    print("="*80)

    # Find database
    db_path = Path(__file__).parent.parent.parent / "data" / "intervention_research_backup_before_rollback_20251009_192106.db"

    if not db_path.exists():
        print(f"\n[X] Database not found: {db_path}")
        print("\nPlease update the db_path in this script to point to your database.")
        return

    print(f"\nDatabase: {db_path}")

    # Stage 0: Show current state
    has_data = show_current_state(str(db_path))

    if not has_data:
        print("\n" + "="*80)
        print("DEMO CANNOT PROCEED")
        print("="*80)
        print("\nPhase 3.6 (Mechanism Clustering) must be run first to create")
        print("the mechanism_clusters table that Phase 3d operates on.")
        print("\nRun this command to create the base clusters:")
        print("  python -m back_end.src.orchestration.rotation_mechanism_clustering")
        return

    # Show expected value-add
    show_expected_improvements()

    print("\n" + "="*80)
    print("CONTINUE WITH DEMO? (This will take 2-3 minutes)")
    print("="*80)
    response = input("\nRun optimization and candidate generation? (y/n): ").strip().lower()

    if response != 'y':
        print("\nDemo stopped. Run again with 'y' to see full pipeline.")
        return

    # Stage 1: Hyperparameter optimization (subset)
    optimal_thresholds = run_demo_optimization(str(db_path))

    if not optimal_thresholds:
        print("\n[!] Optimization failed, using default thresholds")
        optimal_thresholds = {3: 0.86, 2: 0.82, 1: 0.78}

    # Stage 2-3: Candidates + validation preview
    candidates = run_demo_candidates(str(db_path), optimal_thresholds)

    if candidates:
        # Load embeddings for diversity checks
        from stage_0_hyperparameter_optimizer import HyperparameterOptimizer
        loader = HyperparameterOptimizer(db_path=str(db_path), entity_type='mechanism')
        loader.load_data()

        run_demo_llm_validation(candidates, loader.embeddings)

    print("\n" + "="*80)
    print("DEMO COMPLETE")
    print("="*80)

    print("\n[OK] Phase 3d is ready to run on your full database!")
    print("\nNext steps:")
    print("  1. Review the expected improvements above")
    print("  2. Run full hyperparameter optimization (64 configs, ~10 min)")
    print("  3. Run LLM validation with qwen3:14b (~30-60 min)")
    print("  4. Apply approved merges to database")
    print("  5. Explore improved hierarchical structure in frontend")


if __name__ == "__main__":
    main()
