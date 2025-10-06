"""
Evaluator for Hierarchical Semantic Normalization

Tests the automated system against ground truth labeling (50 pairs).
Calculates accuracy metrics, confusion matrix, and provides error analysis.
"""

import os
import json
import logging
from typing import Dict, List, Tuple
from collections import defaultdict, Counter
import numpy as np

# Import local modules
from embedding_engine import EmbeddingEngine
from llm_classifier import LLMClassifier

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Evaluator:
    """
    Evaluate automated normalization against ground truth.
    """

    def __init__(
        self,
        ground_truth_path: str,
        embedding_engine: EmbeddingEngine,
        llm_classifier: LLMClassifier
    ):
        """
        Initialize the evaluator.

        Args:
            ground_truth_path: Path to ground truth JSON file
            embedding_engine: Embedding engine instance
            llm_classifier: LLM classifier instance
        """
        self.ground_truth_path = ground_truth_path
        self.embedding_engine = embedding_engine
        self.llm_classifier = llm_classifier

        # Load ground truth
        self.ground_truth = self._load_ground_truth()

        # Evaluation results
        self.predictions = []
        self.errors = []

        logger.info(f"Evaluator initialized with {len(self.ground_truth)} ground truth pairs")

    def _load_ground_truth(self) -> List[Dict]:
        """Load ground truth data from JSON file."""
        with open(self.ground_truth_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        return data['labeled_pairs']

    def evaluate_pair(self, gt_pair: Dict) -> Dict:
        """
        Evaluate a single intervention pair.

        Args:
            gt_pair: Ground truth pair dict

        Returns:
            Evaluation result dict
        """
        intervention_1 = gt_pair['intervention_1']
        intervention_2 = gt_pair['intervention_2']

        # Generate embeddings
        emb1 = self.embedding_engine.generate_embedding(intervention_1)
        emb2 = self.embedding_engine.generate_embedding(intervention_2)

        # Calculate similarity
        similarity = self.embedding_engine.cosine_similarity(emb1, emb2)

        # Classify relationship
        prediction = self.llm_classifier.classify_relationship(
            intervention_1,
            intervention_2,
            similarity
        )

        # Extract ground truth
        gt_relationship = gt_pair['relationship']
        gt_type = gt_relationship['type_code']
        gt_canonical = gt_relationship['hierarchy'].get('layer_1_canonical')
        gt_same_variant = gt_relationship['hierarchy'].get('same_variant_layer_2', False)

        # Compare predictions
        result = {
            'pair_id': gt_pair['pair_id'],
            'intervention_1': intervention_1,
            'intervention_2': intervention_2,
            'gt_similarity': gt_pair.get('similarity_score', 0.0),
            'pred_similarity': similarity,
            'gt_relationship_type': gt_type,
            'pred_relationship_type': prediction['relationship_type'],
            'gt_canonical': gt_canonical,
            'pred_canonical': prediction.get('layer_1_canonical'),
            'gt_same_variant': gt_same_variant,
            'pred_same_variant': prediction.get('layer_2_same_variant', False),
            'relationship_type_correct': gt_type == prediction['relationship_type'],
            'canonical_correct': gt_canonical == prediction.get('layer_1_canonical'),
            'same_variant_correct': gt_same_variant == prediction.get('layer_2_same_variant', False)
        }

        return result

    def run_evaluation(self) -> Dict:
        """
        Run evaluation on all ground truth pairs.

        Returns:
            Evaluation metrics dict
        """
        logger.info("Starting evaluation...")

        for gt_pair in self.ground_truth:
            try:
                result = self.evaluate_pair(gt_pair)
                self.predictions.append(result)

                # Track errors
                if not result['relationship_type_correct']:
                    self.errors.append({
                        'pair_id': result['pair_id'],
                        'intervention_1': result['intervention_1'],
                        'intervention_2': result['intervention_2'],
                        'error_type': 'relationship_type',
                        'expected': result['gt_relationship_type'],
                        'predicted': result['pred_relationship_type']
                    })

            except Exception as e:
                logger.error(f"Error evaluating pair {gt_pair['pair_id']}: {e}")

        # Calculate metrics
        metrics = self._calculate_metrics()

        return metrics

    def _calculate_metrics(self) -> Dict:
        """Calculate evaluation metrics."""
        if not self.predictions:
            return {}

        total = len(self.predictions)

        # Relationship type accuracy
        relationship_correct = sum(1 for p in self.predictions if p['relationship_type_correct'])
        relationship_accuracy = relationship_correct / total

        # Canonical accuracy (excluding DIFFERENT which has no canonical)
        canonical_pairs = [p for p in self.predictions if p['gt_canonical'] is not None]
        canonical_correct = sum(1 for p in canonical_pairs if p['canonical_correct'])
        canonical_accuracy = canonical_correct / len(canonical_pairs) if canonical_pairs else 0.0

        # Same variant accuracy
        same_variant_correct = sum(1 for p in self.predictions if p['same_variant_correct'])
        same_variant_accuracy = same_variant_correct / total

        # Overall accuracy (all three must be correct)
        overall_correct = sum(
            1 for p in self.predictions
            if p['relationship_type_correct'] and
               (p['gt_canonical'] is None or p['canonical_correct']) and
               p['same_variant_correct']
        )
        overall_accuracy = overall_correct / total

        # Confusion matrix for relationship types
        confusion_matrix = self._build_confusion_matrix()

        # Per-class metrics
        per_class_metrics = self._calculate_per_class_metrics()

        metrics = {
            'total_pairs': total,
            'relationship_type_accuracy': relationship_accuracy,
            'canonical_accuracy': canonical_accuracy,
            'same_variant_accuracy': same_variant_accuracy,
            'overall_accuracy': overall_accuracy,
            'confusion_matrix': confusion_matrix,
            'per_class_metrics': per_class_metrics,
            'errors': self.errors
        }

        return metrics

    def _build_confusion_matrix(self) -> Dict:
        """Build confusion matrix for relationship types."""
        relationship_types = ['EXACT_MATCH', 'VARIANT', 'SUBTYPE', 'SAME_CATEGORY', 'DOSAGE_VARIANT', 'DIFFERENT']

        matrix = defaultdict(lambda: defaultdict(int))

        for pred in self.predictions:
            gt_type = pred['gt_relationship_type']
            pred_type = pred['pred_relationship_type']
            matrix[gt_type][pred_type] += 1

        # Convert to regular dict
        return {gt: dict(preds) for gt, preds in matrix.items()}

    def _calculate_per_class_metrics(self) -> Dict:
        """Calculate precision, recall, F1 per relationship type."""
        relationship_types = set(p['gt_relationship_type'] for p in self.predictions)

        per_class = {}

        for rel_type in relationship_types:
            # True positives
            tp = sum(
                1 for p in self.predictions
                if p['gt_relationship_type'] == rel_type and p['pred_relationship_type'] == rel_type
            )

            # False positives
            fp = sum(
                1 for p in self.predictions
                if p['gt_relationship_type'] != rel_type and p['pred_relationship_type'] == rel_type
            )

            # False negatives
            fn = sum(
                1 for p in self.predictions
                if p['gt_relationship_type'] == rel_type and p['pred_relationship_type'] != rel_type
            )

            # Calculate metrics
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

            per_class[rel_type] = {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'support': tp + fn  # Total ground truth instances
            }

        return per_class

    def print_report(self, metrics: Dict):
        """Print evaluation report."""
        print("\n" + "="*80)
        print("EVALUATION REPORT")
        print("="*80)

        print(f"\nTotal pairs evaluated: {metrics['total_pairs']}")

        print("\n--- ACCURACY METRICS ---")
        print(f"Relationship Type Accuracy: {metrics['relationship_type_accuracy']:.2%}")
        print(f"Canonical Group Accuracy: {metrics['canonical_accuracy']:.2%}")
        print(f"Same Variant Accuracy: {metrics['same_variant_accuracy']:.2%}")
        print(f"Overall Accuracy (all correct): {metrics['overall_accuracy']:.2%}")

        print("\n--- CONFUSION MATRIX ---")
        print("Rows: Ground Truth | Columns: Predicted\n")

        # Get all relationship types
        all_types = set()
        for gt_preds in metrics['confusion_matrix'].values():
            all_types.update(gt_preds.keys())
        all_types = sorted(list(all_types))

        # Print header
        print(f"{'GT \\ Pred':<20}", end="")
        for pred_type in all_types:
            print(f"{pred_type[:15]:<17}", end="")
        print()

        # Print matrix
        for gt_type in sorted(metrics['confusion_matrix'].keys()):
            print(f"{gt_type:<20}", end="")
            for pred_type in all_types:
                count = metrics['confusion_matrix'][gt_type].get(pred_type, 0)
                print(f"{count:<17}", end="")
            print()

        print("\n--- PER-CLASS METRICS ---")
        print(f"{'Type':<20} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Support':<10}")
        print("-" * 66)

        for rel_type, metrics_data in sorted(metrics['per_class_metrics'].items()):
            print(
                f"{rel_type:<20} "
                f"{metrics_data['precision']:<12.2%} "
                f"{metrics_data['recall']:<12.2%} "
                f"{metrics_data['f1']:<12.2%} "
                f"{metrics_data['support']:<10}"
            )

        # Print errors
        if metrics['errors']:
            print(f"\n--- ERRORS ({len(metrics['errors'])}) ---")
            for i, error in enumerate(metrics['errors'][:10], 1):  # Show first 10
                print(f"\n{i}. Pair {error['pair_id']}: {error['intervention_1']} vs {error['intervention_2']}")
                print(f"   Expected: {error['expected']}, Predicted: {error['predicted']}")

            if len(metrics['errors']) > 10:
                print(f"\n... and {len(metrics['errors']) - 10} more errors")

        print("\n" + "="*80)

    def save_results(self, metrics: Dict, output_path: str):
        """Save evaluation results to JSON."""
        results = {
            'ground_truth_path': self.ground_truth_path,
            'total_pairs': metrics['total_pairs'],
            'metrics': {
                'relationship_type_accuracy': metrics['relationship_type_accuracy'],
                'canonical_accuracy': metrics['canonical_accuracy'],
                'same_variant_accuracy': metrics['same_variant_accuracy'],
                'overall_accuracy': metrics['overall_accuracy']
            },
            'confusion_matrix': metrics['confusion_matrix'],
            'per_class_metrics': metrics['per_class_metrics'],
            'predictions': self.predictions,
            'errors': metrics['errors']
        }

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        logger.info(f"Evaluation results saved to: {output_path}")


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate Hierarchical Normalization System")
    parser.add_argument(
        '--ground-truth',
        default='c:/Users/samis/Desktop/MyBiome/back_end/experiments/semantic_normalization/data/ground_truth/labeling_session_hierarchical_ground_truth_20251005_184757.json',
        help='Ground truth JSON path'
    )
    parser.add_argument(
        '--config',
        default='c:/Users/samis/Desktop/MyBiome/back_end/experiments/semantic_normalization/config/config_phase2.yaml',
        help='Config YAML path'
    )
    parser.add_argument(
        '--output',
        default='c:/Users/samis/Desktop/MyBiome/back_end/experiments/semantic_normalization/results/evaluation_results.json',
        help='Output results JSON path'
    )

    args = parser.parse_args()

    # Initialize components
    print("Initializing embedding engine...")
    embedding_engine = EmbeddingEngine(
        cache_path="c:/Users/samis/Desktop/MyBiome/back_end/experiments/semantic_normalization/data/cache/embeddings.pkl"
    )

    print("Initializing LLM classifier...")
    llm_classifier = LLMClassifier(
        canonical_cache_path="c:/Users/samis/Desktop/MyBiome/back_end/experiments/semantic_normalization/data/cache/canonicals.pkl",
        relationship_cache_path="c:/Users/samis/Desktop/MyBiome/back_end/experiments/semantic_normalization/data/cache/llm_decisions.pkl"
    )

    # Create evaluator
    evaluator = Evaluator(
        ground_truth_path=args.ground_truth,
        embedding_engine=embedding_engine,
        llm_classifier=llm_classifier
    )

    # Run evaluation
    print("\nRunning evaluation...")
    metrics = evaluator.run_evaluation()

    # Print report
    evaluator.print_report(metrics)

    # Save results
    evaluator.save_results(metrics, args.output)

    print(f"\nEvaluation complete. Results saved to: {args.output}")


if __name__ == "__main__":
    main()
