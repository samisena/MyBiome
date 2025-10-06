"""
Smart Pair Generator
Generates candidate intervention pairs using fuzzy matching for ground truth labeling.
Enhanced with similarity-based, random, and targeted sampling strategies.
"""

import json
import logging
import random
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Set, Optional
import yaml
from itertools import combinations
from collections import defaultdict


class SmartPairGenerator:
    """Generate candidate pairs for labeling using fuzzy matching with strategic sampling."""

    def __init__(self, config_path: str = None):
        """Initialize pair generator with configuration."""
        if config_path is None:
            config_path = Path(__file__).parent.parent / "config" / "config.yaml"

        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.similarity_min = self.config['labeling']['similarity_threshold_min']
        self.similarity_max = self.config['labeling']['similarity_threshold_max']
        self.candidate_pool_size = self.config['labeling']['candidate_pool_size']

        # Setup logging
        logging.basicConfig(
            level=self.config['logging']['level'],
            format=self.config['logging']['format']
        )
        self.logger = logging.getLogger(__name__)

        # Import fuzzy matching library
        try:
            from rapidfuzz import fuzz
            self.fuzz = fuzz
            self.logger.info("Using rapidfuzz for fuzzy matching")
        except ImportError:
            try:
                from fuzzywuzzy import fuzz
                self.fuzz = fuzz
                self.logger.info("Using fuzzywuzzy for fuzzy matching")
            except ImportError:
                raise ImportError("Please install rapidfuzz or fuzzywuzzy: pip install rapidfuzz")

        # Cache for intervention categories (if available)
        self.intervention_categories = {}

    def calculate_similarity(self, str1: str, str2: str) -> float:
        """Calculate similarity score between two strings."""
        algorithm = self.config['fuzzy_matching']['algorithm']

        if algorithm == "jaro_winkler":
            score = self.fuzz.ratio(str1.lower(), str2.lower())
        elif algorithm == "token_sort":
            score = self.fuzz.token_sort_ratio(str1.lower(), str2.lower())
        else:  # levenshtein / default
            score = self.fuzz.ratio(str1.lower(), str2.lower())

        return score / 100.0  # Normalize to 0-1

    def generate_candidates(self, intervention_names: List[str]) -> List[Dict]:
        """
        Generate candidate pairs with similarity scores.

        Args:
            intervention_names: List of unique intervention names

        Returns:
            List of candidate pairs with metadata
        """
        self.logger.info(f"Generating candidate pairs from {len(intervention_names)} interventions")

        candidates = []
        seen_pairs: Set[Tuple[str, str]] = set()

        # Generate all possible pairs
        total_combinations = len(intervention_names) * (len(intervention_names) - 1) // 2
        self.logger.info(f"Evaluating {total_combinations} possible combinations...")

        for i, name1 in enumerate(intervention_names):
            if i % 50 == 0:
                self.logger.info(f"Progress: {i}/{len(intervention_names)} interventions processed")

            for name2 in intervention_names[i+1:]:
                # Skip identical names
                if name1.lower() == name2.lower():
                    continue

                # Calculate similarity
                similarity = self.calculate_similarity(name1, name2)

                # Filter by similarity threshold
                if self.similarity_min <= similarity <= self.similarity_max:
                    # Ensure canonical ordering (alphabetical)
                    pair_tuple = tuple(sorted([name1, name2]))

                    if pair_tuple not in seen_pairs:
                        seen_pairs.add(pair_tuple)

                        candidates.append({
                            "intervention_1": pair_tuple[0],
                            "intervention_2": pair_tuple[1],
                            "similarity_score": round(similarity, 4),
                            "length_diff": abs(len(name1) - len(name2)),
                            "word_count_1": len(name1.split()),
                            "word_count_2": len(name2.split())
                        })

        # Sort by similarity score (descending) to get best candidates first
        candidates.sort(key=lambda x: x['similarity_score'], reverse=True)

        self.logger.info(f"Generated {len(candidates)} candidate pairs")
        return candidates[:self.candidate_pool_size]

    def categorize_candidates(self, candidates: List[Dict]) -> Dict[str, List[Dict]]:
        """
        Categorize candidates into likely matches and non-matches.

        High similarity (>0.75): Likely matches (synonyms, variants)
        Medium similarity (0.50-0.75): Edge cases
        Low similarity (0.40-0.50): Likely non-matches (confusing cases)
        """
        categorized = {
            "likely_match": [],
            "edge_case": [],
            "likely_no_match": []
        }

        for candidate in candidates:
            score = candidate['similarity_score']

            if score > 0.75:
                categorized["likely_match"].append(candidate)
            elif score >= 0.50:
                categorized["edge_case"].append(candidate)
            else:
                categorized["likely_no_match"].append(candidate)

        self.logger.info(f"Categorized candidates:")
        self.logger.info(f"  - Likely match: {len(categorized['likely_match'])}")
        self.logger.info(f"  - Edge cases: {len(categorized['edge_case'])}")
        self.logger.info(f"  - Likely no match: {len(categorized['likely_no_match'])}")

        return categorized

    def generate_stratified_candidates(
        self,
        intervention_names: List[str],
        intervention_metadata: Optional[Dict[str, Dict]] = None,
        target_count: int = 500
    ) -> List[Dict]:
        """
        Generate stratified candidate pairs using multiple sampling strategies.

        Strategy:
        - Similarity-based sampling (60%): 300 pairs across similarity ranges
        - Random sampling (20%): 100 pairs for DIFFERENT examples
        - Targeted sampling (20%): 100 pairs from same drug class/category

        Args:
            intervention_names: List of unique intervention names
            intervention_metadata: Optional dict mapping names to {category, ...}
            target_count: Total number of candidate pairs to generate

        Returns:
            List of candidate pairs with metadata
        """
        self.logger.info(f"Generating {target_count} stratified candidate pairs...")

        # First, generate ALL possible pairs with similarity scores
        all_pairs = self._generate_all_pairs_with_scores(intervention_names)

        # Sort by similarity for easier sampling
        all_pairs.sort(key=lambda x: x['similarity_score'], reverse=True)

        # Strategy 1: Similarity-based sampling (60% = 300 pairs)
        similarity_pairs = self._sample_by_similarity_ranges(
            all_pairs,
            {
                (0.85, 0.95): int(target_count * 0.20),  # 100 pairs
                (0.75, 0.85): int(target_count * 0.20),  # 100 pairs
                (0.65, 0.75): int(target_count * 0.20)   # 100 pairs
            }
        )

        # Strategy 2: Random sampling for low similarity (20% = 100 pairs)
        random_pairs = self._sample_random_low_similarity(
            all_pairs,
            count=int(target_count * 0.20),
            similarity_range=(0.40, 0.65)
        )

        # Strategy 3: Targeted sampling from same category (20% = 100 pairs)
        if intervention_metadata:
            targeted_pairs = self._sample_same_category_pairs(
                intervention_names,
                intervention_metadata,
                count=int(target_count * 0.20)
            )
        else:
            # Fallback: sample more random pairs
            targeted_pairs = self._sample_random_low_similarity(
                all_pairs,
                count=int(target_count * 0.20),
                similarity_range=(0.30, 0.60)
            )

        # Combine all strategies
        combined_pairs = similarity_pairs + random_pairs + targeted_pairs

        # Remove duplicates (keep first occurrence)
        seen_pairs = set()
        unique_pairs = []
        for pair in combined_pairs:
            pair_key = tuple(sorted([pair['intervention_1'], pair['intervention_2']]))
            if pair_key not in seen_pairs:
                seen_pairs.add(pair_key)
                unique_pairs.append(pair)

        self.logger.info(f"Generated {len(unique_pairs)} unique candidate pairs")
        self.logger.info(f"  - Similarity-based: {len(similarity_pairs)}")
        self.logger.info(f"  - Random sampling: {len(random_pairs)}")
        self.logger.info(f"  - Targeted sampling: {len(targeted_pairs)}")

        return unique_pairs[:target_count]

    def _generate_all_pairs_with_scores(self, intervention_names: List[str]) -> List[Dict]:
        """Generate all possible pairs with similarity scores."""
        pairs = []
        total_combinations = len(intervention_names) * (len(intervention_names) - 1) // 2

        self.logger.info(f"Computing similarity for {total_combinations} possible pairs...")

        for i, name1 in enumerate(intervention_names):
            if i % 50 == 0 and i > 0:
                self.logger.info(f"Progress: {i}/{len(intervention_names)} interventions processed")

            for name2 in intervention_names[i+1:]:
                if name1.lower() == name2.lower():
                    continue

                similarity = self.calculate_similarity(name1, name2)

                pairs.append({
                    "intervention_1": name1,
                    "intervention_2": name2,
                    "similarity_score": round(similarity, 4),
                    "length_diff": abs(len(name1) - len(name2)),
                    "word_count_1": len(name1.split()),
                    "word_count_2": len(name2.split())
                })

        return pairs

    def _sample_by_similarity_ranges(
        self,
        all_pairs: List[Dict],
        ranges: Dict[Tuple[float, float], int]
    ) -> List[Dict]:
        """Sample pairs from specific similarity ranges."""
        sampled = []

        for (min_sim, max_sim), count in ranges.items():
            # Filter pairs in range
            in_range = [p for p in all_pairs
                       if min_sim <= p['similarity_score'] < max_sim]

            # Sample requested count (or all if fewer available)
            sample_count = min(count, len(in_range))
            sampled.extend(random.sample(in_range, sample_count))

            self.logger.info(f"Sampled {sample_count} pairs from range [{min_sim}, {max_sim})")

        return sampled

    def _sample_random_low_similarity(
        self,
        all_pairs: List[Dict],
        count: int,
        similarity_range: Tuple[float, float]
    ) -> List[Dict]:
        """Sample random pairs from low similarity range."""
        min_sim, max_sim = similarity_range

        in_range = [p for p in all_pairs
                   if min_sim <= p['similarity_score'] < max_sim]

        sample_count = min(count, len(in_range))
        sampled = random.sample(in_range, sample_count)

        self.logger.info(f"Sampled {sample_count} random pairs from range [{min_sim}, {max_sim})")
        return sampled

    def _sample_same_category_pairs(
        self,
        intervention_names: List[str],
        intervention_metadata: Dict[str, Dict],
        count: int
    ) -> List[Dict]:
        """
        Sample pairs from same intervention category (e.g., different statins, different probiotics).

        Args:
            intervention_names: List of intervention names
            intervention_metadata: Dict mapping names to {category, ...}
            count: Number of pairs to sample

        Returns:
            List of candidate pairs from same categories
        """
        # Group interventions by category
        by_category = defaultdict(list)
        for name in intervention_names:
            if name in intervention_metadata:
                category = intervention_metadata[name].get('category', 'unknown')
                by_category[category].append(name)

        # Generate pairs within each category
        category_pairs = []
        for category, names in by_category.items():
            if len(names) < 2:
                continue

            # Generate all pairs within this category
            for i, name1 in enumerate(names):
                for name2 in names[i+1:]:
                    similarity = self.calculate_similarity(name1, name2)

                    category_pairs.append({
                        "intervention_1": name1,
                        "intervention_2": name2,
                        "similarity_score": round(similarity, 4),
                        "length_diff": abs(len(name1) - len(name2)),
                        "word_count_1": len(name1.split()),
                        "word_count_2": len(name2.split()),
                        "same_category": category
                    })

        # Sample requested count
        sample_count = min(count, len(category_pairs))
        sampled = random.sample(category_pairs, sample_count) if category_pairs else []

        self.logger.info(f"Sampled {sample_count} same-category pairs from {len(by_category)} categories")
        return sampled

    def save_candidates(self, candidates: List[Dict], output_path: Path = None):
        """Save candidate pairs to JSON file."""
        if output_path is None:
            output_dir = Path(self.config['paths']['ground_truth_dir'])
            output_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = output_dir / f"candidate_pairs_{timestamp}.json"

        categorized = self.categorize_candidates(candidates)

        output_data = {
            "metadata": {
                "generation_timestamp": datetime.now().isoformat(),
                "total_candidates": len(candidates),
                "similarity_range": [self.similarity_min, self.similarity_max],
                "algorithm": self.config['fuzzy_matching']['algorithm']
            },
            "categorized": categorized,
            "all_candidates": candidates
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        self.logger.info(f"Saved {len(candidates)} candidate pairs to {output_path}")
        return output_path


def main():
    """CLI entry point for candidate generation."""
    from .data_exporter import InterventionDataExporter

    # Load latest export
    exporter = InterventionDataExporter()
    export_data = exporter.get_latest_export()

    # Generate candidates
    generator = SmartPairGenerator()
    unique_names = export_data['unique_names']

    print(f"\nGenerating candidate pairs from {len(unique_names)} unique interventions...")
    candidates = generator.generate_candidates(unique_names)

    # Save candidates
    output_path = generator.save_candidates(candidates)

    print("\n" + "="*60)
    print("CANDIDATE PAIR GENERATION COMPLETE")
    print("="*60)
    print(f"Total candidates generated: {len(candidates)}")
    print(f"Similarity range: {generator.similarity_min} - {generator.similarity_max}")
    print(f"Saved to: {output_path}")
    print("="*60 + "\n")

    # Show sample
    print("Sample candidate pairs:")
    for i, candidate in enumerate(candidates[:10], 1):
        try:
            print(f"{i}. [{candidate['similarity_score']:.2f}] '{candidate['intervention_1']}' vs '{candidate['intervention_2']}'")
        except UnicodeEncodeError:
            # Handle special characters in Windows terminal
            print(f"{i}. [{candidate['similarity_score']:.2f}] (pair with special characters)")

    if len(candidates) > 10:
        print(f"... and {len(candidates) - 10} more")


if __name__ == "__main__":
    main()
