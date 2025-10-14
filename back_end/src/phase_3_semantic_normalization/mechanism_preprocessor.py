"""
Mechanism Text Preprocessor - Hierarchical Decomposition

Implements selective hierarchical decomposition for complex mechanism texts:
- Identifies complex mechanisms (> 80 chars, contains connectors)
- Uses LLM to decompose into parent + child mechanisms
- Extracts target organs/tissues
- Validates decomposition quality

This module supports the hybrid preprocessing approach:
1. Baseline clustering on raw texts
2. Selective decomposition for complex/low-coherence mechanisms
3. Re-clustering on enhanced dataset
4. Compare metrics and decide keep/revert

Architecture:
- Reuses LLMClassifier for decomposition
- Integrates with MechanismNormalizer for re-clustering
- Validation checkpoints with quick feedback loops
"""

import re
import json
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime

# Import LLM classifier
from .phase_3_llm_classifier import LLMClassifier

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class MechanismDecomposition:
    """Result of mechanism decomposition."""
    original_text: str
    is_complex: bool
    parent_mechanism: Optional[str] = None
    child_mechanisms: List[str] = field(default_factory=list)
    target_organ: Optional[str] = None
    reasoning: Optional[str] = None
    confidence: float = 0.0
    source: str = 'llm'  # 'llm' or 'fallback'


@dataclass
class PreprocessingResult:
    """Result of preprocessing operation."""
    total_mechanisms: int
    complex_count: int
    decomposed_count: int
    decomposition_success_rate: float
    avg_children_per_decomposition: float
    elapsed_time_seconds: float
    decompositions: List[MechanismDecomposition]


class MechanismPreprocessor:
    """
    Preprocessor for mechanism texts.

    Implements selective hierarchical decomposition strategy:
    - Identify complex mechanisms
    - Decompose using LLM
    - Validate decomposition quality
    """

    # Complexity indicators
    LENGTH_THRESHOLD = 80  # characters
    COMPLEXITY_PATTERNS = [
        r'\bvia\b',
        r'\bthrough\b',
        r'\bby\b',
        r',\s*(?=\S)',  # Commas (list separator)
        r'\band\b.*\band\b',  # Multiple "and" (multiple pathways)
        r'\bwhich\b',
        r'\bthat\b',
        r'\bresulting in\b',
        r'\bleading to\b',
    ]

    def __init__(
        self,
        llm_classifier: Optional[LLMClassifier] = None,
        cache_dir: Optional[str] = None,
        decomposition_threshold: float = 0.85  # Confidence threshold for accepting decomposition
    ):
        """
        Initialize preprocessor.

        Args:
            llm_classifier: Optional LLMClassifier instance (creates new if None)
            cache_dir: Cache directory for LLM decisions
            decomposition_threshold: Confidence threshold for accepting decomposition
        """
        if llm_classifier is None:
            if cache_dir:
                from pathlib import Path
                cache_path = Path(cache_dir) / "mechanism_decompositions.pkl"
                self.llm_classifier = LLMClassifier(
                    canonical_cache_path=str(cache_path)
                )
            else:
                from .config import CACHE_DIR
                cache_path = CACHE_DIR / "mechanism_decompositions.pkl"
                self.llm_classifier = LLMClassifier(
                    canonical_cache_path=str(cache_path)
                )
        else:
            self.llm_classifier = llm_classifier

        self.decomposition_threshold = decomposition_threshold

        logger.info("MechanismPreprocessor initialized")

    def is_complex(self, mechanism_text: str) -> Tuple[bool, List[str]]:
        """
        Check if mechanism is complex and needs decomposition.

        Args:
            mechanism_text: Mechanism text to check

        Returns:
            Tuple of (is_complex, reasons)
        """
        reasons = []

        # Check length
        if len(mechanism_text) > self.LENGTH_THRESHOLD:
            reasons.append(f"length>{self.LENGTH_THRESHOLD}")

        # Check complexity patterns
        for pattern in self.COMPLEXITY_PATTERNS:
            if re.search(pattern, mechanism_text, re.IGNORECASE):
                pattern_name = pattern.replace(r'\b', '').replace(r'\s*', '').replace('(?=\\S)', '')
                reasons.append(f"pattern:{pattern_name}")

        is_complex = len(reasons) > 0

        return is_complex, reasons

    def decompose_mechanism(
        self,
        mechanism_text: str,
        force: bool = False
    ) -> MechanismDecomposition:
        """
        Decompose complex mechanism using LLM.

        Args:
            mechanism_text: Mechanism to decompose
            force: Force decomposition even if not complex

        Returns:
            MechanismDecomposition result
        """
        # Check if complex
        is_complex, reasons = self.is_complex(mechanism_text)

        if not is_complex and not force:
            return MechanismDecomposition(
                original_text=mechanism_text,
                is_complex=False,
                source='skip'
            )

        # Generate decomposition prompt
        prompt = self._create_decomposition_prompt(mechanism_text)

        try:
            # Call LLM
            response = self.llm_classifier._call_llm(prompt)

            # Parse response
            result = self.llm_classifier._parse_json_response(response)

            # Validate response
            if 'parent_mechanism' not in result:
                raise ValueError("Missing parent_mechanism in response")

            # Extract fields
            parent = result.get('parent_mechanism', '')
            children = result.get('child_mechanisms', [])
            target_organ = result.get('target_organ')
            reasoning = result.get('reasoning', '')
            confidence = result.get('confidence', 0.8)

            # Ensure children is a list
            if isinstance(children, str):
                children = [children]

            return MechanismDecomposition(
                original_text=mechanism_text,
                is_complex=True,
                parent_mechanism=parent,
                child_mechanisms=children,
                target_organ=target_organ,
                reasoning=reasoning,
                confidence=confidence,
                source='llm'
            )

        except Exception as e:
            logger.error(f"Failed to decompose mechanism: {e}")

            # Fallback: no decomposition
            return MechanismDecomposition(
                original_text=mechanism_text,
                is_complex=True,
                parent_mechanism=mechanism_text,  # Keep original as parent
                child_mechanisms=[],
                confidence=0.0,
                source='fallback'
            )

    def _create_decomposition_prompt(self, mechanism_text: str) -> str:
        """
        Create LLM prompt for mechanism decomposition.

        Args:
            mechanism_text: Mechanism to decompose

        Returns:
            Prompt string
        """
        prompt = f"""Decompose this biological mechanism into hierarchical components.

Mechanism: "{mechanism_text}"

Extract:
1. **parent_mechanism**: The primary biological process or pathway (most general)
2. **child_mechanisms**: List of specific effects or sub-processes (can be empty if mechanism is already specific)
3. **target_organ**: Anatomical location if specified (or null if not mentioned)
4. **reasoning**: Brief explanation of the decomposition
5. **confidence**: Your confidence in this decomposition (0.0-1.0)

Output format (JSON only):
{{
  "parent_mechanism": "primary pathway or process",
  "child_mechanisms": ["specific effect 1", "specific effect 2"],
  "target_organ": "anatomical location or null",
  "reasoning": "explanation",
  "confidence": 0.9
}}

Examples:

Input: "enhances dopamine and norepinephrine activity in the brain via presynaptic reuptake inhibition"
Output:
{{
  "parent_mechanism": "presynaptic reuptake inhibition",
  "child_mechanisms": ["enhanced dopamine activity", "enhanced norepinephrine activity"],
  "target_organ": "brain",
  "reasoning": "Primary mechanism is reuptake inhibition, which leads to enhanced neurotransmitter activity",
  "confidence": 0.95
}}

Input: "gut microbiome modulation, reduced inflammation, improved gut barrier integrity"
Output:
{{
  "parent_mechanism": "microbiome-mediated gut homeostasis",
  "child_mechanisms": ["gut microbiome modulation", "reduced inflammation", "improved gut barrier integrity"],
  "target_organ": "gastrointestinal tract",
  "reasoning": "Multiple related effects on gut health, united by microbiome involvement",
  "confidence": 0.90
}}

Input: "inflammation reduction"
Output:
{{
  "parent_mechanism": "inflammation reduction",
  "child_mechanisms": [],
  "target_organ": null,
  "reasoning": "Already a specific mechanism, no further decomposition needed",
  "confidence": 0.85
}}

Now decompose the mechanism above. Output JSON only:"""

        return prompt

    def preprocess_mechanisms(
        self,
        mechanisms: List[str],
        selective: bool = True,
        max_decompositions: Optional[int] = None
    ) -> PreprocessingResult:
        """
        Preprocess list of mechanisms.

        Args:
            mechanisms: List of mechanism texts
            selective: If True, only decompose complex mechanisms
            max_decompositions: Optional limit on decompositions (for testing)

        Returns:
            PreprocessingResult with decompositions
        """
        start_time = datetime.now()

        logger.info(f"Preprocessing {len(mechanisms)} mechanisms (selective={selective})...")

        decompositions = []
        complex_count = 0
        decomposed_count = 0

        for i, mechanism in enumerate(mechanisms):
            # Check if complex
            is_complex_flag, reasons = self.is_complex(mechanism)

            if is_complex_flag:
                complex_count += 1

            # Decompose if needed
            if (selective and is_complex_flag) or not selective:
                # Check max limit
                if max_decompositions and decomposed_count >= max_decompositions:
                    break

                decomposition = self.decompose_mechanism(mechanism, force=not selective)

                # Only count successful decompositions
                if decomposition.source == 'llm' and decomposition.confidence >= self.decomposition_threshold:
                    decomposed_count += 1

                decompositions.append(decomposition)

                # Log progress
                if (i + 1) % 50 == 0:
                    logger.info(f"  Processed {i + 1}/{len(mechanisms)} mechanisms")

        elapsed_time = (datetime.now() - start_time).total_seconds()

        # Compute statistics
        success_rate = decomposed_count / complex_count if complex_count > 0 else 0.0

        children_counts = [len(d.child_mechanisms) for d in decompositions
                          if d.source == 'llm' and len(d.child_mechanisms) > 0]
        avg_children = sum(children_counts) / len(children_counts) if children_counts else 0.0

        result = PreprocessingResult(
            total_mechanisms=len(mechanisms),
            complex_count=complex_count,
            decomposed_count=decomposed_count,
            decomposition_success_rate=success_rate,
            avg_children_per_decomposition=avg_children,
            elapsed_time_seconds=elapsed_time,
            decompositions=decompositions
        )

        logger.info(f"\nPreprocessing complete:")
        logger.info(f"  Total mechanisms: {result.total_mechanisms}")
        logger.info(f"  Complex mechanisms: {result.complex_count} ({result.complex_count/result.total_mechanisms:.1%})")
        logger.info(f"  Decomposed: {result.decomposed_count}")
        logger.info(f"  Success rate: {result.decomposition_success_rate:.1%}")
        logger.info(f"  Avg children per decomposition: {result.avg_children:.1f}")
        logger.info(f"  Time: {result.elapsed_time_seconds:.1f}s")

        return result

    def generate_enhanced_dataset(
        self,
        mechanisms: List[str],
        decompositions: List[MechanismDecomposition],
        include_originals: bool = True
    ) -> List[str]:
        """
        Generate enhanced mechanism dataset with decompositions.

        Args:
            mechanisms: Original mechanism texts
            decompositions: Decomposition results
            include_originals: Include original texts for non-decomposed mechanisms

        Returns:
            Enhanced list of mechanism texts
        """
        enhanced = []

        # Create lookup for decompositions
        decomp_map = {d.original_text: d for d in decompositions}

        for mechanism in mechanisms:
            if mechanism in decomp_map:
                decomp = decomp_map[mechanism]

                # Only use decomposition if successful
                if decomp.source == 'llm' and decomp.confidence >= self.decomposition_threshold:
                    # Add parent
                    if decomp.parent_mechanism:
                        enhanced.append(decomp.parent_mechanism)

                    # Add children
                    enhanced.extend(decomp.child_mechanisms)
                else:
                    # Use original
                    if include_originals:
                        enhanced.append(mechanism)
            else:
                # No decomposition, use original
                if include_originals:
                    enhanced.append(mechanism)

        logger.info(f"Enhanced dataset: {len(mechanisms)} → {len(enhanced)} mechanisms")

        return enhanced

    def validate_decomposition_quality(
        self,
        decompositions: List[MechanismDecomposition],
        sample_size: int = 20
    ) -> float:
        """
        Validate decomposition quality via manual review of sample.

        For automation, uses heuristics. In production, would involve human review.

        Args:
            decompositions: List of decompositions to validate
            sample_size: Number to sample for review

        Returns:
            Accuracy score (0.0-1.0)
        """
        # Filter successful decompositions
        successful = [d for d in decompositions
                     if d.source == 'llm' and len(d.child_mechanisms) > 0]

        if not successful:
            logger.warning("No successful decompositions to validate")
            return 0.0

        # Sample
        import random
        sample = random.sample(successful, min(sample_size, len(successful)))

        correct_count = 0
        logger.info(f"\nValidating {len(sample)} decompositions:")

        for decomp in sample:
            # Heuristic validation:
            # 1. Parent is shorter than original (more general)
            # 2. Children are semantically related (similar length)
            # 3. Confidence >= threshold

            parent_shorter = len(decomp.parent_mechanism) < len(decomp.original_text)
            high_confidence = decomp.confidence >= self.decomposition_threshold

            # Children length similarity
            if decomp.child_mechanisms:
                child_lengths = [len(c) for c in decomp.child_mechanisms]
                avg_length = sum(child_lengths) / len(child_lengths)
                std_length = (sum((l - avg_length) ** 2 for l in child_lengths) / len(child_lengths)) ** 0.5
                cv = std_length / avg_length if avg_length > 0 else 1.0
                children_similar = cv < 0.5
            else:
                children_similar = True

            is_correct = parent_shorter and high_confidence and children_similar

            if is_correct:
                correct_count += 1

            # Log sample
            logger.info(f"  Original: {decomp.original_text[:60]}...")
            logger.info(f"  Parent: {decomp.parent_mechanism}")
            logger.info(f"  Children: {decomp.child_mechanisms}")
            logger.info(f"  Validation: {'✓' if is_correct else '✗'}")

        accuracy = correct_count / len(sample)
        logger.info(f"\nValidation accuracy: {accuracy:.1%} ({correct_count}/{len(sample)})")

        return accuracy


def main():
    """Command-line interface for preprocessor."""
    import argparse

    parser = argparse.ArgumentParser(description="Mechanism Text Preprocessor")
    parser.add_argument('--db-path', required=True, help='Path to intervention_research.db')
    parser.add_argument('--action', choices=['analyze', 'decompose', 'validate'], default='analyze',
                       help='Action to perform')
    parser.add_argument('--selective', action='store_true', help='Only decompose complex mechanisms')
    parser.add_argument('--max-decompositions', type=int, help='Limit decompositions (for testing)')
    parser.add_argument('--sample-size', type=int, default=20, help='Sample size for validation')

    args = parser.parse_args()

    # Load mechanisms from database
    import sqlite3
    conn = sqlite3.connect(args.db_path)
    cursor = conn.cursor()

    # Check if interventions table exists
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='interventions'")
    if not cursor.fetchone():
        print("Error: interventions table does not exist")
        conn.close()
        return

    # Load mechanisms
    cursor.execute("""
        SELECT DISTINCT mechanism
        FROM interventions
        WHERE mechanism IS NOT NULL AND mechanism != ''
        ORDER BY mechanism
    """)

    mechanisms = [row[0] for row in cursor.fetchall()]
    conn.close()

    if not mechanisms:
        print("Error: No mechanisms found in database")
        return

    print(f"Loaded {len(mechanisms)} mechanisms from database")

    # Create preprocessor
    preprocessor = MechanismPreprocessor()

    if args.action == 'analyze':
        # Analyze complexity
        complex_count = 0
        for mechanism in mechanisms:
            is_complex, reasons = preprocessor.is_complex(mechanism)
            if is_complex:
                complex_count += 1

        print(f"\nComplexity Analysis:")
        print(f"  Total mechanisms: {len(mechanisms)}")
        print(f"  Complex: {complex_count} ({complex_count/len(mechanisms):.1%})")
        print(f"  Simple: {len(mechanisms) - complex_count} ({(len(mechanisms)-complex_count)/len(mechanisms):.1%})")

    elif args.action == 'decompose':
        # Preprocess mechanisms
        result = preprocessor.preprocess_mechanisms(
            mechanisms,
            selective=args.selective,
            max_decompositions=args.max_decompositions
        )

        print(f"\nDecomposition complete:")
        print(f"  Decomposed: {result.decomposed_count}")
        print(f"  Success rate: {result.decomposition_success_rate:.1%}")

    elif args.action == 'validate':
        # Decompose and validate
        result = preprocessor.preprocess_mechanisms(
            mechanisms,
            selective=args.selective,
            max_decompositions=args.max_decompositions
        )

        accuracy = preprocessor.validate_decomposition_quality(
            result.decompositions,
            sample_size=args.sample_size
        )

        print(f"\nValidation accuracy: {accuracy:.1%}")


if __name__ == "__main__":
    main()
