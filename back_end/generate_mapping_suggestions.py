#!/usr/bin/env python3
"""
Standalone Mapping Suggestion Generator

This script analyzes existing data and suggests potential canonical mappings
without applying them. It generates CSV reports for human review.

Usage: python generate_mapping_suggestions.py
"""

import sqlite3
import csv
import sys
import os
from datetime import datetime
from typing import List, Dict, Optional, Tuple

# Add the src directory to the path so we can import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from entity_normalizer import EntityNormalizer


class MappingSuggestionGenerator:
    """Generates mapping suggestions for existing data"""

    def __init__(self, db_path: str):
        """Initialize with database connection"""
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.normalizer = EntityNormalizer(self.conn)
        self.suggestions = []

    def load_unique_terms(self, entity_type: str) -> List[Tuple[str, int]]:
        """
        Load unique terms from the database with frequency counts

        Args:
            entity_type: 'intervention' or 'condition'

        Returns:
            List of (term, frequency) tuples
        """
        cursor = self.conn.cursor()

        if entity_type == 'intervention':
            cursor.execute("""
                SELECT intervention_name, COUNT(*) as frequency
                FROM interventions
                GROUP BY intervention_name
                ORDER BY frequency DESC, intervention_name
            """)
        elif entity_type == 'condition':
            cursor.execute("""
                SELECT health_condition, COUNT(*) as frequency
                FROM interventions
                GROUP BY health_condition
                ORDER BY frequency DESC, health_condition
            """)
        else:
            raise ValueError("entity_type must be 'intervention' or 'condition'")

        return [(row[0], row[1]) for row in cursor.fetchall()]

    def find_best_mapping_suggestion(self, term: str, entity_type: str) -> Optional[Dict]:
        """
        Find the best canonical mapping suggestion using SAFE methods for medical terms

        MEDICAL SAFETY PRIORITY:
        - No dangerous similarity matching (prebiotics/probiotics confusion)
        - Conservative pattern matching only
        - Flag uncertain matches for LLM verification
        - Better to have unmapped terms than incorrect medical mappings!

        Args:
            term: The original term to find mapping for
            entity_type: 'intervention' or 'condition'

        Returns:
            Dictionary with mapping suggestion or None if no good match
        """
        # Use the new safe matching method
        safe_matches = self.normalizer.find_safe_matches_only(term, entity_type)

        if safe_matches:
            best_match = safe_matches[0]  # Already sorted by confidence

            # Removed debug output

            # Convert to our expected format
            confidence = best_match.get('confidence', 0.95)  # Default for safe matches
            method = best_match.get('match_method', 'unknown_safe_match')

            # Add safety notes
            if method == 'existing_mapping':
                notes = 'Already mapped in database'
            elif method == 'exact_normalized':
                notes = 'Exact match after case/punctuation normalization'
            elif 'pattern' in method:
                notes = f"Safe pattern match: {method}"
            else:
                notes = f"Safe match via {method}"

            return {
                'original_term': term,
                'suggested_canonical': best_match['canonical_name'],
                'confidence': confidence,
                'method': method,
                'canonical_id': best_match['id'],
                'notes': notes
            }

        # Check if we should flag this for LLM verification
        # (e.g., if it's a medical term that might have variants we don't recognize)
        if self._should_flag_for_llm_verification(term, entity_type):
            return {
                'original_term': term,
                'suggested_canonical': term,  # Keep as-is for now
                'confidence': 0.0,
                'method': 'needs_llm_verification',
                'canonical_id': None,
                'notes': 'Medical term requires LLM verification for safe mapping'
            }

        # No good match found
        return {
            'original_term': term,
            'suggested_canonical': term,  # Keep as-is
            'confidence': 0.0,
            'method': 'no_safe_match',
            'canonical_id': None,
            'notes': 'No safe canonical mapping found - medical terms require manual review'
        }

    def _should_flag_for_llm_verification(self, term: str, entity_type: str) -> bool:
        """
        Determine if a term should be flagged for LLM verification.

        Flags terms that:
        - Look medical but aren't in our system
        - Could be variations of existing terms
        - Contain medical prefixes/suffixes that suggest they're related to existing terms

        Args:
            term: The term to evaluate
            entity_type: 'intervention' or 'condition'

        Returns:
            True if term should be sent to LLM for verification
        """
        normalized_term = self.normalizer.normalize_term(term)

        # Medical prefixes that suggest this might be a variant of something we know
        medical_prefixes = ['pre', 'post', 'anti', 'pro', 'hyper', 'hypo', 'multi', 'mono']
        medical_suffixes = ['itis', 'osis', 'emia', 'pathy', 'therapy', 'biotic', 'genic']

        # Check if term has medical-sounding components
        has_medical_component = any(
            prefix in normalized_term for prefix in medical_prefixes
        ) or any(
            suffix in normalized_term for suffix in medical_suffixes
        )

        # For now, be conservative - only flag obvious medical terms
        # This prevents flooding the LLM queue
        if has_medical_component and len(normalized_term) > 6:
            return True

        return False

    def generate_suggestions_for_entity_type(self, entity_type: str, max_terms: Optional[int] = None) -> List[Dict]:
        """
        Generate mapping suggestions for all terms of a specific entity type

        Args:
            entity_type: 'intervention' or 'condition'
            max_terms: Optional limit on number of terms to process

        Returns:
            List of suggestion dictionaries
        """
        print(f"\n=== Processing {entity_type}s ===")

        # Load all unique terms
        terms = self.load_unique_terms(entity_type)
        if max_terms:
            terms = terms[:max_terms]

        print(f"Loaded {len(terms)} unique {entity_type} terms")

        suggestions = []
        processed = 0

        for term, frequency in terms:
            processed += 1
            if processed % 50 == 0:  # Progress indicator
                print(f"  Processed {processed}/{len(terms)} {entity_type}s...")

            suggestion = self.find_best_mapping_suggestion(term, entity_type)
            if suggestion:
                suggestion['entity_type'] = entity_type
                suggestion['frequency'] = frequency
                suggestions.append(suggestion)

        print(f"Completed processing {len(suggestions)} {entity_type} suggestions")
        return suggestions

    def generate_summary_statistics(self, suggestions: List[Dict]) -> Dict:
        """Generate summary statistics for the suggestions"""

        total_terms = len(suggestions)

        # Count by method
        method_counts = {}
        confidence_ranges = {'high': 0, 'medium': 0, 'low': 0, 'none': 0}
        mappable_terms = 0

        for suggestion in suggestions:
            method = suggestion['method']
            confidence = suggestion['confidence']

            method_counts[method] = method_counts.get(method, 0) + 1

            if confidence >= 0.8:
                confidence_ranges['high'] += 1
                mappable_terms += 1
            elif confidence >= 0.6:
                confidence_ranges['medium'] += 1
                mappable_terms += 1
            elif confidence > 0:
                confidence_ranges['low'] += 1
                mappable_terms += 1
            else:
                confidence_ranges['none'] += 1

        return {
            'total_terms': total_terms,
            'mappable_terms': mappable_terms,
            'unmappable_terms': total_terms - mappable_terms,
            'mappable_percentage': (mappable_terms / total_terms) * 100 if total_terms > 0 else 0,
            'method_counts': method_counts,
            'confidence_ranges': confidence_ranges
        }

    def save_suggestions_to_csv(self, suggestions: List[Dict], filename: str):
        """Save suggestions to CSV file"""

        fieldnames = [
            'entity_type', 'original_term', 'frequency', 'suggested_canonical',
            'confidence', 'method', 'canonical_id', 'notes'
        ]

        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            # Sort by confidence (high to low) then by frequency (high to low)
            sorted_suggestions = sorted(
                suggestions,
                key=lambda x: (x['confidence'], x['frequency']),
                reverse=True
            )

            for suggestion in sorted_suggestions:
                writer.writerow(suggestion)

        print(f"Saved {len(suggestions)} suggestions to {filename}")

    def run_full_analysis(self, max_interventions: Optional[int] = None, max_conditions: Optional[int] = None):
        """Run the complete mapping suggestion analysis"""

        print("="*60)
        print("MAPPING SUGGESTION GENERATOR")
        print("="*60)
        print(f"Database: {self.db_path}")
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        all_suggestions = []

        # Process interventions
        intervention_suggestions = self.generate_suggestions_for_entity_type('intervention', max_interventions)
        all_suggestions.extend(intervention_suggestions)

        # Process conditions
        condition_suggestions = self.generate_suggestions_for_entity_type('condition', max_conditions)
        all_suggestions.extend(condition_suggestions)

        # Generate summary statistics
        print(f"\n=== SUMMARY STATISTICS ===")

        intervention_stats = self.generate_summary_statistics(
            [s for s in all_suggestions if s['entity_type'] == 'intervention']
        )
        condition_stats = self.generate_summary_statistics(
            [s for s in all_suggestions if s['entity_type'] == 'condition']
        )
        overall_stats = self.generate_summary_statistics(all_suggestions)

        print(f"\nInterventions:")
        print(f"  Total terms: {intervention_stats['total_terms']}")
        print(f"  Mappable: {intervention_stats['mappable_terms']} ({intervention_stats['mappable_percentage']:.1f}%)")
        print(f"  Top methods: {dict(list(sorted(intervention_stats['method_counts'].items(), key=lambda x: x[1], reverse=True))[:3])}")

        print(f"\nConditions:")
        print(f"  Total terms: {condition_stats['total_terms']}")
        print(f"  Mappable: {condition_stats['mappable_terms']} ({condition_stats['mappable_percentage']:.1f}%)")
        print(f"  Top methods: {dict(list(sorted(condition_stats['method_counts'].items(), key=lambda x: x[1], reverse=True))[:3])}")

        print(f"\nOverall:")
        print(f"  Total terms: {overall_stats['total_terms']}")
        print(f"  Mappable: {overall_stats['mappable_terms']} ({overall_stats['mappable_percentage']:.1f}%)")
        print(f"  Confidence distribution: {overall_stats['confidence_ranges']}")

        # Save to CSV files
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Save all suggestions
        all_filename = f"mapping_suggestions_all_{timestamp}.csv"
        self.save_suggestions_to_csv(all_suggestions, all_filename)

        # Save high-confidence suggestions only
        high_confidence_suggestions = [s for s in all_suggestions if s['confidence'] >= 0.8]
        if high_confidence_suggestions:
            high_confidence_filename = f"mapping_suggestions_high_confidence_{timestamp}.csv"
            self.save_suggestions_to_csv(high_confidence_suggestions, high_confidence_filename)
            print(f"Saved {len(high_confidence_suggestions)} high-confidence suggestions to {high_confidence_filename}")

        # Save interventions only
        intervention_filename = f"mapping_suggestions_interventions_{timestamp}.csv"
        self.save_suggestions_to_csv(intervention_suggestions, intervention_filename)

        # Save conditions only
        condition_filename = f"mapping_suggestions_conditions_{timestamp}.csv"
        self.save_suggestions_to_csv(condition_suggestions, condition_filename)

        print(f"\n=== FILES GENERATED ===")
        print(f"1. {all_filename} - All suggestions")
        print(f"2. {intervention_filename} - Intervention suggestions only")
        print(f"3. {condition_filename} - Condition suggestions only")
        if high_confidence_suggestions:
            print(f"4. {high_confidence_filename} - High confidence suggestions only")

        return all_suggestions

    def close(self):
        """Close database connection"""
        self.conn.close()


def main():
    """Main function to run the mapping suggestion generator"""

    # Database path
    db_path = "data/processed/intervention_research.db"

    if not os.path.exists(db_path):
        print(f"Error: Database not found at {db_path}")
        sys.exit(1)

    # Create generator
    generator = MappingSuggestionGenerator(db_path)

    try:
        # Run analysis (limit for testing, remove limits for full analysis)
        suggestions = generator.run_full_analysis(
            max_interventions=200,  # Remove this limit for full analysis
            max_conditions=200      # Remove this limit for full analysis
        )

        print(f"\n[SUCCESS] Generated {len(suggestions)} mapping suggestions!")
        print("Review the CSV files and approve suggestions before applying them.")

    except Exception as e:
        print(f"Error during analysis: {e}")
        sys.exit(1)
    finally:
        generator.close()


if __name__ == "__main__":
    main()