"""
Emerging category analyzer for discovering new intervention types.
Analyzes interventions classified as "emerging" to identify patterns and suggest new categories.
"""

import json
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass
from collections import Counter, defaultdict
from datetime import datetime
import re

from back_end.src.data.config import setup_logging
from back_end.src.data.repositories import repository_manager
from back_end.src.interventions.taxonomy import InterventionType, intervention_taxonomy

logger = setup_logging(__name__, 'emerging_category_analyzer.log')


@dataclass
class EmergingCategoryCandidate:
    """Represents a candidate for a new intervention category."""

    proposed_name: str
    intervention_count: int
    unique_papers: int
    common_keywords: List[str]
    sample_interventions: List[str]
    confidence_score: float
    rationale_summary: str


class EmergingCategoryAnalyzer:
    """
    Analyzes emerging interventions to discover new category patterns.
    Uses clustering and keyword analysis to identify potential new categories.
    """

    def __init__(self, repository_mgr=None):
        """
        Initialize the analyzer.

        Args:
            repository_mgr: Repository manager instance (optional, uses global if None)
        """
        self.repository_mgr = repository_mgr or repository_manager
        self.taxonomy = intervention_taxonomy

    def analyze_emerging_interventions(self, min_intervention_count: int = 3,
                                     min_unique_papers: int = 2) -> List[EmergingCategoryCandidate]:
        """
        Analyze all emerging interventions to discover new category patterns.

        Args:
            min_intervention_count: Minimum number of interventions required for a candidate
            min_unique_papers: Minimum number of unique papers required for a candidate

        Returns:
            List of emerging category candidates sorted by confidence
        """
        # Starting analysis (logging removed for performance)

        # Get all emerging interventions
        emerging_interventions = self._get_emerging_interventions()

        if not emerging_interventions:
            # No emerging interventions found (logging removed for performance)
            return []

        # Analyzing emerging interventions (logging removed for performance)

        # Group by proposed category
        category_groups = self._group_by_proposed_category(emerging_interventions)

        # Analyze each group
        candidates = []
        for proposed_category, interventions in category_groups.items():
            candidate = self._analyze_category_group(
                proposed_category, interventions, min_intervention_count, min_unique_papers
            )
            if candidate:
                candidates.append(candidate)

        # Sort by confidence score
        candidates.sort(key=lambda x: x.confidence_score, reverse=True)

        # Found category candidates (logging removed for performance)
        return candidates

    def _get_emerging_interventions(self) -> List[Dict[str, Any]]:
        """Retrieve all interventions classified as emerging from the database."""
        try:
            with self.repository_mgr.get_connection() as conn:
                cursor = conn.cursor()

                cursor.execute('''
                    SELECT
                        intervention_name,
                        intervention_details,
                        health_condition,
                        correlation_type,
                        supporting_quote,
                        paper_id,
                        extraction_model
                    FROM interventions
                    WHERE intervention_category = 'emerging'
                    ORDER BY intervention_name
                ''')

                interventions = []
                for row in cursor.fetchall():
                    # Parse intervention_details JSON
                    details = {}
                    if row[1]:
                        try:
                            details = json.loads(row[1]) if isinstance(row[1], str) else row[1]
                        except json.JSONDecodeError:
                            logger.warning(f"Could not parse intervention_details for {row[0]}")

                    interventions.append({
                        'intervention_name': row[0],
                        'intervention_details': details,
                        'health_condition': row[2],
                        'correlation_type': row[3],
                        'supporting_quote': row[4],
                        'paper_id': row[5],
                        'extraction_model': row[6]
                    })

                return interventions

        except Exception as e:
            logger.error(f"Error retrieving emerging interventions: {e}")
            return []

    def _group_by_proposed_category(self, interventions: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Group interventions by their proposed category."""
        groups = defaultdict(list)

        for intervention in interventions:
            details = intervention.get('intervention_details', {})
            proposed_category = details.get('proposed_category', 'unknown')

            # Normalize category name
            proposed_category = self._normalize_category_name(proposed_category)
            groups[proposed_category].append(intervention)

        return dict(groups)

    def _normalize_category_name(self, category_name: str) -> str:
        """Normalize category names for better grouping."""
        if not category_name or category_name == 'unknown':
            return 'uncategorized'

        # Convert to lowercase and remove special characters
        normalized = re.sub(r'[^a-zA-Z0-9\s]', '', category_name.lower())
        normalized = re.sub(r'\s+', '_', normalized.strip())

        # Handle common variations
        synonyms = {
            'digital_therapeutics': 'digital_health',
            'digital_medicine': 'digital_health',
            'telemedicine': 'digital_health',
            'gene_therapy': 'genetic_therapy',
            'genetic_intervention': 'genetic_therapy',
            'precision_medicine': 'personalized_medicine',
            'individualized_therapy': 'personalized_medicine',
        }

        return synonyms.get(normalized, normalized)

    def _analyze_category_group(self, proposed_category: str, interventions: List[Dict[str, Any]],
                               min_intervention_count: int, min_unique_papers: int) -> Optional[EmergingCategoryCandidate]:
        """Analyze a group of interventions for the same proposed category."""

        if len(interventions) < min_intervention_count:
            return None

        unique_papers = len(set(i['paper_id'] for i in interventions))
        if unique_papers < min_unique_papers:
            return None

        # Extract keywords from intervention names and rationales
        keywords = self._extract_keywords(interventions)
        common_keywords = [word for word, count in keywords.most_common(10)]

        # Get sample interventions
        sample_interventions = list(set(i['intervention_name'] for i in interventions[:5]))

        # Generate rationale summary
        rationale_summary = self._generate_rationale_summary(interventions)

        # Calculate confidence score
        confidence_score = self._calculate_confidence_score(
            interventions, unique_papers, keywords, proposed_category
        )

        return EmergingCategoryCandidate(
            proposed_name=proposed_category,
            intervention_count=len(interventions),
            unique_papers=unique_papers,
            common_keywords=common_keywords,
            sample_interventions=sample_interventions,
            confidence_score=confidence_score,
            rationale_summary=rationale_summary
        )

    def _extract_keywords(self, interventions: List[Dict[str, Any]]) -> Counter:
        """Extract and count keywords from intervention names and rationales."""
        keywords = Counter()

        for intervention in interventions:
            # Extract from intervention name
            name_words = re.findall(r'\b\w{3,}\b', intervention['intervention_name'].lower())
            keywords.update(name_words)

            # Extract from category rationale
            details = intervention.get('intervention_details', {})
            rationale = details.get('category_rationale', '')
            if rationale:
                rationale_words = re.findall(r'\b\w{4,}\b', rationale.lower())
                keywords.update(rationale_words)

        # Remove common stop words
        stop_words = {
            'intervention', 'therapy', 'treatment', 'medication', 'drug', 'approach',
            'method', 'technique', 'procedure', 'study', 'research', 'clinical'
        }

        for stop_word in stop_words:
            if stop_word in keywords:
                del keywords[stop_word]

        return keywords

    def _generate_rationale_summary(self, interventions: List[Dict[str, Any]]) -> str:
        """Generate a summary of rationales for this category."""
        rationales = []

        for intervention in interventions:
            details = intervention.get('intervention_details', {})
            rationale = details.get('category_rationale', '')
            if rationale and len(rationale) > 10:
                rationales.append(rationale)

        if not rationales:
            return "No detailed rationales provided"

        # Find common themes in rationales
        common_words = Counter()
        for rationale in rationales:
            words = re.findall(r'\b\w{4,}\b', rationale.lower())
            common_words.update(words)

        top_themes = [word for word, count in common_words.most_common(5)]
        return f"Common themes: {', '.join(top_themes)}"

    def _calculate_confidence_score(self, interventions: List[Dict[str, Any]], unique_papers: int,
                                   keywords: Counter, proposed_category: str) -> float:
        """Calculate confidence score for this category candidate."""

        # Base score from intervention count and paper diversity
        base_score = min(0.4, len(interventions) * 0.05)  # Max 0.4 from count
        diversity_score = min(0.3, unique_papers * 0.1)   # Max 0.3 from diversity

        # Keyword consistency score
        if keywords:
            top_keyword_frequency = keywords.most_common(1)[0][1] / len(interventions)
            keyword_score = min(0.2, top_keyword_frequency)
        else:
            keyword_score = 0

        # Category name quality score
        name_quality = 0.1 if proposed_category != 'uncategorized' else 0

        total_score = base_score + diversity_score + keyword_score + name_quality
        return min(1.0, total_score)

    def generate_analysis_report(self, candidates: List[EmergingCategoryCandidate]) -> Dict[str, Any]:
        """Generate a comprehensive analysis report."""

        if not candidates:
            return {
                'summary': 'No emerging category candidates found',
                'recommendations': [],
                'total_candidates': 0
            }

        # High confidence candidates (score > 0.7)
        high_confidence = [c for c in candidates if c.confidence_score > 0.7]

        # Medium confidence candidates (score 0.4-0.7)
        medium_confidence = [c for c in candidates if 0.4 <= c.confidence_score <= 0.7]

        recommendations = []

        for candidate in high_confidence:
            recommendations.append({
                'action': 'create_new_category',
                'category_name': candidate.proposed_name,
                'priority': 'high',
                'justification': f"Strong evidence: {candidate.intervention_count} interventions "
                               f"across {candidate.unique_papers} papers"
            })

        for candidate in medium_confidence[:3]:  # Top 3 medium confidence
            recommendations.append({
                'action': 'monitor_and_collect_more_data',
                'category_name': candidate.proposed_name,
                'priority': 'medium',
                'justification': f"Moderate evidence: needs more interventions for validation"
            })

        return {
            'summary': f"Analyzed {len(candidates)} emerging category candidates",
            'high_confidence_candidates': len(high_confidence),
            'medium_confidence_candidates': len(medium_confidence),
            'recommendations': recommendations,
            'total_candidates': len(candidates),
            'detailed_candidates': [
                {
                    'name': c.proposed_name,
                    'intervention_count': c.intervention_count,
                    'unique_papers': c.unique_papers,
                    'confidence_score': round(c.confidence_score, 3),
                    'sample_interventions': c.sample_interventions[:3],
                    'common_keywords': c.common_keywords[:5]
                }
                for c in candidates[:10]  # Top 10 candidates
            ]
        }

    def export_candidates_for_review(self, candidates: List[EmergingCategoryCandidate],
                                   output_path: str) -> bool:
        """Export candidates to a file for manual review."""
        try:
            export_data = {
                'analysis_timestamp': str(datetime.now()),
                'total_candidates': len(candidates),
                'candidates': [
                    {
                        'proposed_name': c.proposed_name,
                        'intervention_count': c.intervention_count,
                        'unique_papers': c.unique_papers,
                        'confidence_score': c.confidence_score,
                        'common_keywords': c.common_keywords,
                        'sample_interventions': c.sample_interventions,
                        'rationale_summary': c.rationale_summary
                    }
                    for c in candidates
                ]
            }

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)

            # Exported candidates (logging removed for performance)
            return True

        except Exception as e:
            logger.error(f"Error exporting candidates: {e}")
            return False


# Global instance
emerging_category_analyzer = EmergingCategoryAnalyzer()