"""
Group-Based Semantic Categorization Experiment

Tests categorizing canonical groups instead of individual interventions.
Expected benefits:
- 80% reduction in LLM calls (10,000 interventions → ~2,000 groups)
- Better semantic context for categorization
- Consistent categories across intervention variants
"""

from .group_categorizer import GroupBasedCategorizer
from .validation import validate_category_coverage, validate_group_purity, compare_with_existing
from .experiment_runner import run_experiment

__all__ = [
    'GroupBasedCategorizer',
    'validate_category_coverage',
    'validate_group_purity',
    'compare_with_existing',
    'run_experiment'
]
