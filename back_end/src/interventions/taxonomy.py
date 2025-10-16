"""
Simplified Intervention Taxonomy - For Phase 1/2 Validation Only

This is a minimal taxonomy implementation to support Phase 1/2 validation.
Phase 3c now uses dynamic category discovery instead of hard-coded taxonomy.
"""

from enum import Enum
from typing import Dict, List, Any


class InterventionType(Enum):
    """Primary intervention categories."""
    EXERCISE = "exercise"
    DIET = "diet"
    SUPPLEMENT = "supplement"
    MEDICATION = "medication"
    THERAPY = "therapy"
    LIFESTYLE = "lifestyle"
    SURGERY = "surgery"
    TEST = "test"
    DEVICE = "device"
    PROCEDURE = "procedure"
    BIOLOGICS = "biologics"
    GENE_THERAPY = "gene_therapy"
    EMERGING = "emerging"


class CategoryField:
    """Minimal field definition for validation."""
    def __init__(self, name: str, data_type: str, required: bool = False, validation_rules: Dict = None):
        self.name = name
        self.data_type = data_type
        self.required = required
        self.validation_rules = validation_rules or {}


class CategoryDefinition:
    """Minimal category definition for validation."""
    def __init__(self, category: InterventionType, fields: List[CategoryField]):
        self.category = category
        self._fields = fields

    def get_all_fields(self) -> List[CategoryField]:
        return self._fields


class InterventionTaxonomy:
    """
    Simplified taxonomy for Phase 1/2 validation only.
    Phase 3c uses dynamic category discovery.
    """

    def __init__(self):
        self._categories = self._build_minimal_categories()

    def _build_minimal_categories(self) -> Dict[InterventionType, CategoryDefinition]:
        """Build minimal category definitions with common fields."""
        common_fields = [
            CategoryField("dosage", "string"),
            CategoryField("frequency", "string"),
            CategoryField("duration", "string"),
            CategoryField("intensity", "string"),
        ]

        categories = {}
        for intervention_type in InterventionType:
            categories[intervention_type] = CategoryDefinition(
                intervention_type,
                common_fields
            )

        return categories

    def get_category(self, category: InterventionType) -> CategoryDefinition:
        """Get category definition."""
        return self._categories.get(category)


# Global instance for Phase 1/2 validation
intervention_taxonomy = InterventionTaxonomy()
