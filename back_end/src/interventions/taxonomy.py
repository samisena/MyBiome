"""
Intervention taxonomy and categorization system.
Defines the structure and hierarchy of health interventions.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Any
from enum import Enum


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
    EMERGING = "emerging"


@dataclass
class InterventionField:
    """Defines a field structure for intervention data."""
    name: str
    required: bool = True
    data_type: str = "string"  # string, number, list, boolean
    validation_rules: Dict[str, Any] = field(default_factory=dict)
    description: str = ""


@dataclass
class InterventionCategory:
    """Defines a complete intervention category with its rules and structure."""

    category: InterventionType
    display_name: str
    description: str
    subcategories: List[str] = field(default_factory=list)
    required_fields: List[InterventionField] = field(default_factory=list)
    optional_fields: List[InterventionField] = field(default_factory=list)
    validation_rules: Dict[str, Any] = field(default_factory=dict)

    def get_all_fields(self) -> List[InterventionField]:
        """Get all fields (required + optional) for this category."""
        return self.required_fields + self.optional_fields

    def get_field_names(self) -> Set[str]:
        """Get all field names for this category."""
        return {field.name for field in self.get_all_fields()}


class CategoryBuilder:
    """Helper class to build intervention categories with common patterns."""

    @staticmethod
    def get_common_required_fields() -> List[InterventionField]:
        """Get fields required for all intervention categories."""
        return [
            InterventionField("intervention_name", True, "string",
                            description="Specific name of the intervention")
        ]

    @staticmethod
    def get_common_optional_fields() -> List[InterventionField]:
        """Get optional fields common to many intervention categories."""
        return [
            InterventionField("duration", False, "string",
                            description="Duration of intervention"),
            InterventionField("frequency", False, "string",
                            description="Frequency of intervention")
        ]

    @staticmethod
    def create_category(category_type: InterventionType, display_name: str,
                       description: str, subcategories: List[str],
                       specific_required: List[InterventionField] = None,
                       specific_optional: List[InterventionField] = None) -> InterventionCategory:
        """Create a category with common fields plus specific ones."""
        specific_required = specific_required or []
        specific_optional = specific_optional or []

        # Combine common fields with specific ones
        required_fields = CategoryBuilder.get_common_required_fields() + specific_required
        optional_fields = CategoryBuilder.get_common_optional_fields() + specific_optional

        return InterventionCategory(
            category=category_type,
            display_name=display_name,
            description=description,
            subcategories=subcategories,
            required_fields=required_fields,
            optional_fields=optional_fields
        )


class InterventionTaxonomy:
    """Complete taxonomy of health interventions."""
    
    def __init__(self):
        self.categories = self._build_taxonomy()
    
    def _build_taxonomy(self) -> Dict[InterventionType, InterventionCategory]:
        """Build the complete intervention taxonomy."""
        
        categories = {}
        
        # EXERCISE Category
        categories[InterventionType.EXERCISE] = CategoryBuilder.create_category(
            category_type=InterventionType.EXERCISE,
            display_name="Exercise & Physical Activity",
            description="Physical exercise interventions including aerobic, resistance, and flexibility training",
            subcategories=[
                "aerobic", "resistance", "flexibility", "balance", "hiit",
                "yoga", "pilates", "sports", "walking", "cycling", "swimming"
            ],
            specific_required=[
                InterventionField("exercise_type", True, "string",
                                description="Type of exercise (aerobic, resistance, etc.)")
            ],
            specific_optional=[
                InterventionField("intensity", False, "string",
                                validation_rules={"allowed_values": ["low", "moderate", "high", "vigorous"]},
                                description="Exercise intensity level"),
                InterventionField("total_program_duration", False, "string",
                                description="Total length of intervention (e.g., '12 weeks')"),
                InterventionField("supervision", False, "string",
                                description="Level of supervision (supervised, unsupervised, etc.)")
            ]
        )
        
        # DIET Category
        categories[InterventionType.DIET] = CategoryBuilder.create_category(
            category_type=InterventionType.DIET,
            display_name="Diet & Nutrition",
            description="Dietary interventions including specific diets, foods, and nutritional modifications",
            subcategories=[
                "mediterranean", "dash", "ketogenic", "low_carb", "low_fat", "vegetarian", "vegan",
                "intermittent_fasting", "caloric_restriction", "specific_foods", "macronutrient_modification"
            ],
            specific_required=[
                InterventionField("diet_type", True, "string",
                                description="Type of dietary intervention")
            ],
            specific_optional=[
                InterventionField("compliance_measure", False, "string",
                                description="How compliance was measured"),
                InterventionField("caloric_intake", False, "string",
                                description="Target caloric intake if specified"),
                InterventionField("macronutrient_ratio", False, "string",
                                description="Macronutrient ratios (e.g., 'low carb: <20% calories')"),
                InterventionField("specific_foods", False, "list",
                                description="List of specific foods included/excluded")
            ]
        )
        
        # SUPPLEMENT Category
        categories[InterventionType.SUPPLEMENT] = CategoryBuilder.create_category(
            category_type=InterventionType.SUPPLEMENT,
            display_name="Supplements & Nutraceuticals",
            description="Nutritional supplements including vitamins, minerals, herbs, and probiotics",
            subcategories=[
                "vitamin", "mineral", "herbal", "probiotic", "prebiotic", "omega3",
                "amino_acid", "protein", "fiber", "antioxidant", "botanical"
            ],
            specific_required=[
                InterventionField("supplement_name", True, "string",
                                description="Specific supplement name")
            ],
            specific_optional=[
                InterventionField("dosage", False, "string",
                                description="Dosage amount and units"),
                InterventionField("form", False, "string",
                                description="Form of supplement (capsule, powder, liquid, etc.)"),
                InterventionField("brand", False, "string",
                                description="Brand name if specified"),
                InterventionField("active_ingredient", False, "string",
                                description="Primary active ingredient")
            ]
        )
        
        # MEDICATION Category
        categories[InterventionType.MEDICATION] = InterventionCategory(
            category=InterventionType.MEDICATION,
            display_name="Medications & Pharmaceuticals",
            description="Pharmaceutical interventions including prescription and over-the-counter drugs",
            subcategories=[
                "antidepressant", "anxiolytic", "antipsychotic", "antibiotic", "anti_inflammatory",
                "antihypertensive", "diabetes_medication", "pain_medication", "hormone", "immunosuppressant"
            ],
            required_fields=[
                InterventionField("medication_name", True, "string",
                                description="Generic or brand name of medication"),
                InterventionField("intervention_name", True, "string",
                                description="Common name of the medication intervention")
            ],
            optional_fields=[
                InterventionField("dosage", False, "string",
                                description="Dosage amount and units"),
                InterventionField("route", False, "string",
                                validation_rules={"allowed_values": ["oral", "iv", "im", "topical", "inhaled", "sublingual"]},
                                description="Route of administration"),
                InterventionField("frequency", False, "string",
                                description="Dosing frequency"),
                InterventionField("duration", False, "string",
                                description="Duration of treatment"),
                InterventionField("drug_class", False, "string",
                                description="Pharmacological class of the medication")
            ]
        )
        
        # THERAPY Category
        categories[InterventionType.THERAPY] = InterventionCategory(
            category=InterventionType.THERAPY,
            display_name="Therapy & Counseling",
            description="Therapeutic interventions including psychological, physical, and behavioral therapies",
            subcategories=[
                "cbt", "dbt", "act", "mindfulness", "psychotherapy", "group_therapy",
                "physical_therapy", "occupational_therapy", "massage", "acupuncture", "counseling"
            ],
            required_fields=[
                InterventionField("therapy_type", True, "string",
                                description="Type of therapy (CBT, physical therapy, etc.)"),
                InterventionField("intervention_name", True, "string",
                                description="Specific name of the therapeutic intervention")
            ],
            optional_fields=[
                InterventionField("duration", False, "string",
                                description="Duration per session"),
                InterventionField("frequency", False, "string",
                                description="Frequency of sessions"),
                InterventionField("total_sessions", False, "number",
                                description="Total number of sessions"),
                InterventionField("delivery_method", False, "string",
                                validation_rules={"allowed_values": ["individual", "group", "online", "phone", "self_guided"]},
                                description="Method of therapy delivery"),
                InterventionField("therapist_type", False, "string",
                                description="Type of therapist or practitioner")
            ]
        )
        
        # LIFESTYLE Category
        categories[InterventionType.LIFESTYLE] = InterventionCategory(
            category=InterventionType.LIFESTYLE,
            display_name="Lifestyle Modifications",
            description="Lifestyle interventions including sleep, stress management, and behavioral changes",
            subcategories=[
                "sleep_hygiene", "stress_management", "smoking_cessation", "alcohol_reduction",
                "social_support", "environmental_modification", "time_management", "relaxation"
            ],
            required_fields=[
                InterventionField("lifestyle_type", True, "string",
                                description="Type of lifestyle modification"),
                InterventionField("intervention_name", True, "string",
                                description="Specific name of the lifestyle intervention")
            ],
            optional_fields=[
                InterventionField("duration", False, "string",
                                description="Duration of intervention"),
                InterventionField("intensity", False, "string",
                                description="Intensity or level of intervention"),
                InterventionField("support_type", False, "string",
                                description="Type of support provided"),
                InterventionField("target_behavior", False, "string",
                                description="Specific behavior being targeted"),
                InterventionField("measurement_method", False, "string",
                                description="How the lifestyle change was measured")
            ]
        )

        # SURGERY Category
        categories[InterventionType.SURGERY] = CategoryBuilder.create_category(
            category_type=InterventionType.SURGERY,
            display_name="Surgical Interventions",
            description="Surgical procedures and operations for medical treatment and intervention",
            subcategories=[
                "minimally_invasive", "open_surgery", "laparoscopic", "endoscopic", "robotic",
                "reconstructive", "bariatric", "cardiac", "neurological", "orthopedic", "transplant"
            ],
            specific_required=[
                InterventionField("surgery_type", True, "string",
                                description="Type of surgical procedure")
            ],
            specific_optional=[
                InterventionField("approach", False, "string",
                                validation_rules={"allowed_values": ["open", "laparoscopic", "endoscopic", "robotic", "minimally_invasive"]},
                                description="Surgical approach method"),
                InterventionField("anesthesia_type", False, "string",
                                validation_rules={"allowed_values": ["general", "local", "regional", "conscious_sedation"]},
                                description="Type of anesthesia used"),
                InterventionField("postop_care", False, "string",
                                description="Post-operative care requirements"),
                InterventionField("recovery_time", False, "string",
                                description="Expected recovery time"),
                InterventionField("complications", False, "list",
                                description="Reported complications or adverse events")
            ]
        )

        # TEST Category
        categories[InterventionType.TEST] = CategoryBuilder.create_category(
            category_type=InterventionType.TEST,
            display_name="Tests & Diagnostics",
            description="Medical tests and diagnostic procedures for identifying health conditions",
            subcategories=[
                "blood_test", "breath_test", "stool_test", "urine_test", "genetic_test",
                "imaging", "biopsy", "endoscopy", "functional_test", "biomarker", "lab_test"
            ],
            specific_required=[
                InterventionField("test_type", True, "string",
                                description="Type of test or diagnostic procedure")
            ],
            specific_optional=[
                InterventionField("specimen_type", False, "string",
                                validation_rules={"allowed_values": ["blood", "urine", "stool", "saliva", "breath", "tissue", "other"]},
                                description="Type of specimen collected"),
                InterventionField("test_method", False, "string",
                                description="Specific testing methodology or technique"),
                InterventionField("biomarkers", False, "list",
                                description="Specific biomarkers or analytes measured"),
                InterventionField("diagnostic_purpose", False, "string",
                                description="Primary purpose of the test (screening, diagnosis, monitoring)"),
                InterventionField("test_accuracy", False, "string",
                                description="Reported accuracy, sensitivity, or specificity"),
                InterventionField("reference_range", False, "string",
                                description="Normal or reference range for test results")
            ]
        )

        # EMERGING Category
        categories[InterventionType.EMERGING] = CategoryBuilder.create_category(
            category_type=InterventionType.EMERGING,
            display_name="Emerging Interventions",
            description="New or novel intervention types that don't fit existing categories",
            subcategories=[
                "digital_health", "biotechnology", "nanotechnology", "gene_therapy", "precision_medicine",
                "virtual_reality", "artificial_intelligence", "biomarker_guided", "personalized", "novel"
            ],
            specific_required=[
                InterventionField("proposed_category", True, "string",
                                description="Suggested new category name for this intervention"),
                InterventionField("category_rationale", True, "string",
                                description="Explanation for why this needs a new category")
            ],
            specific_optional=[
                InterventionField("novelty_description", False, "string",
                                description="Description of what makes this intervention novel"),
                InterventionField("similarity_to_existing", False, "string",
                                description="How this relates to existing categories"),
                InterventionField("potential_impact", False, "string",
                                description="Potential clinical or research impact")
            ]
        )

        return categories
    
    def get_category(self, category: InterventionType) -> InterventionCategory:
        """Get category definition by type."""
        return self.categories[category]
    
    def get_all_categories(self) -> Dict[InterventionType, InterventionCategory]:
        """Get all category definitions."""
        return self.categories
    
    def get_category_names(self) -> List[str]:
        """Get list of all category names."""
        return [cat.value for cat in InterventionType]
    
    def validate_category(self, category_name: str) -> bool:
        """Validate if a category name is supported."""
        try:
            InterventionType(category_name)
            return True
        except ValueError:
            return False
    
    def get_subcategories(self, category: InterventionType) -> List[str]:
        """Get subcategories for a specific intervention type."""
        return self.categories[category].subcategories
    
    def find_category_by_subcategory(self, subcategory: str) -> Optional[InterventionType]:
        """Find the main category that contains a given subcategory."""
        subcategory_lower = subcategory.lower()
        for cat_type, category in self.categories.items():
            if subcategory_lower in [sub.lower() for sub in category.subcategories]:
                return cat_type
        return None

    def get_intervention_hierarchy(self, intervention_category: str,
                                 intervention_subcategory: str = None) -> Dict[str, Any]:
        """Get hierarchical information for an intervention."""
        try:
            category_type = InterventionType(intervention_category)
            category_def = self.get_category(category_type)

            hierarchy = {
                'category': {
                    'type': category_type.value,
                    'display_name': category_def.display_name,
                    'description': category_def.description
                },
                'subcategory': None,
                'related_subcategories': category_def.subcategories
            }

            if intervention_subcategory:
                # Validate subcategory exists
                if intervention_subcategory.lower() in [sub.lower() for sub in category_def.subcategories]:
                    hierarchy['subcategory'] = {
                        'name': intervention_subcategory,
                        'parent_category': category_type.value
                    }

            return hierarchy

        except ValueError:
            return {}

    def get_related_interventions(self, intervention_category: str,
                                intervention_subcategory: str = None) -> List[str]:
        """Get related intervention categories/subcategories."""
        try:
            category_type = InterventionType(intervention_category)
            category_def = self.get_category(category_type)

            related = []

            # Add all subcategories from the same category
            if intervention_subcategory:
                related.extend([
                    sub for sub in category_def.subcategories
                    if sub.lower() != intervention_subcategory.lower()
                ])
            else:
                related.extend(category_def.subcategories)

            return related

        except ValueError:
            return []

    def suggest_intervention_category(self, intervention_name: str) -> Optional[InterventionType]:
        """Suggest the most appropriate category for an intervention name."""
        intervention_lower = intervention_name.lower()

        # Simple keyword-based suggestion
        category_keywords = {
            InterventionType.EXERCISE: ['exercise', 'training', 'activity', 'workout', 'fitness', 'aerobic', 'resistance'],
            InterventionType.DIET: ['diet', 'nutrition', 'food', 'eating', 'caloric', 'macronutrient'],
            InterventionType.SUPPLEMENT: ['supplement', 'vitamin', 'mineral', 'probiotic', 'omega'],
            InterventionType.MEDICATION: ['medication', 'drug', 'pharmaceutical', 'prescription', 'pill'],
            InterventionType.THERAPY: ['therapy', 'counseling', 'treatment', 'psychotherapy', 'cbt'],
            InterventionType.LIFESTYLE: ['lifestyle', 'behavioral', 'stress', 'sleep', 'smoking'],
            InterventionType.SURGERY: ['surgery', 'surgical', 'operation', 'procedure', 'laparoscopic'],
            InterventionType.TEST: ['test', 'testing', 'diagnostic', 'blood test', 'breath test', 'stool test', 'biomarker', 'screening']
        }

        for category, keywords in category_keywords.items():
            if any(keyword in intervention_lower for keyword in keywords):
                return category

        return None


# Global instance
intervention_taxonomy = InterventionTaxonomy()