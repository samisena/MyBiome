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


class InterventionTaxonomy:
    """Complete taxonomy of health interventions."""
    
    def __init__(self):
        self.categories = self._build_taxonomy()
    
    def _build_taxonomy(self) -> Dict[InterventionType, InterventionCategory]:
        """Build the complete intervention taxonomy."""
        
        categories = {}
        
        # EXERCISE Category
        categories[InterventionType.EXERCISE] = InterventionCategory(
            category=InterventionType.EXERCISE,
            display_name="Exercise & Physical Activity",
            description="Physical exercise interventions including aerobic, resistance, and flexibility training",
            subcategories=[
                "aerobic", "resistance", "flexibility", "balance", "hiit", 
                "yoga", "pilates", "sports", "walking", "cycling", "swimming"
            ],
            required_fields=[
                InterventionField("exercise_type", True, "string", 
                                description="Type of exercise (aerobic, resistance, etc.)"),
                InterventionField("intervention_name", True, "string",
                                description="Specific name of the exercise intervention")
            ],
            optional_fields=[
                InterventionField("duration", False, "string", 
                                description="Duration per session (e.g., '30 minutes')"),
                InterventionField("frequency", False, "string",
                                description="Frequency per week (e.g., '3x per week')"),
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
        categories[InterventionType.DIET] = InterventionCategory(
            category=InterventionType.DIET,
            display_name="Diet & Nutrition",
            description="Dietary interventions including specific diets, foods, and nutritional modifications",
            subcategories=[
                "mediterranean", "dash", "ketogenic", "low_carb", "low_fat", "vegetarian", "vegan",
                "intermittent_fasting", "caloric_restriction", "specific_foods", "macronutrient_modification"
            ],
            required_fields=[
                InterventionField("diet_type", True, "string",
                                description="Type of dietary intervention"),
                InterventionField("intervention_name", True, "string",
                                description="Specific name of the dietary intervention")
            ],
            optional_fields=[
                InterventionField("duration", False, "string",
                                description="Duration of dietary intervention"),
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
        categories[InterventionType.SUPPLEMENT] = InterventionCategory(
            category=InterventionType.SUPPLEMENT,
            display_name="Supplements & Nutraceuticals", 
            description="Nutritional supplements including vitamins, minerals, herbs, and probiotics",
            subcategories=[
                "vitamin", "mineral", "herbal", "probiotic", "prebiotic", "omega3", 
                "amino_acid", "protein", "fiber", "antioxidant", "botanical"
            ],
            required_fields=[
                InterventionField("supplement_name", True, "string",
                                description="Specific supplement name"),
                InterventionField("intervention_name", True, "string",
                                description="Common name of the supplement intervention")
            ],
            optional_fields=[
                InterventionField("dosage", False, "string",
                                description="Dosage amount and units"),
                InterventionField("duration", False, "string",
                                description="Duration of supplementation"),
                InterventionField("frequency", False, "string",
                                description="Frequency of administration"),
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


# Global instance
intervention_taxonomy = InterventionTaxonomy()