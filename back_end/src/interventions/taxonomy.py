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
    DEVICE = "device"
    PROCEDURE = "procedure"
    BIOLOGICS = "biologics"
    GENE_THERAPY = "gene_therapy"
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
            description="Physical exercise interventions including aerobic, resistance, flexibility training, yoga, walking, swimming, and sports",
            subcategories=[],
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
            description="Dietary interventions including Mediterranean diet, ketogenic diet, intermittent fasting, caloric restriction, and specific foods",
            subcategories=[],
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
            description="Nutritional supplements including vitamins, minerals, herbs, probiotics, omega-3, and other nutraceuticals",
            subcategories=[],
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
            description="Pharmaceutical drugs including prescription and over-the-counter medications (small molecule drugs)",
            subcategories=[],
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
            description="Therapeutic interventions including psychological therapy (CBT, psychotherapy), physical therapy, occupational therapy, and behavioral therapies",
            subcategories=[],
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
            description="Lifestyle interventions including sleep hygiene, stress management, smoking cessation, alcohol reduction, and behavioral changes",
            subcategories=[],
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
            description="Surgical procedures including minimally invasive surgery, open surgery, laparoscopic, bariatric, cardiac, and transplant operations",
            subcategories=[],
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
            description="Medical tests and diagnostic procedures including blood tests, genetic testing, imaging, biomarker analysis, and screening",
            subcategories=[],
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

        # DEVICE Category
        categories[InterventionType.DEVICE] = CategoryBuilder.create_category(
            category_type=InterventionType.DEVICE,
            display_name="Medical Devices & Implants",
            description="Medical devices, implants, wearables, and monitoring tools including pacemakers, insulin pumps, CPAP machines, and hearing aids",
            subcategories=[],
            specific_required=[
                InterventionField("device_type", True, "string",
                                description="Type of medical device")
            ],
            specific_optional=[
                InterventionField("device_model", False, "string",
                                description="Specific device model or brand"),
                InterventionField("implantation_method", False, "string",
                                description="Method of implantation or attachment"),
                InterventionField("monitoring_frequency", False, "string",
                                description="Frequency of monitoring or data collection"),
                InterventionField("battery_life", False, "string",
                                description="Battery life or replacement schedule"),
                InterventionField("maintenance_requirements", False, "string",
                                description="Maintenance or calibration requirements")
            ]
        )

        # PROCEDURE Category
        categories[InterventionType.PROCEDURE] = CategoryBuilder.create_category(
            category_type=InterventionType.PROCEDURE,
            display_name="Medical Procedures",
            description="Non-surgical medical procedures including endoscopy, dialysis, blood transfusion, radiation therapy, and colonoscopy",
            subcategories=[],
            specific_required=[
                InterventionField("procedure_type", True, "string",
                                description="Type of medical procedure")
            ],
            specific_optional=[
                InterventionField("procedure_duration", False, "string",
                                description="Duration of the procedure"),
                InterventionField("anesthesia_type", False, "string",
                                validation_rules={"allowed_values": ["none", "local", "conscious_sedation", "general"]},
                                description="Type of anesthesia used"),
                InterventionField("recovery_time", False, "string",
                                description="Expected recovery time"),
                InterventionField("procedure_frequency", False, "string",
                                description="Frequency of procedure repetition"),
                InterventionField("preparation_requirements", False, "string",
                                description="Preparation required before procedure")
            ]
        )

        # BIOLOGICS Category
        categories[InterventionType.BIOLOGICS] = CategoryBuilder.create_category(
            category_type=InterventionType.BIOLOGICS,
            display_name="Biological Medicines",
            description="Biological medicines including monoclonal antibodies, vaccines, immunotherapies, insulin, and other biological drugs",
            subcategories=[],
            specific_required=[
                InterventionField("biologic_type", True, "string",
                                description="Type of biological medicine (antibody, vaccine, etc.)")
            ],
            specific_optional=[
                InterventionField("target_antigen", False, "string",
                                description="Target antigen or receptor"),
                InterventionField("dosage", False, "string",
                                description="Dosage amount and units"),
                InterventionField("route", False, "string",
                                validation_rules={"allowed_values": ["iv", "subcutaneous", "im", "oral", "inhaled"]},
                                description="Route of administration"),
                InterventionField("frequency", False, "string",
                                description="Dosing frequency"),
                InterventionField("immunogenicity_risk", False, "string",
                                description="Risk of immune response to biologic")
            ]
        )

        # GENE_THERAPY Category
        categories[InterventionType.GENE_THERAPY] = CategoryBuilder.create_category(
            category_type=InterventionType.GENE_THERAPY,
            display_name="Gene & Cellular Therapy",
            description="Genetic and cellular interventions including CRISPR gene editing, CAR-T cell therapy, stem cell therapy, and gene transfer",
            subcategories=[],
            specific_required=[
                InterventionField("therapy_type", True, "string",
                                description="Type of gene/cellular therapy")
            ],
            specific_optional=[
                InterventionField("target_gene", False, "string",
                                description="Target gene or genetic pathway"),
                InterventionField("vector_type", False, "string",
                                description="Vector used for gene delivery (viral, non-viral, etc.)"),
                InterventionField("cell_source", False, "string",
                                description="Source of cells (autologous, allogeneic, etc.)"),
                InterventionField("modification_method", False, "string",
                                description="Method of genetic modification"),
                InterventionField("administration_route", False, "string",
                                description="Route of therapy administration")
            ]
        )

        # EMERGING Category
        categories[InterventionType.EMERGING] = CategoryBuilder.create_category(
            category_type=InterventionType.EMERGING,
            display_name="Emerging Interventions",
            description="New or novel intervention types that don't fit existing categories",
            subcategories=[],
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


# Global instance
intervention_taxonomy = InterventionTaxonomy()