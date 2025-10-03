"""
Condition taxonomy and categorization system.
Defines the structure and hierarchy of health conditions.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Any
from enum import Enum


class ConditionType(Enum):
    """Primary condition categories."""
    CARDIAC = "cardiac"
    NEUROLOGICAL = "neurological"
    DIGESTIVE = "digestive"
    PULMONARY = "pulmonary"
    ENDOCRINE = "endocrine"
    RENAL = "renal"
    ONCOLOGICAL = "oncological"
    RHEUMATOLOGICAL = "rheumatological"
    PSYCHIATRIC = "psychiatric"
    MUSCULOSKELETAL = "musculoskeletal"
    DERMATOLOGICAL = "dermatological"
    INFECTIOUS = "infectious"
    IMMUNOLOGICAL = "immunological"
    HEMATOLOGICAL = "hematological"
    NUTRITIONAL = "nutritional"
    TOXICOLOGICAL = "toxicological"
    PARASITIC = "parasitic"
    OTHER = "other"


@dataclass
class ConditionCategory:
    """Defines a complete condition category with its rules and structure."""

    category: ConditionType
    display_name: str
    description: str
    examples: List[str] = field(default_factory=list)
    related_specialties: List[str] = field(default_factory=list)


class ConditionTaxonomy:
    """Complete taxonomy of health conditions."""

    def __init__(self):
        self.categories = self._build_taxonomy()

    def _build_taxonomy(self) -> Dict[ConditionType, ConditionCategory]:
        """Build the complete condition taxonomy."""

        categories = {}

        # CARDIAC Category
        categories[ConditionType.CARDIAC] = ConditionCategory(
            category=ConditionType.CARDIAC,
            display_name="Cardiovascular Conditions",
            description="Conditions affecting the heart and blood vessels including coronary artery disease, heart failure, hypertension, arrhythmias, and vascular diseases",
            examples=[
                "coronary artery disease", "heart failure", "hypertension",
                "atrial fibrillation", "myocardial infarction", "STEMI",
                "angina", "cardiomyopathy", "valvular heart disease"
            ],
            related_specialties=["cardiology"]
        )

        # NEUROLOGICAL Category
        categories[ConditionType.NEUROLOGICAL] = ConditionCategory(
            category=ConditionType.NEUROLOGICAL,
            display_name="Neurological Conditions",
            description="Conditions affecting the brain, spinal cord, and nervous system including neurodegenerative diseases, stroke, epilepsy, and cognitive disorders",
            examples=[
                "stroke", "Alzheimer's disease", "Parkinson's disease",
                "epilepsy", "multiple sclerosis", "ADHD",
                "cognitive impairment", "dementia", "neuropathy"
            ],
            related_specialties=["neurology"]
        )

        # DIGESTIVE Category
        categories[ConditionType.DIGESTIVE] = ConditionCategory(
            category=ConditionType.DIGESTIVE,
            display_name="Digestive & Gastrointestinal",
            description="Conditions affecting the digestive system including esophagus, stomach, intestines, liver, and pancreas",
            examples=[
                "gastroesophageal reflux disease", "inflammatory bowel disease",
                "irritable bowel syndrome", "cirrhosis", "peptic ulcer disease",
                "Crohn's disease", "ulcerative colitis", "H. pylori infection",
                "Helicobacter pylori infection"
            ],
            related_specialties=["gastroenterology"]
        )

        # PULMONARY Category
        categories[ConditionType.PULMONARY] = ConditionCategory(
            category=ConditionType.PULMONARY,
            display_name="Respiratory & Pulmonary",
            description="Conditions affecting the lungs and respiratory system including obstructive lung diseases, infections, and respiratory failure",
            examples=[
                "chronic obstructive pulmonary disease", "COPD", "asthma",
                "pneumonia", "pulmonary embolism", "COPD-PH",
                "respiratory failure", "bronchitis"
            ],
            related_specialties=["pulmonology"]
        )

        # ENDOCRINE Category
        categories[ConditionType.ENDOCRINE] = ConditionCategory(
            category=ConditionType.ENDOCRINE,
            display_name="Endocrine & Metabolic",
            description="Conditions affecting hormones and metabolism including diabetes, thyroid disorders, and metabolic syndrome",
            examples=[
                "diabetes mellitus", "type 2 diabetes", "thyroid disorders",
                "obesity", "osteoporosis", "polycystic ovary syndrome", "PCOS",
                "metabolic syndrome", "hyperthyroidism", "hypothyroidism"
            ],
            related_specialties=["endocrinology"]
        )

        # RENAL Category
        categories[ConditionType.RENAL] = ConditionCategory(
            category=ConditionType.RENAL,
            display_name="Kidney & Urinary",
            description="Conditions affecting the kidneys and urinary system including kidney disease, stones, and glomerular disorders",
            examples=[
                "chronic kidney disease", "acute kidney injury",
                "kidney stones", "glomerulonephritis", "C3 glomerulopathy",
                "polycystic kidney disease", "nephrotic syndrome"
            ],
            related_specialties=["nephrology"]
        )

        # ONCOLOGICAL Category
        categories[ConditionType.ONCOLOGICAL] = ConditionCategory(
            category=ConditionType.ONCOLOGICAL,
            display_name="Cancer & Oncology",
            description="Malignant neoplasms and cancers affecting any organ system",
            examples=[
                "lung cancer", "breast cancer", "colorectal cancer",
                "prostate cancer", "leukemia", "HER2-mutant metastatic non-small cell lung cancer",
                "metastatic colorectal cancer", "mCRC"
            ],
            related_specialties=["oncology"]
        )

        # RHEUMATOLOGICAL Category
        categories[ConditionType.RHEUMATOLOGICAL] = ConditionCategory(
            category=ConditionType.RHEUMATOLOGICAL,
            display_name="Autoimmune & Rheumatic",
            description="Autoimmune conditions and rheumatic diseases affecting joints, muscles, and connective tissue",
            examples=[
                "rheumatoid arthritis", "osteoarthritis",
                "systemic lupus erythematosus", "gout", "fibromyalgia",
                "Giant Cell Arteritis", "GCA", "vasculitis",
                "large vessel vasculitis", "ankylosing spondylitis"
            ],
            related_specialties=["rheumatology"]
        )

        # PSYCHIATRIC Category
        categories[ConditionType.PSYCHIATRIC] = ConditionCategory(
            category=ConditionType.PSYCHIATRIC,
            display_name="Mental Health",
            description="Mental health conditions and psychiatric disorders",
            examples=[
                "major depressive disorder", "MDD", "anxiety disorders",
                "bipolar disorder", "schizophrenia", "depression",
                "attention deficit hyperactivity disorder", "PTSD"
            ],
            related_specialties=["psychiatry"]
        )

        # MUSCULOSKELETAL Category
        categories[ConditionType.MUSCULOSKELETAL] = ConditionCategory(
            category=ConditionType.MUSCULOSKELETAL,
            display_name="Musculoskeletal & Orthopedic",
            description="Conditions affecting bones, muscles, tendons, and ligaments",
            examples=[
                "fractures", "osteoarthritis", "back pain",
                "rotator cuff tear", "anterior cruciate ligament injury",
                "ACL injury", "tendinitis", "osteoporosis"
            ],
            related_specialties=["orthopedics"]
        )

        # DERMATOLOGICAL Category
        categories[ConditionType.DERMATOLOGICAL] = ConditionCategory(
            category=ConditionType.DERMATOLOGICAL,
            display_name="Skin & Dermatological",
            description="Conditions affecting the skin, hair, and nails",
            examples=[
                "acne vulgaris", "atopic dermatitis", "psoriasis",
                "skin cancer", "rosacea", "eczema", "melanoma"
            ],
            related_specialties=["dermatology"]
        )

        # INFECTIOUS Category
        categories[ConditionType.INFECTIOUS] = ConditionCategory(
            category=ConditionType.INFECTIOUS,
            display_name="Infectious Diseases",
            description="Bacterial, viral, and fungal infections (excluding parasitic)",
            examples=[
                "human immunodeficiency virus", "HIV", "HIV-1 infection",
                "tuberculosis", "hepatitis B", "sepsis", "influenza",
                "COVID-19", "SARS-CoV-2 infection", "pneumonia"
            ],
            related_specialties=["infectious_disease"]
        )

        # IMMUNOLOGICAL Category
        categories[ConditionType.IMMUNOLOGICAL] = ConditionCategory(
            category=ConditionType.IMMUNOLOGICAL,
            display_name="Allergic & Immunological",
            description="Allergies, hypersensitivity reactions, and immune system disorders",
            examples=[
                "IgE-mediated hen's egg allergy", "food allergies",
                "allergic rhinitis", "immunodeficiency", "anaphylaxis",
                "hypersensitivity reactions"
            ],
            related_specialties=[]
        )

        # HEMATOLOGICAL Category
        categories[ConditionType.HEMATOLOGICAL] = ConditionCategory(
            category=ConditionType.HEMATOLOGICAL,
            display_name="Blood & Hematological",
            description="Conditions affecting blood cells, clotting, and hematopoietic system",
            examples=[
                "anemia", "thrombocytopenia", "hemophilia",
                "sickle cell disease", "clotting disorders",
                "thrombosis", "bleeding disorders"
            ],
            related_specialties=["hematology"]
        )

        # NUTRITIONAL Category
        categories[ConditionType.NUTRITIONAL] = ConditionCategory(
            category=ConditionType.NUTRITIONAL,
            display_name="Nutritional Deficiencies",
            description="Conditions caused by nutrient deficiencies or malnutrition",
            examples=[
                "vitamin D deficiency", "vitamin B12 deficiency",
                "iron deficiency", "malnutrition", "scurvy",
                "kwashiorkor", "rickets"
            ],
            related_specialties=[]
        )

        # TOXICOLOGICAL Category
        categories[ConditionType.TOXICOLOGICAL] = ConditionCategory(
            category=ConditionType.TOXICOLOGICAL,
            display_name="Poisoning & Toxicity",
            description="Conditions caused by poisoning, toxins, or drug toxicity",
            examples=[
                "drug toxicity", "heavy metal poisoning",
                "carbon monoxide poisoning", "overdose",
                "medication toxicity", "lead poisoning"
            ],
            related_specialties=["toxicology"]
        )

        # PARASITIC Category
        categories[ConditionType.PARASITIC] = ConditionCategory(
            category=ConditionType.PARASITIC,
            display_name="Parasitic Infections",
            description="Conditions caused by parasitic organisms",
            examples=[
                "malaria", "toxoplasmosis", "helminth infections",
                "giardiasis", "schistosomiasis", "tapeworm infection"
            ],
            related_specialties=[]
        )

        # OTHER Category
        categories[ConditionType.OTHER] = ConditionCategory(
            category=ConditionType.OTHER,
            display_name="Other/Uncategorized",
            description="Conditions that don't fit into standard categories or are multisystem",
            examples=[
                "rare diseases", "multisystem conditions", "unclassified syndromes"
            ],
            related_specialties=[]
        )

        return categories

    def get_category(self, category: ConditionType) -> ConditionCategory:
        """Get category definition by type."""
        return self.categories[category]

    def get_all_categories(self) -> Dict[ConditionType, ConditionCategory]:
        """Get all category definitions."""
        return self.categories

    def get_category_by_name(self, category_name: str) -> Optional[ConditionCategory]:
        """Get category definition by string name."""
        try:
            category_type = ConditionType(category_name.lower())
            return self.categories[category_type]
        except (ValueError, KeyError):
            return None


# Global instance
condition_taxonomy = ConditionTaxonomy()
