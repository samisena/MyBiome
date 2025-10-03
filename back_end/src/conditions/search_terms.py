"""
Search term definitions for condition categories.
Provides PubMed search optimization terms for each condition category.
"""

from typing import List, Dict
from back_end.src.conditions.taxonomy import ConditionType


class ConditionSearchTerms:
    """Manages search terms for condition categories."""

    def __init__(self):
        self.search_terms = self._build_search_terms()

    def _build_search_terms(self) -> Dict[ConditionType, List[str]]:
        """Build search terms for each condition category."""

        terms = {}

        # CARDIAC
        terms[ConditionType.CARDIAC] = [
            "cardiovascular disease", "heart disease", "coronary artery disease",
            "heart failure", "hypertension", "myocardial infarction",
            "atrial fibrillation", "arrhythmia", "angina", "cardiomyopathy"
        ]

        # NEUROLOGICAL
        terms[ConditionType.NEUROLOGICAL] = [
            "neurological disorder", "stroke", "Alzheimer disease",
            "Parkinson disease", "epilepsy", "multiple sclerosis",
            "dementia", "cognitive impairment", "neuropathy"
        ]

        # DIGESTIVE
        terms[ConditionType.DIGESTIVE] = [
            "gastrointestinal disease", "digestive disorder", "GERD",
            "inflammatory bowel disease", "IBD", "irritable bowel syndrome",
            "cirrhosis", "peptic ulcer", "Crohn disease", "ulcerative colitis"
        ]

        # PULMONARY
        terms[ConditionType.PULMONARY] = [
            "respiratory disease", "pulmonary disease", "COPD",
            "chronic obstructive pulmonary disease", "asthma",
            "pneumonia", "pulmonary embolism", "lung disease"
        ]

        # ENDOCRINE
        terms[ConditionType.ENDOCRINE] = [
            "diabetes mellitus", "type 2 diabetes", "thyroid disorder",
            "obesity", "metabolic syndrome", "osteoporosis",
            "polycystic ovary syndrome", "PCOS", "endocrine disorder"
        ]

        # RENAL
        terms[ConditionType.RENAL] = [
            "chronic kidney disease", "renal disease", "kidney disease",
            "acute kidney injury", "kidney stones", "glomerulonephritis",
            "nephropathy", "renal failure"
        ]

        # ONCOLOGICAL
        terms[ConditionType.ONCOLOGICAL] = [
            "cancer", "neoplasm", "carcinoma", "tumor", "malignancy",
            "lung cancer", "breast cancer", "colorectal cancer",
            "prostate cancer", "leukemia", "lymphoma"
        ]

        # RHEUMATOLOGICAL
        terms[ConditionType.RHEUMATOLOGICAL] = [
            "rheumatoid arthritis", "autoimmune disease", "lupus",
            "systemic lupus erythematosus", "osteoarthritis", "gout",
            "vasculitis", "fibromyalgia", "rheumatic disease"
        ]

        # PSYCHIATRIC
        terms[ConditionType.PSYCHIATRIC] = [
            "depression", "major depressive disorder", "anxiety disorder",
            "bipolar disorder", "schizophrenia", "ADHD",
            "attention deficit hyperactivity disorder", "PTSD",
            "mental health", "psychiatric disorder"
        ]

        # MUSCULOSKELETAL
        terms[ConditionType.MUSCULOSKELETAL] = [
            "musculoskeletal disorder", "osteoarthritis", "fracture",
            "back pain", "joint disease", "tendinitis",
            "ligament injury", "bone disease"
        ]

        # DERMATOLOGICAL
        terms[ConditionType.DERMATOLOGICAL] = [
            "skin disease", "dermatological disorder", "acne",
            "psoriasis", "eczema", "atopic dermatitis",
            "skin cancer", "melanoma", "rosacea"
        ]

        # INFECTIOUS
        terms[ConditionType.INFECTIOUS] = [
            "infection", "infectious disease", "HIV", "tuberculosis",
            "hepatitis", "sepsis", "influenza", "COVID-19",
            "bacterial infection", "viral infection"
        ]

        # IMMUNOLOGICAL
        terms[ConditionType.IMMUNOLOGICAL] = [
            "allergy", "allergic disease", "hypersensitivity",
            "food allergy", "immunodeficiency", "immune disorder",
            "anaphylaxis", "allergic rhinitis"
        ]

        # HEMATOLOGICAL
        terms[ConditionType.HEMATOLOGICAL] = [
            "hematological disorder", "anemia", "blood disorder",
            "thrombocytopenia", "hemophilia", "sickle cell disease",
            "clotting disorder", "thrombosis"
        ]

        # NUTRITIONAL
        terms[ConditionType.NUTRITIONAL] = [
            "nutritional deficiency", "vitamin deficiency",
            "malnutrition", "iron deficiency", "vitamin D deficiency",
            "vitamin B12 deficiency", "scurvy"
        ]

        # TOXICOLOGICAL
        terms[ConditionType.TOXICOLOGICAL] = [
            "poisoning", "toxicity", "drug toxicity",
            "heavy metal poisoning", "overdose",
            "toxic exposure", "intoxication"
        ]

        # PARASITIC
        terms[ConditionType.PARASITIC] = [
            "parasitic infection", "malaria", "helminth infection",
            "toxoplasmosis", "giardiasis", "parasite"
        ]

        # OTHER
        terms[ConditionType.OTHER] = [
            "rare disease", "syndrome", "disorder"
        ]

        return terms

    def get_terms_for_category(self, category: ConditionType) -> List[str]:
        """
        Get search terms for a specific condition category.

        Args:
            category: The condition category

        Returns:
            List of search terms for PubMed queries
        """
        return self.search_terms.get(category, [])

    def get_all_terms(self) -> Dict[ConditionType, List[str]]:
        """Get all search terms for all categories."""
        return self.search_terms


# Global instance
search_terms = ConditionSearchTerms()
