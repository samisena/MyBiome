"""
Search terms library for different intervention categories.
Used for building PubMed queries and identifying relevant papers.
"""

from typing import Dict, List, Set
from src.interventions.taxonomy import InterventionType


class InterventionSearchTerms:
    """Comprehensive search terms for each intervention category."""
    
    def __init__(self):
        self.search_terms = self._build_search_terms()
    
    def _build_search_terms(self) -> Dict[InterventionType, Dict[str, List[str]]]:
        """Build comprehensive search terms for each intervention category."""
        
        terms = {}
        
        # EXERCISE terms
        terms[InterventionType.EXERCISE] = {
            'primary': [
                'exercise[Title/Abstract]',
                'physical activity[Title/Abstract]', 
                'training[Title/Abstract]',
                '"Exercise"[MeSH Terms]',
                '"Physical Fitness"[MeSH Terms]',
                '"Exercise Therapy"[MeSH Terms]'
            ],
            'specific': [
                'aerobic exercise[Title/Abstract]',
                'resistance training[Title/Abstract]',
                'strength training[Title/Abstract]',
                'cardiovascular exercise[Title/Abstract]',
                'endurance training[Title/Abstract]',
                'weight training[Title/Abstract]',
                'yoga[Title/Abstract]',
                'pilates[Title/Abstract]',
                'walking[Title/Abstract]',
                'running[Title/Abstract]',
                'cycling[Title/Abstract]',
                'swimming[Title/Abstract]',
                'HIIT[Title/Abstract]',
                '"high intensity interval training"[Title/Abstract]',
                'flexibility training[Title/Abstract]',
                'balance training[Title/Abstract]'
            ]
        }
        
        # DIET terms
        terms[InterventionType.DIET] = {
            'primary': [
                'diet[Title/Abstract]',
                'dietary intervention[Title/Abstract]',
                'nutrition[Title/Abstract]',
                '"Diet"[MeSH Terms]',
                '"Nutritional Sciences"[MeSH Terms]',
                '"Diet Therapy"[MeSH Terms]'
            ],
            'specific': [
                'mediterranean diet[Title/Abstract]',
                'DASH diet[Title/Abstract]',
                'ketogenic diet[Title/Abstract]',
                'low carb diet[Title/Abstract]',
                'low fat diet[Title/Abstract]',
                'vegetarian diet[Title/Abstract]',
                'vegan diet[Title/Abstract]',
                'intermittent fasting[Title/Abstract]',
                'caloric restriction[Title/Abstract]',
                'dietary pattern[Title/Abstract]',
                'nutritional intervention[Title/Abstract]',
                'food intervention[Title/Abstract]',
                'macronutrient[Title/Abstract]',
                'micronutrient[Title/Abstract]'
            ]
        }
        
        # SUPPLEMENT terms  
        terms[InterventionType.SUPPLEMENT] = {
            'primary': [
                'supplement[Title/Abstract]',
                'supplementation[Title/Abstract]',
                'nutraceutical[Title/Abstract]',
                '"Dietary Supplements"[MeSH Terms]',
                '"Vitamins"[MeSH Terms]',
                '"Minerals"[MeSH Terms]'
            ],
            'specific': [
                'vitamin[Title/Abstract]',
                'mineral supplement[Title/Abstract]',
                'herbal supplement[Title/Abstract]',
                'probiotic[Title/Abstract]',
                'prebiotic[Title/Abstract]',
                'omega-3[Title/Abstract]',
                'fish oil[Title/Abstract]',
                'amino acid[Title/Abstract]',
                'protein supplement[Title/Abstract]',
                'fiber supplement[Title/Abstract]',
                'antioxidant[Title/Abstract]',
                'botanical extract[Title/Abstract]',
                'lactobacillus[Title/Abstract]',
                'bifidobacterium[Title/Abstract]',
                '"ascorbic acid"[Title/Abstract]',
                'vitamin D[Title/Abstract]',
                'vitamin B[Title/Abstract]',
                'calcium[Title/Abstract]',
                'magnesium[Title/Abstract]',
                'zinc[Title/Abstract]',
                'iron[Title/Abstract]'
            ]
        }
        
        # MEDICATION terms
        terms[InterventionType.MEDICATION] = {
            'primary': [
                'medication[Title/Abstract]',
                'drug therapy[Title/Abstract]',
                'pharmaceutical[Title/Abstract]',
                'pharmacological[Title/Abstract]',
                '"Drug Therapy"[MeSH Terms]',
                '"Pharmaceutical Preparations"[MeSH Terms]'
            ],
            'specific': [
                'antidepressant[Title/Abstract]',
                'SSRI[Title/Abstract]',
                'anxiolytic[Title/Abstract]',
                'benzodiazepine[Title/Abstract]',
                'antipsychotic[Title/Abstract]',
                'antibiotic[Title/Abstract]',
                'anti-inflammatory[Title/Abstract]',
                'NSAID[Title/Abstract]',
                'antihypertensive[Title/Abstract]',
                'ACE inhibitor[Title/Abstract]',
                'beta blocker[Title/Abstract]',
                'diabetes medication[Title/Abstract]',
                'metformin[Title/Abstract]',
                'insulin[Title/Abstract]',
                'hormone therapy[Title/Abstract]',
                'immunosuppressant[Title/Abstract]',
                'corticosteroid[Title/Abstract]'
            ]
        }
        
        # THERAPY terms
        terms[InterventionType.THERAPY] = {
            'primary': [
                'therapy[Title/Abstract]',
                'treatment[Title/Abstract]',
                'intervention[Title/Abstract]',
                'counseling[Title/Abstract]',
                '"Psychotherapy"[MeSH Terms]',
                '"Behavioral Therapy"[MeSH Terms]',
                '"Physical Therapy Modalities"[MeSH Terms]'
            ],
            'specific': [
                'cognitive behavioral therapy[Title/Abstract]',
                'CBT[Title/Abstract]',
                'dialectical behavior therapy[Title/Abstract]',
                'DBT[Title/Abstract]',
                'acceptance commitment therapy[Title/Abstract]',
                'ACT[Title/Abstract]',
                'mindfulness[Title/Abstract]',
                'meditation[Title/Abstract]',
                'psychotherapy[Title/Abstract]',
                'group therapy[Title/Abstract]',
                'individual therapy[Title/Abstract]',
                'physical therapy[Title/Abstract]',
                'physiotherapy[Title/Abstract]',
                'occupational therapy[Title/Abstract]',
                'massage therapy[Title/Abstract]',
                'acupuncture[Title/Abstract]',
                'chiropractic[Title/Abstract]',
                'rehabilitation[Title/Abstract]'
            ]
        }
        
        # LIFESTYLE terms
        terms[InterventionType.LIFESTYLE] = {
            'primary': [
                'lifestyle intervention[Title/Abstract]',
                'lifestyle modification[Title/Abstract]',
                'behavioral intervention[Title/Abstract]',
                'behavior change[Title/Abstract]',
                '"Life Style"[MeSH Terms]',
                '"Behavior Therapy"[MeSH Terms]'
            ],
            'specific': [
                'sleep hygiene[Title/Abstract]',
                'sleep intervention[Title/Abstract]',
                'stress management[Title/Abstract]',
                'stress reduction[Title/Abstract]',
                'relaxation[Title/Abstract]',
                'smoking cessation[Title/Abstract]',
                'alcohol reduction[Title/Abstract]',
                'social support[Title/Abstract]',
                'peer support[Title/Abstract]',
                'environmental modification[Title/Abstract]',
                'time management[Title/Abstract]',
                'self-care[Title/Abstract]',
                'wellness program[Title/Abstract]',
                'health promotion[Title/Abstract]',
                'behavior modification[Title/Abstract]'
            ]
        }
        
        return terms
    
    def get_terms_for_category(self, category: InterventionType, 
                              include_specific: bool = True) -> List[str]:
        """Get search terms for a specific intervention category."""
        if category not in self.search_terms:
            return []
            
        terms = self.search_terms[category]['primary'].copy()
        if include_specific:
            terms.extend(self.search_terms[category]['specific'])
        return terms
    
    def get_all_intervention_terms(self) -> List[str]:
        """Get all intervention terms across all categories."""
        all_terms = []
        for category_terms in self.search_terms.values():
            all_terms.extend(category_terms['primary'])
            all_terms.extend(category_terms['specific'])
        return list(set(all_terms))  # Remove duplicates
    
    def build_intervention_query_part(self, categories: List[InterventionType] = None) -> str:
        """Build the intervention part of a PubMed query."""
        if not categories:
            categories = list(InterventionType)
        
        all_terms = []
        for category in categories:
            terms = self.get_terms_for_category(category, include_specific=True)
            all_terms.extend(terms)
        
        # Remove duplicates and join with OR
        unique_terms = list(set(all_terms))
        return '(' + ' OR '.join(unique_terms) + ')'
    
    def get_study_type_terms(self) -> List[str]:
        """Get terms for filtering to intervention studies."""
        return [
            '"Randomized Controlled Trial"[Publication Type]',
            '"Clinical Trial"[Publication Type]', 
            '"Controlled Clinical Trial"[Publication Type]',
            'randomized[Title/Abstract]',
            'controlled trial[Title/Abstract]',
            'intervention study[Title/Abstract]',
            'clinical trial[Title/Abstract]',
            'RCT[Title/Abstract]'
        ]
    
    def build_study_type_filter(self) -> str:
        """Build study type filter for intervention studies."""
        study_terms = self.get_study_type_terms()
        return '(' + ' OR '.join(study_terms) + ')'


# Global instance
search_terms = InterventionSearchTerms()