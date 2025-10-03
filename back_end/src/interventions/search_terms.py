"""
Search terms library for different intervention categories.
Used for building PubMed queries and identifying relevant papers.
"""

from typing import Dict, List, Set
from back_end.src.interventions.taxonomy import InterventionType


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

        # TEST terms
        terms[InterventionType.TEST] = {
            'primary': [
                'test[Title/Abstract]',
                'testing[Title/Abstract]',
                'diagnostic[Title/Abstract]',
                'diagnosis[Title/Abstract]',
                'screening[Title/Abstract]',
                '"Diagnostic Tests, Routine"[MeSH Terms]',
                '"Laboratory Techniques and Procedures"[MeSH Terms]',
                '"Mass Screening"[MeSH Terms]'
            ],
            'specific': [
                'blood test[Title/Abstract]',
                'breath test[Title/Abstract]',
                'stool test[Title/Abstract]',
                'urine test[Title/Abstract]',
                'genetic test[Title/Abstract]',
                'biomarker[Title/Abstract]',
                'laboratory test[Title/Abstract]',
                'lab test[Title/Abstract]',
                'serology[Title/Abstract]',
                'biochemistry[Title/Abstract]',
                'hematology[Title/Abstract]',
                'endoscopy[Title/Abstract]',
                'colonoscopy[Title/Abstract]',
                'biopsy[Title/Abstract]',
                'imaging[Title/Abstract]',
                'MRI[Title/Abstract]',
                'CT scan[Title/Abstract]',
                'ultrasound[Title/Abstract]',
                'X-ray[Title/Abstract]',
                'functional test[Title/Abstract]',
                'stress test[Title/Abstract]',
                'allergy test[Title/Abstract]',
                'microbiome analysis[Title/Abstract]',
                'metabolomics[Title/Abstract]',
                'proteomics[Title/Abstract]',
                'genomics[Title/Abstract]'
            ]
        }

        # SURGERY terms
        terms[InterventionType.SURGERY] = {
            'primary': [
                'surgery[Title/Abstract]',
                'surgical procedure[Title/Abstract]',
                'operation[Title/Abstract]',
                'surgical intervention[Title/Abstract]',
                '"Surgical Procedures, Operative"[MeSH Terms]',
                '"Surgery"[MeSH Terms]'
            ],
            'specific': [
                'laparoscopic surgery[Title/Abstract]',
                'endoscopic surgery[Title/Abstract]',
                'minimally invasive surgery[Title/Abstract]',
                'robotic surgery[Title/Abstract]',
                'open surgery[Title/Abstract]',
                'bariatric surgery[Title/Abstract]',
                'cardiac surgery[Title/Abstract]',
                'neurosurgery[Title/Abstract]',
                'orthopedic surgery[Title/Abstract]',
                'transplant surgery[Title/Abstract]',
                'reconstructive surgery[Title/Abstract]',
                'surgical resection[Title/Abstract]',
                'arthroscopic surgery[Title/Abstract]',
                'coronary bypass[Title/Abstract]',
                'cholecystectomy[Title/Abstract]',
                'appendectomy[Title/Abstract]',
                'mastectomy[Title/Abstract]',
                'hysterectomy[Title/Abstract]'
            ]
        }

        # DEVICE terms
        terms[InterventionType.DEVICE] = {
            'primary': [
                'medical device[Title/Abstract]',
                'implant[Title/Abstract]',
                'wearable device[Title/Abstract]',
                '"Equipment and Supplies"[MeSH Terms]',
                '"Prostheses and Implants"[MeSH Terms]',
                'monitoring device[Title/Abstract]'
            ],
            'specific': [
                'pacemaker[Title/Abstract]',
                'insulin pump[Title/Abstract]',
                'continuous glucose monitor[Title/Abstract]',
                'CGM[Title/Abstract]',
                'CPAP[Title/Abstract]',
                'hearing aid[Title/Abstract]',
                'cochlear implant[Title/Abstract]',
                'defibrillator[Title/Abstract]',
                'ICD[Title/Abstract]',
                'cardiac monitor[Title/Abstract]',
                'wearable sensor[Title/Abstract]',
                'infusion pump[Title/Abstract]',
                'ventilator[Title/Abstract]',
                'nebulizer[Title/Abstract]',
                'orthotic device[Title/Abstract]',
                'prosthetic[Title/Abstract]',
                'stent[Title/Abstract]',
                'catheter[Title/Abstract]',
                'intraocular lens[Title/Abstract]',
                'neurostimulator[Title/Abstract]'
            ]
        }

        # PROCEDURE terms
        terms[InterventionType.PROCEDURE] = {
            'primary': [
                'medical procedure[Title/Abstract]',
                'therapeutic procedure[Title/Abstract]',
                '"Therapeutic Procedures"[MeSH Terms]',
                'intervention[Title/Abstract]'
            ],
            'specific': [
                'endoscopy[Title/Abstract]',
                'colonoscopy[Title/Abstract]',
                'dialysis[Title/Abstract]',
                'hemodialysis[Title/Abstract]',
                'peritoneal dialysis[Title/Abstract]',
                'blood transfusion[Title/Abstract]',
                'radiation therapy[Title/Abstract]',
                'radiotherapy[Title/Abstract]',
                'chemotherapy[Title/Abstract]',
                'plasmapheresis[Title/Abstract]',
                'phototherapy[Title/Abstract]',
                'cryotherapy[Title/Abstract]',
                'electroconvulsive therapy[Title/Abstract]',
                'ECT[Title/Abstract]',
                'transcranial magnetic stimulation[Title/Abstract]',
                'TMS[Title/Abstract]',
                'oxygen therapy[Title/Abstract]',
                'hyperbaric oxygen[Title/Abstract]',
                'intravenous therapy[Title/Abstract]',
                'IV therapy[Title/Abstract]',
                'catheterization[Title/Abstract]',
                'angioplasty[Title/Abstract]'
            ]
        }

        # BIOLOGICS terms
        terms[InterventionType.BIOLOGICS] = {
            'primary': [
                'biologic[Title/Abstract]',
                'biological therapy[Title/Abstract]',
                'monoclonal antibody[Title/Abstract]',
                '"Biological Products"[MeSH Terms]',
                '"Antibodies, Monoclonal"[MeSH Terms]',
                'immunotherapy[Title/Abstract]'
            ],
            'specific': [
                'vaccine[Title/Abstract]',
                'vaccination[Title/Abstract]',
                'antibody therapy[Title/Abstract]',
                'mAb[Title/Abstract]',
                'TNF inhibitor[Title/Abstract]',
                'anti-TNF[Title/Abstract]',
                'checkpoint inhibitor[Title/Abstract]',
                'PD-1 inhibitor[Title/Abstract]',
                'PD-L1 inhibitor[Title/Abstract]',
                'immune checkpoint[Title/Abstract]',
                'cytokine therapy[Title/Abstract]',
                'interferon[Title/Abstract]',
                'interleukin[Title/Abstract]',
                'growth factor[Title/Abstract]',
                'erythropoietin[Title/Abstract]',
                'insulin therapy[Title/Abstract]',
                'biologic DMARD[Title/Abstract]',
                'targeted therapy[Title/Abstract]',
                'immunoglobulin[Title/Abstract]',
                'IVIG[Title/Abstract]'
            ]
        }

        # GENE_THERAPY terms
        terms[InterventionType.GENE_THERAPY] = {
            'primary': [
                'gene therapy[Title/Abstract]',
                'cellular therapy[Title/Abstract]',
                'gene editing[Title/Abstract]',
                '"Genetic Therapy"[MeSH Terms]',
                '"Gene Editing"[MeSH Terms]',
                'cell therapy[Title/Abstract]'
            ],
            'specific': [
                'CRISPR[Title/Abstract]',
                'CAR-T[Title/Abstract]',
                'CAR T cell[Title/Abstract]',
                'chimeric antigen receptor[Title/Abstract]',
                'stem cell therapy[Title/Abstract]',
                'stem cell transplant[Title/Abstract]',
                'bone marrow transplant[Title/Abstract]',
                'hematopoietic stem cell[Title/Abstract]',
                'HSCT[Title/Abstract]',
                'gene transfer[Title/Abstract]',
                'viral vector[Title/Abstract]',
                'adenoviral vector[Title/Abstract]',
                'lentiviral vector[Title/Abstract]',
                'AAV vector[Title/Abstract]',
                'genome editing[Title/Abstract]',
                'genetic modification[Title/Abstract]',
                'RNA interference[Title/Abstract]',
                'RNAi[Title/Abstract]',
                'antisense therapy[Title/Abstract]',
                'mesenchymal stem cell[Title/Abstract]'
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


# Global instance
search_terms = InterventionSearchTerms()