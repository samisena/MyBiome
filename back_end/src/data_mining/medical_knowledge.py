"""
Centralized medical knowledge repository for data mining modules.
Single source of truth for condition clusters, intervention categories,
synergies, and medical mappings to eliminate redundancy across modules.
"""

from typing import Dict, List, Set, Tuple, Optional


class MedicalKnowledge:
    """Centralized medical knowledge and categorizations."""

    # Condition clusters organized by biological systems/mechanisms
    CONDITION_CLUSTERS = {
        'inflammatory': [
            'arthritis', 'rheumatoid_arthritis', 'ibs', 'crohns', 'inflammatory_bowel_disease',
            'autoimmune_disease', 'psoriasis', 'lupus', 'ulcerative_colitis'
        ],
        'neuropsychological': [
            'depression', 'anxiety', 'ptsd', 'ocd', 'adhd', 'panic_disorder',
            'bipolar', 'schizophrenia', 'social_anxiety', 'generalized_anxiety'
        ],
        'metabolic': [
            'diabetes', 'diabetes_type2', 'obesity', 'metabolic_syndrome', 'pcos',
            'insulin_resistance', 'fatty_liver', 'prediabetes'
        ],
        'cardiovascular': [
            'hypertension', 'heart_disease', 'arrhythmia', 'atherosclerosis',
            'stroke', 'high_cholesterol', 'coronary_artery_disease'
        ],
        'respiratory': [
            'asthma', 'copd', 'lung_fibrosis', 'covid', 'long_covid',
            'bronchitis', 'emphysema', 'pulmonary_hypertension'
        ],
        'neurological': [
            'alzheimers', 'parkinsons', 'multiple_sclerosis', 'epilepsy',
            'migraine', 'chronic_fatigue', 'neuropathy', 'dementia'
        ],
        'gastrointestinal': [
            'ibs', 'crohns', 'ulcerative_colitis', 'gerd', 'leaky_gut',
            'sibo', 'gastritis', 'celiac_disease', 'diverticulitis'
        ],
        'pain_related': [
            'chronic_pain', 'fibromyalgia', 'arthritis', 'migraine',
            'neuropathy', 'back_pain', 'neck_pain', 'joint_pain'
        ],
        'developmental': [
            'autism', 'adhd', 'autism_sensory', 'learning_disabilities',
            'dyslexia', 'developmental_delays'
        ],
        'sleep_related': [
            'insomnia', 'sleep_apnea', 'narcolepsy', 'restless_legs',
            'circadian_rhythm_disorder', 'hypersomnia'
        ],
        'hormonal': [
            'thyroid_disorders', 'pcos', 'menopause', 'low_testosterone',
            'adrenal_fatigue', 'hypothyroidism', 'hyperthyroidism'
        ],
        'skin_conditions': [
            'eczema', 'psoriasis', 'acne', 'rosacea', 'dermatitis',
            'hives', 'vitiligo'
        ]
    }

    # Medical specialty groupings (alternative organization)
    MEDICAL_SPECIALTIES = {
        'mental_health': CONDITION_CLUSTERS['neuropsychological'],
        'autoimmune': CONDITION_CLUSTERS['inflammatory'][:6],  # First 6 are primarily autoimmune
        'digestive': CONDITION_CLUSTERS['gastrointestinal'],
        'cardiac': CONDITION_CLUSTERS['cardiovascular'],
        'pulmonary': CONDITION_CLUSTERS['respiratory'],
        'endocrine': CONDITION_CLUSTERS['metabolic'] + CONDITION_CLUSTERS['hormonal']
    }

    # Known synergistic intervention combinations
    KNOWN_SYNERGIES = {
        ('probiotics', 'prebiotics'): {
            'mechanism': 'Prebiotics feed probiotics',
            'recommendation': 'Take together - prebiotics feed the probiotics',
            'pathways': ['microbiome_enhancement', 'gut_barrier_function'],
            'synergy_factor': 1.5
        },
        ('vitamin_d', 'magnesium'): {
            'mechanism': 'Magnesium activates vitamin D',
            'recommendation': 'Magnesium helps vitamin D absorption and activation',
            'pathways': ['bone_metabolism', 'immune_function'],
            'synergy_factor': 1.4
        },
        ('cbt', 'exercise'): {
            'mechanism': 'Exercise boosts mood; CBT maintains gains',
            'recommendation': 'Exercise provides biological boost; CBT maintains psychological gains',
            'pathways': ['neuroplasticity', 'stress_response'],
            'synergy_factor': 1.6
        },
        ('omega_3', 'vitamin_d'): {
            'mechanism': 'Both support anti-inflammatory pathways',
            'recommendation': 'Synergistic anti-inflammatory effects',
            'pathways': ['inflammation_reduction', 'immune_modulation'],
            'synergy_factor': 1.3
        },
        ('meditation', 'exercise'): {
            'mechanism': 'Meditation enhances exercise benefits via stress reduction',
            'recommendation': 'Meditation amplifies exercise-induced neuroplasticity',
            'pathways': ['stress_response', 'neuroplasticity', 'autonomic_balance'],
            'synergy_factor': 1.4
        },
        ('caffeine', 'l_theanine'): {
            'mechanism': 'L-theanine smooths caffeine stimulation',
            'recommendation': 'L-theanine provides focus without jitters',
            'pathways': ['neurotransmitter_balance', 'attention_enhancement'],
            'synergy_factor': 1.3
        },
        ('curcumin', 'black_pepper'): {
            'mechanism': 'Piperine increases curcumin absorption',
            'recommendation': 'Black pepper increases curcumin bioavailability 20x',
            'pathways': ['bioavailability_enhancement', 'anti_inflammatory'],
            'synergy_factor': 2.0
        },
        ('zinc', 'vitamin_c'): {
            'mechanism': 'Synergistic immune support',
            'recommendation': 'Enhanced immune function when combined',
            'pathways': ['immune_enhancement', 'antioxidant_activity'],
            'synergy_factor': 1.4
        },
        ('vitamin_d', 'vitamin_k2'): {
            'mechanism': 'K2 directs calcium to bones, not arteries',
            'recommendation': 'K2 prevents arterial calcification from vitamin D',
            'pathways': ['bone_metabolism', 'cardiovascular_protection'],
            'synergy_factor': 1.5
        },
        ('iron', 'vitamin_c'): {
            'mechanism': 'Vitamin C enhances iron absorption',
            'recommendation': 'Take iron with vitamin C source for better absorption',
            'pathways': ['mineral_absorption', 'hematological_function'],
            'synergy_factor': 1.6
        }
    }

    # Mechanism-intervention mappings
    MECHANISM_INTERVENTIONS = {
        'anti_inflammatory': [
            'omega_3', 'turmeric', 'curcumin', 'cold_exposure',
            'anti_inflammatory_diet', 'probiotics', 'green_tea',
            'resveratrol', 'quercetin', 'ginger'
        ],
        'neuroprotective': [
            'meditation', 'exercise', 'omega_3', 'lion_mane_mushroom',
            'nootropics', 'cold_exposure', 'intermittent_fasting',
            'blueberries', 'green_tea', 'dark_chocolate'
        ],
        'metabolic_enhancing': [
            'exercise', 'intermittent_fasting', 'cold_exposure',
            'metformin', 'berberine', 'chromium', 'cinnamon',
            'apple_cider_vinegar', 'green_tea'
        ],
        'microbiome_support': [
            'probiotics', 'prebiotics', 'fermented_foods',
            'fiber', 'resistant_starch', 'polyphenols',
            'avoid_antibiotics', 'diverse_diet'
        ],
        'stress_reducing': [
            'meditation', 'yoga', 'breathing_exercises',
            'adaptogenic_herbs', 'magnesium', 'l_theanine',
            'ashwagandha', 'rhodiola', 'massage'
        ],
        'immune_modulating': [
            'vitamin_d', 'vitamin_c', 'zinc', 'probiotics',
            'elderberry', 'echinacea', 'garlic', 'mushrooms',
            'astragalus', 'selenium'
        ],
        'hormonal_balancing': [
            'exercise', 'stress_management', 'adequate_sleep',
            'vitamin_d', 'omega_3', 'magnesium', 'zinc',
            'ashwagandha', 'maca', 'dim'
        ],
        'neurotransmitter_support': [
            'exercise', 'omega_3', 'probiotics', 'tryptophan',
            'tyrosine', 'b_vitamins', 'magnesium', 'sam_e',
            '5_htp', 'gaba'
        ],
        'mitochondrial_support': [
            'coq10', 'pqq', 'nad_precursors', 'alpha_lipoic_acid',
            'acetyl_l_carnitine', 'resveratrol', 'exercise',
            'cold_exposure', 'red_light_therapy'
        ],
        'detoxification': [
            'cruciferous_vegetables', 'milk_thistle', 'nac',
            'glutathione', 'chlorella', 'cilantro', 'sweating',
            'hydration', 'fiber'
        ]
    }

    # Intervention categories and classifications
    INTERVENTION_CATEGORIES = {
        'dietary': [
            'mediterranean_diet', 'ketogenic_diet', 'anti_inflammatory_diet',
            'low_fodmap', 'gluten_free', 'dairy_free', 'paleo_diet',
            'vegan_diet', 'intermittent_fasting', 'caloric_restriction'
        ],
        'supplement': [
            'omega_3', 'vitamin_d', 'magnesium', 'probiotics', 'vitamin_c',
            'zinc', 'b_complex', 'iron', 'calcium', 'multivitamin',
            'coq10', 'turmeric', 'ashwagandha'
        ],
        'lifestyle': [
            'exercise', 'meditation', 'yoga', 'sleep_hygiene',
            'stress_management', 'breathing_exercises', 'cold_exposure',
            'sauna', 'massage', 'acupuncture'
        ],
        'pharmaceutical': [
            'ssri', 'metformin', 'statin', 'ace_inhibitor', 'beta_blocker',
            'ppi', 'nsaid', 'antibiotic', 'antiviral', 'corticosteroid'
        ],
        'behavioral': [
            'cbt', 'mindfulness', 'biofeedback', 'exposure_therapy',
            'dialectical_behavior_therapy', 'acceptance_commitment_therapy',
            'emdr', 'group_therapy'
        ],
        'botanical': [
            'turmeric', 'ginger', 'ashwagandha', 'rhodiola', 'ginseng',
            'valerian', 'passionflower', 'st_johns_wort', 'echinacea',
            'milk_thistle'
        ],
        'physical': [
            'physical_therapy', 'chiropractic', 'massage', 'acupuncture',
            'dry_needling', 'cupping', 'stretching', 'foam_rolling'
        ],
        'environmental': [
            'light_therapy', 'air_purification', 'emf_reduction',
            'noise_reduction', 'nature_exposure', 'grounding',
            'temperature_regulation'
        ]
    }

    # Classification thresholds (unified across modules)
    CLASSIFICATION_THRESHOLDS = {
        'established': {
            'min_evidence': 10,
            'min_confidence': 0.75,
            'min_studies': 5
        },
        'promising': {
            'min_evidence': 5,
            'min_confidence': 0.60,
            'min_studies': 3
        },
        'emerging': {
            'min_evidence': 2,
            'min_confidence': 0.40,
            'min_studies': 1
        },
        'experimental': {
            'min_evidence': 1,
            'min_confidence': 0.20,
            'min_studies': 1
        },
        'insufficient': {
            'min_evidence': 0,
            'min_confidence': 0,
            'min_studies': 0
        }
    }

    # Mechanism interpretation descriptions
    MECHANISM_INTERPRETATIONS = {
        'neuropsych': 'Mental health and neuropsychological interventions',
        'gut_microbiome': 'Gut microbiome and digestive health',
        'inflammation': 'Inflammatory processes and immune response',
        'metabolic': 'Metabolic and endocrine regulation',
        'stress_response': 'Stress response and HPA axis',
        'neurotransmitter': 'Neurotransmitter and brain chemistry',
        'hormonal': 'Hormonal balance and endocrine function',
        'mitochondrial': 'Cellular energy and mitochondrial function',
        'oxidative_stress': 'Oxidative stress and antioxidant systems',
        'epigenetic': 'Gene expression and epigenetic regulation',
        'vascular': 'Blood flow and vascular health',
        'structural': 'Physical structure and biomechanics'
    }

    @classmethod
    def get_condition_cluster(cls, condition: str) -> Optional[str]:
        """
        Find which cluster a condition belongs to.

        Args:
            condition: Medical condition name

        Returns:
            Cluster name or None if not found
        """
        condition_lower = condition.lower().replace(' ', '_')
        for cluster_name, conditions in cls.CONDITION_CLUSTERS.items():
            if condition_lower in [c.lower() for c in conditions]:
                return cluster_name
        return None

    @classmethod
    def get_related_conditions(cls, condition: str, max_conditions: int = 10) -> List[str]:
        """
        Get conditions related to the given condition.

        Args:
            condition: Target condition
            max_conditions: Maximum number of related conditions

        Returns:
            List of related conditions
        """
        cluster = cls.get_condition_cluster(condition)
        if not cluster:
            return []

        conditions = cls.CONDITION_CLUSTERS[cluster]
        # Remove the target condition and return others
        related = [c for c in conditions if c.lower() != condition.lower()]
        return related[:max_conditions]

    @classmethod
    def get_intervention_category(cls, intervention: str) -> Optional[str]:
        """
        Find which category an intervention belongs to.

        Args:
            intervention: Intervention name

        Returns:
            Category name or None if not found
        """
        intervention_lower = intervention.lower().replace(' ', '_')
        for category, interventions in cls.INTERVENTION_CATEGORIES.items():
            if intervention_lower in [i.lower() for i in interventions]:
                return category
        return None

    @classmethod
    def get_synergy_info(cls, intervention1: str, intervention2: str) -> Optional[Dict]:
        """
        Get synergy information for an intervention pair.

        Args:
            intervention1: First intervention
            intervention2: Second intervention

        Returns:
            Synergy information or None if no known synergy
        """
        key1 = (intervention1.lower(), intervention2.lower())
        key2 = (intervention2.lower(), intervention1.lower())

        for key in [key1, key2]:
            if key in cls.KNOWN_SYNERGIES:
                return cls.KNOWN_SYNERGIES[key]
        return None

    @classmethod
    def is_condition_in_cluster(cls, condition: str, cluster: str) -> bool:
        """
        Check if a condition belongs to a specific cluster.

        Args:
            condition: Medical condition
            cluster: Cluster name

        Returns:
            True if condition is in cluster
        """
        if cluster not in cls.CONDITION_CLUSTERS:
            return False
        condition_lower = condition.lower().replace(' ', '_')
        return condition_lower in [c.lower() for c in cls.CONDITION_CLUSTERS[cluster]]

    @classmethod
    def get_mechanisms_for_intervention(cls, intervention: str) -> List[str]:
        """
        Get mechanisms associated with an intervention.

        Args:
            intervention: Intervention name

        Returns:
            List of mechanism names
        """
        intervention_lower = intervention.lower().replace(' ', '_')
        mechanisms = []
        for mechanism, interventions in cls.MECHANISM_INTERVENTIONS.items():
            if intervention_lower in [i.lower() for i in interventions]:
                mechanisms.append(mechanism)
        return mechanisms