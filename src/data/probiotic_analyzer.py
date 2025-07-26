import sqlite3
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from tqdm import tqdm
import faiss
from transformers import pipeline, AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from src.data.models import Author, Paper
from src.data.database_manager import DatabaseManager
from src.data.paper_parser import PubmedParser

project_root = Path(__file__).parent.parent.parent

import re
from pathlib import Path

from transformers import pipeline, AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class ProbioticAnalyzer:
    """Analyzes probiotic research papers to extract strain-condition relationships"""
    """Analyzes probiotic research papers to extract strain-condition relationships"""
    
    def __init__(self, project_root):
        self.project_root = project_root
        self.db_path = project_root / "data" / "processed" / 'pubmed_research.db'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        #* Initialize models
        self._initialize_models()
        
        #* Pre-compute reference embeddings for outcomes
        self._initialize_reference_embeddings()
    
    
    def _initialize_models(self):
        """Initializes LLM models"""
        print("Loading models...")
        
        #* 1. Named Entity Recognition using BioBERT
        self.ner_pipeline = pipeline(
            "ner",
            model="dmis-lab/biobert-base-cased-v1.2",
            aggregation_strategy="simple",
            device=0 if torch.cuda.is_available() else -1
        )
        
        #* 2. Semantic embeddings using PubMedBERT
        self.sentence_model = SentenceTransformer('pritamdeka/S-PubMedBert-MS-MARCO')
        
        #* 3. Relation extraction using BioBERT
        self.relation_tokenizer = AutoTokenizer.from_pretrained(
            "dmis-lab/biobert-base-cased-v1.2"
        )
        self.relation_model = AutoModel.from_pretrained(
            "dmis-lab/biobert-base-cased-v1.2"
        ).to(self.device)
        
        print("All models loaded successfully!")
    
    def _initialize_reference_embeddings(self):
        """Pre-compute reference embeddings for outcome detection"""
        positive_refs = [
            "The treatment was effective and improved symptoms",
            "Significant improvement was observed",
            "The intervention successfully reduced disease markers"
        ]
        
        negative_refs = [
            "The treatment was ineffective",
            "No improvement was observed",
            "Adverse effects were reported"
        ]
        
        neutral_refs = [
            "No significant difference was observed between groups",
            "The treatment had no effect on the condition",
            "Results showed no change in symptoms",
            "There was no statistical difference between treatment and control",
            "The intervention did not affect the outcome",
            "No clinically meaningful changes were detected"
        ]
        
        self.pos_embeddings = self.sentence_model.encode(positive_refs)
        self.neg_embeddings = self.sentence_model.encode(negative_refs)
        self.neutral_embeddings = self.sentence_model.encode(neutral_refs)
    
    
    def prepare_data_from_db(self) -> pd.DataFrame:
        """Loads data from SQLite database as a pandas DataFrame"""
        with sqlite3.connect(self.db_path) as conn:
            query = """
                SELECT
                    pmid as paper_id,
                    title,
                    abstract
                FROM papers
                WHERE abstract IS NOT NULL AND abstract != ''
            """
            
            df = pd.read_sql_query(query, conn)
            df['full_text'] = df['title'].fillna('') + ' ' + df['abstract'].fillna('')
            print(f"Loaded {len(df)} papers from database")
            return df
    
    
    def extract_entities(self, text: str) -> Dict[str, List[Dict]]:
        """Uses BioBERT NER to extract strains and conditions with improved probiotic detection"""
        
        #* Detects named entities:
        entities = self.ner_pipeline(text[:512])  # BioBERT token limit
        
        categorized_entities = {
            'strains': [],
            'conditions': [],
            'outcomes': []
        }
        
        # Enhanced strain detection with common probiotic genera and species patterns
        strain_keywords = [
            'bacterium', 'bacillus', 'coccus', 'bifidum', 'lactis', 
            'lactobacillus', 'streptococcus', 'saccharomyces', 'bifido',
            'lacto', 'strepto', 'entero', 'clostridium', 'escherichia',
            'propionibacterium', 'akkermansia', 'faecalibacterium',
            'bacteroides', 'prevotella', 'ruminococcus', 'roseburia',
            'probiotic', 'probiotics', 'synbiotic', 'prebiotic'
        ]
        
        # Common probiotic genus patterns
        genus_patterns = [
            'Lactobacillus', 'Bifidobacterium', 'Streptococcus', 'Enterococcus',
            'Saccharomyces', 'Bacillus', 'Lactococcus', 'Pediococcus',
            'Leuconostoc', 'Propionibacterium', 'Clostridium', 'Escherichia',
            'Weissella', 'Oenococcus', 'Lactiplantibacillus', 'Lacticaseibacillus',
            'Limosilactobacillus', 'Ligilactobacillus', 'Lacticaseibacillus'
        ]
        
        # Common species epithets in probiotics
        species_patterns = [
            'acidophilus', 'casei', 'rhamnosus', 'bulgaricus', 'plantarum',
            'brevis', 'fermentum', 'reuteri', 'johnsonii', 'lactis',
            'longum', 'breve', 'infantis', 'bifidum', 'animalis',
            'thermophilus', 'helveticus', 'delbrueckii', 'paracasei',
            'salivarius', 'gasseri', 'crispatus', 'jensenii',
            'boulardii', 'cerevisiae', 'coagulans', 'clausii'
        ]
        
        # Process entities with enhanced strain detection
        for entity in entities:
            entity_text = entity['word']
            entity_lower = entity_text.lower()
            
            # Check for strain indicators
            is_strain = False
            
            # Check if entity contains strain keywords
            if any(keyword in entity_lower for keyword in strain_keywords):
                is_strain = True
            
            # Check if entity matches genus patterns (case-sensitive)
            elif any(genus in entity_text for genus in genus_patterns):
                is_strain = True
            
            # Check if entity contains species epithets
            elif any(species in entity_lower for species in species_patterns):
                is_strain = True
            
            # Check for strain number patterns (e.g., "BB-12", "LGG", "GG")
            elif re.search(r'\b[A-Z]{1,3}[-\s]?\d+\b', entity_text):
                is_strain = True
            
            # Check for full species names (e.g., "L. acidophilus")
            elif re.search(r'\b[A-Z]\.\s*[a-z]+\b', entity_text):
                is_strain = True
            
            if is_strain:
                categorized_entities['strains'].append(entity)
            elif entity['entity_group'] in ['DISEASE', 'PHENOTYPE', 'CHEMICAL']:
                # CHEMICAL added as some conditions involve metabolic markers
                categorized_entities['conditions'].append(entity)
        
        # Also search for strains using regex patterns in the full text
        strain_regex_patterns = [
            r'\b(?:Lactobacillus|Bifidobacterium|Streptococcus|L\.|B\.|S\.)\s+[a-z]+\b',
            r'\b[A-Z][a-z]+\s+[a-z]+\s+(?:subsp\.|ssp\.)\s+[a-z]+\b',
            r'\b(?:strain|ATCC|DSM|NCFM|Bb12|LGG|GG|BB-12)\s*[\d\-]+\b',
            r'\b(?:Lactobacillus|Bifidobacterium)\s+(?:strain\s+)?[A-Z]{1,3}\d+\b',
            r'\b[A-Z][a-z]+\s+[a-z]+\s+[A-Z]{1,3}\d+\b',  # e.g., "Lactobacillus rhamnosus GG"
            r'\bprobiotic[s]?\s+(?:strain[s]?|bacteria|supplement[s]?)\b'
        ]
        
        for pattern in strain_regex_patterns:
            matches = re.finditer(pattern, text[:512], re.IGNORECASE)
            for match in matches:
                strain_text = match.group()
                # Check if this strain is already captured
                if not any(strain_text.lower() in s['word'].lower() for s in categorized_entities['strains']):
                    categorized_entities['strains'].append({
                        'word': strain_text,
                        'start': match.start(),
                        'end': match.end(),
                        'entity_group': 'STRAIN',
                        'score': 0.9  # High confidence for regex matches
                    })
        
        categorized_entities['outcomes'] = self._extract_outcomes(text)
        return categorized_entities
    
    
    def _extract_study_type(self, text: str) -> Dict[str, float]:
        """Extract study type and assign evidence strength score"""
        text_lower = text.lower()
        
        # Patterns for different study types
        rct_patterns = [
            'randomized controlled trial',
            'randomized clinical trial',
            'randomised controlled trial',
            'randomised clinical trial',
            'rct',
            'double-blind',
            'double blind',
            'placebo-controlled',
            'placebo controlled',
            'randomized trial',
            'randomised trial'
        ]
        
        observational_patterns = [
            'observational study',
            'cohort study',
            'case-control',
            'case control',
            'cross-sectional',
            'cross sectional',
            'retrospective',
            'prospective cohort',
            'epidemiological'
        ]
        
        systematic_review_patterns = [
            'systematic review',
            'meta-analysis',
            'meta analysis',
            'pooled analysis'
        ]
        
        case_report_patterns = [
            'case report',
            'case series',
            'case study'
        ]
        
        in_vitro_patterns = [
            'in vitro',
            'in-vitro',
            'cell culture',
            'cell line',
            'cultured cells'
        ]
        
        in_vivo_patterns = [
            'in vivo',
            'in-vivo',
            'animal model',
            'mouse model',
            'rat model',
            'murine'
        ]
        
        # Check for study type and assign evidence strength
        if any(pattern in text_lower for pattern in systematic_review_patterns):
            return {
                'study_type': 'systematic_review',
                'evidence_strength': 1.0  # Highest evidence
            }
        elif any(pattern in text_lower for pattern in rct_patterns):
            return {
                'study_type': 'randomized_controlled_trial',
                'evidence_strength': 0.9
            }
        elif any(pattern in text_lower for pattern in observational_patterns):
            return {
                'study_type': 'observational_study',
                'evidence_strength': 0.6
            }
        elif any(pattern in text_lower for pattern in in_vivo_patterns):
            return {
                'study_type': 'in_vivo_study',
                'evidence_strength': 0.4
            }
        elif any(pattern in text_lower for pattern in case_report_patterns):
            return {
                'study_type': 'case_report',
                'evidence_strength': 0.3
            }
        elif any(pattern in text_lower for pattern in in_vitro_patterns):
            return {
                'study_type': 'in_vitro_study',
                'evidence_strength': 0.2
            }
        else:
            return {
                'study_type': 'unspecified',
                'evidence_strength': 0.5  # Default moderate evidence
            }
    
    
    def _extract_outcomes(self, text: str) -> List[Dict]:
        """Extract outcome sentences from text"""
        outcomes = []
        sentences = text.split('.')[:20]  # First 20 sentences
        
        for sentence in sentences:
            if len(sentence.strip()) < 20:
                continue
            
            embedding = self.sentence_model.encode(sentence)
            
            #* Calculates similarities with reference sentences
            pos_sim = np.mean([cosine_similarity([embedding], [ref_emb])[0][0]
                              for ref_emb in self.pos_embeddings])
            neg_sim = np.mean([cosine_similarity([embedding], [ref_emb])[0][0]
                              for ref_emb in self.neg_embeddings])
            neutral_sim = np.mean([cosine_similarity([embedding], [ref_emb])[0][0]
                                 for ref_emb in self.neutral_embeddings])
            
            max_sim = max(pos_sim, neg_sim, neutral_sim)
            
            if max_sim > 0.7:
                if pos_sim == max_sim:
                    outcome_type = 'positive'
                elif neg_sim == max_sim:
                    outcome_type = 'negative'
                else:
                    outcome_type = 'neutral'
                
                outcomes.append({
                    'text': sentence.strip(),
                    'type': outcome_type,
                    'confidence': max_sim
                })
        
        return outcomes
    
    
    def extract_strain_condition_relationships(self, paper_data: Dict) -> List[Dict]:
        """Extract relationships between strains and conditions in a paper"""
        text = paper_data['full_text']
        entities = self.extract_entities(text)
        relationships = []
        
        # Extract study type and evidence strength
        study_info = self._extract_study_type(text)
        
        # Only process if we have both strains and conditions
        if not entities['strains'] or not entities['conditions']:
            return relationships
        
        for strain in entities['strains']:
            for condition in entities['conditions']:
                # Check proximity first
                if abs(strain['start'] - condition['start']) > 500:
                    continue
                
                context = self._extract_context_window(
                    text, strain['start'], strain['end'],
                    condition['start'], condition['end']
                )
                
                if context:
                    relationship_info = self._analyze_relationship(
                        context, strain['word'], condition['word']
                    )
                    
                    relevant_outcomes = self._find_relevant_outcomes(
                        strain, condition, entities['outcomes'], text
                    )
                    
                    relationships.append({
                        'strain': strain['word'],
                        'condition': condition['word'],
                        'confidence': relationship_info['confidence'],
                        'relationship_type': relationship_info['type'],
                        'context': context,
                        'outcomes': relevant_outcomes,
                        'paper_id': paper_data.get('paper_id', 'unknown'),
                        'study_type': study_info['study_type'],
                        'evidence_strength': study_info['evidence_strength']
                    })
        
        return relationships
    
    
    def _extract_context_window(self, text: str, start1: int, end1: int,
                               start2: int, end2: int, window: int = 200) -> Optional[str]:
        """Extract context window around two entities"""
        context_start = max(0, min(start1, start2) - window)
        context_end = min(len(text), max(end1, end2) + window)
        return text[context_start:context_end]
    
    
    def _analyze_relationship(self, context: str, strain: str, condition: str) -> Dict:
        """Analyze the relationship between strain and condition"""
        marked_context = context.replace(strain, f"[STRAIN]{strain}[/STRAIN]")
        marked_context = marked_context.replace(condition, f"[CONDITION]{condition}[/CONDITION]")
        
        inputs = self.relation_tokenizer(
            marked_context,
            return_tensors="pt",
            max_length=512,
            padding=True,
            truncation=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.relation_model(**inputs)
            relationship_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]
        
        relationship_types = {
            'therapeutic': "The probiotic strain treats and improves the condition",
            'preventive': "The probiotic strain prevents the condition",
            'neutral': "The probiotic strain has no significant effect on the condition",
            'associated': "The probiotic strain is associated with the condition"
        }
        
        similarities = {}
        for rel_type, description in relationship_types.items():
            inputs = self.relation_tokenizer(
                description,
                return_tensors='pt',
                max_length=512,
                truncation=True
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.relation_model(**inputs)
                type_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]
            
            similarities[rel_type] = cosine_similarity([relationship_embedding], [type_embedding])[0][0]
        
        best_type = max(similarities.items(), key=lambda x: x[1])
        
        return {
            'type': best_type[0],
            'confidence': best_type[1],
            'all_scores': similarities
        }
    
    
    def _find_relevant_outcomes(self, strain: Dict, condition: Dict,
                               outcomes: List[Dict], full_text: str) -> List[Dict]:
        """Find outcomes relevant to strain-condition pair"""
        relevant_outcomes = []
        
        for outcome in outcomes:
            outcome_start = full_text.find(outcome['text'])
            if outcome_start == -1:
                continue
            
            strain_distance = abs(outcome_start - strain['start'])
            condition_distance = abs(outcome_start - condition['start'])
            
            if strain_distance < 300 or condition_distance < 300:
                relevant_outcomes.append({
                    'text': outcome['text'],
                    'type': outcome['type'],  # Now includes 'positive', 'negative', or 'neutral'
                    'confidence': outcome['confidence']
                })
        
        return relevant_outcomes
    
    
    def build_knowledge_graph(self, relationships: List[Dict]) -> Dict:
        """Build knowledge graph from relationships"""
        knowledge_graph = {
            'nodes': {
                'strains': {},
                'conditions': {}
            },
            'edges': []
        }
        
        for rel in relationships:
            strain = rel['strain']
            condition = rel['condition']
            
            if strain not in knowledge_graph['nodes']['strains']:
                knowledge_graph['nodes']['strains'][strain] = {
                    'name': strain,
                    'paper_count': 0
                }
            
            if condition not in knowledge_graph['nodes']['conditions']:
                knowledge_graph['nodes']['conditions'][condition] = {
                    'name': condition,
                    'paper_count': 0
                }
            
            knowledge_graph['nodes']['strains'][strain]['paper_count'] += 1
            knowledge_graph['nodes']['conditions'][condition]['paper_count'] += 1
            
            edge = {
                'source': strain,
                'target': condition,
                'type': rel['relationship_type'],
                'confidence': rel['confidence'],
                'outcomes': rel.get('outcomes', []),
                'paper_id': rel.get('paper_id'),
                'study_type': rel.get('study_type', 'unspecified'),
                'evidence_strength': rel.get('evidence_strength', 0.5)
            }
            knowledge_graph['edges'].append(edge)
        
        return knowledge_graph
    
    
    def aggregate_relationships(self, all_relationships: List[Dict]) -> pd.DataFrame:
        """Aggregate relationships by strain-condition pair"""
        pair_data = defaultdict(lambda: {
            'papers': [],
            'confidence_scores': [],
            'relationship_types': [],
            'positive_outcomes': 0,
            'negative_outcomes': 0,
            'neutral_outcomes': 0,
            'total_outcomes': 0,
            'study_types': [],
            'evidence_strengths': []
        })
        
        for rel in all_relationships:
            key = (rel['strain'], rel['condition'])
            
            pair_data[key]['papers'].append(rel.get('paper_id', 'unknown'))
            pair_data[key]['confidence_scores'].append(rel['confidence'])
            pair_data[key]['relationship_types'].append(rel['relationship_type'])
            pair_data[key]['study_types'].append(rel.get('study_type', 'unspecified'))
            pair_data[key]['evidence_strengths'].append(rel.get('evidence_strength', 0.5))
            
            for outcome in rel.get('outcomes', []):
                if outcome['type'] == 'positive':
                    pair_data[key]['positive_outcomes'] += 1
                elif outcome['type'] == 'negative':
                    pair_data[key]['negative_outcomes'] += 1
                elif outcome['type'] == 'neutral':
                    pair_data[key]['neutral_outcomes'] += 1
                pair_data[key]['total_outcomes'] += 1
        
        aggregated_data = []
        for (strain, condition), data in pair_data.items():
            avg_confidence = np.mean(data['confidence_scores'])
            avg_evidence_strength = np.mean(data['evidence_strengths'])
            paper_count = len(set(data['papers']))
            
            # Count study types
            study_type_counts = defaultdict(int)
            for study_type in data['study_types']:
                study_type_counts[study_type] += 1
            
            # Count relationship types
            type_counts = defaultdict(int)
            for rel_type in data['relationship_types']:
                type_counts[rel_type] += 1
            
            primary_type = max(type_counts.items(), key=lambda x: x[1])[0]
            
            # Calculate outcome score
            if data['total_outcomes'] > 0:
                # Positive outcomes get +1, negative get -1, neutral get 0
                outcome_score = (data['positive_outcomes'] - data['negative_outcomes']) / data['total_outcomes']
            else:
                outcome_score = 0
            
            # Calculate weighted evidence strength
            # Combines average confidence, evidence strength from study types, and paper count
            weighted_evidence = avg_confidence * avg_evidence_strength * np.log1p(paper_count)
            
            aggregated_data.append({
                'strain': strain,
                'condition': condition,
                'paper_count': paper_count,
                'avg_confidence': avg_confidence,
                'avg_evidence_strength': avg_evidence_strength,
                'primary_relationship': primary_type,
                'positive_outcomes': data['positive_outcomes'],
                'negative_outcomes': data['negative_outcomes'],
                'neutral_outcomes': data['neutral_outcomes'],
                'outcome_score': outcome_score,
                'evidence_strength': paper_count * avg_confidence,  # Keep original metric
                'weighted_evidence_strength': weighted_evidence,  # New weighted metric
                'rct_count': study_type_counts.get('randomized_controlled_trial', 0),
                'systematic_review_count': study_type_counts.get('systematic_review', 0),
                'observational_count': study_type_counts.get('observational_study', 0),
                'in_vivo_count': study_type_counts.get('in_vivo_study', 0),
                'in_vitro_count': study_type_counts.get('in_vitro_study', 0),
                'case_report_count': study_type_counts.get('case_report', 0),
                'predominant_study_type': max(study_type_counts.items(), key=lambda x: x[1])[0] if study_type_counts else 'unspecified'
            })
        
        return pd.DataFrame(aggregated_data).sort_values('weighted_evidence_strength', ascending=False)
    
    
    def generate_insights_report(self, aggregated_df: pd.DataFrame, 
                                knowledge_graph: Dict) -> Dict:
        """Generate insights from analysis"""
        insights = {
            'summary_statistics': {
                'total_relationships': len(aggregated_df),
                'unique_strains': aggregated_df['strain'].nunique(),
                'unique_conditions': aggregated_df['condition'].nunique(),
                'total_rcts': aggregated_df['rct_count'].sum(),
                'total_systematic_reviews': aggregated_df['systematic_review_count'].sum(),
                'total_observational_studies': aggregated_df['observational_count'].sum()
            },
            'top_therapeutic_relationships': [],
            'top_neutral_relationships': [],
            'most_studied_pairs': [],
            'highest_quality_evidence': []
        }
        
        # Top therapeutic relationships
        therapeutic_df = aggregated_df[
            aggregated_df['primary_relationship'] == 'therapeutic'
        ].sort_values('weighted_evidence_strength', ascending=False)
        
        for _, row in therapeutic_df.head(10).iterrows():
            insights['top_therapeutic_relationships'].append({
                'strain': row['strain'],
                'condition': row['condition'],
                'outcome_score': row['outcome_score'],
                'evidence': row['paper_count'],
                'weighted_evidence': row['weighted_evidence_strength'],
                'predominant_study_type': row['predominant_study_type']
            })
        
        # Top neutral relationships (no effect)
        neutral_df = aggregated_df[
            aggregated_df['primary_relationship'] == 'neutral'
        ].sort_values('weighted_evidence_strength', ascending=False)
        
        for _, row in neutral_df.head(10).iterrows():
            insights['top_neutral_relationships'].append({
                'strain': row['strain'],
                'condition': row['condition'],
                'neutral_outcomes': row['neutral_outcomes'],
                'evidence': row['paper_count'],
                'weighted_evidence': row['weighted_evidence_strength'],
                'predominant_study_type': row['predominant_study_type']
            })
        
        # Most studied pairs
        most_studied = aggregated_df.sort_values('paper_count', ascending=False)
        for _, row in most_studied.head(10).iterrows():
            insights['most_studied_pairs'].append({
                'strain': row['strain'],
                'condition': row['condition'],
                'papers': row['paper_count'],
                'rct_count': row['rct_count'],
                'systematic_review_count': row['systematic_review_count']
            })
        
        # Highest quality evidence (based on RCTs and systematic reviews)
        high_quality_df = aggregated_df[
            (aggregated_df['rct_count'] > 0) | (aggregated_df['systematic_review_count'] > 0)
        ].sort_values('weighted_evidence_strength', ascending=False)
        
        for _, row in high_quality_df.head(10).iterrows():
            insights['highest_quality_evidence'].append({
                'strain': row['strain'],
                'condition': row['condition'],
                'relationship': row['primary_relationship'],
                'rct_count': row['rct_count'],
                'systematic_review_count': row['systematic_review_count'],
                'weighted_evidence': row['weighted_evidence_strength']
            })
        
        return insights
    
    def save_results(self, aggregated_df: pd.DataFrame):
        """Save analysis results to database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS strain_condition_analysis (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    strain TEXT,
                    condition TEXT,
                    paper_count INTEGER,
                    avg_confidence REAL,
                    avg_evidence_strength REAL,
                    primary_relationship TEXT,
                    positive_outcomes INTEGER,
                    negative_outcomes INTEGER,
                    neutral_outcomes INTEGER,
                    outcome_score REAL,
                    evidence_strength REAL,
                    weighted_evidence_strength REAL,
                    rct_count INTEGER,
                    systematic_review_count INTEGER,
                    observational_count INTEGER,
                    in_vivo_count INTEGER,
                    in_vitro_count INTEGER,
                    case_report_count INTEGER,
                    predominant_study_type TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            aggregated_df.to_sql('strain_condition_analysis', conn, 
                               if_exists='append', index=False)
            conn.commit()
        print("Results saved to database")
    
    
    def run_complete_analysis(self, sample_size: Optional[int] = None):
        """Run the complete analysis pipeline"""
        print("=== Starting Unified Probiotic Analysis ===\n")
        
        # Load data
        df = self.prepare_data_from_db()
        if sample_size:
            df = df.sample(min(sample_size, len(df)), random_state=42)
        print(f"Using sample of {len(df)} papers")
        
        # Extract relationships from each paper
        all_relationships = []
        print("\nExtracting strain-condition relationships...")
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing papers"):
            paper_relationships = self.extract_strain_condition_relationships(row)
            all_relationships.extend(paper_relationships)
        
        print(f"\nFound {len(all_relationships)} total relationships")
        
        # Build knowledge graph
        print("\nBuilding knowledge graph...")
        knowledge_graph = self.build_knowledge_graph(all_relationships)
        
        # Aggregate relationships
        print("\nAggregating relationships...")
        aggregated_df = self.aggregate_relationships(all_relationships)
        
        # Generate insights
        print("\nGenerating insights report...")
        insights = self.generate_insights_report(aggregated_df, knowledge_graph)
        
        # Save results
        print("\nSaving results...")
        self.save_results(aggregated_df)
        
        # Print summary
        self._print_summary(insights)
        
        return {
            'relationships': all_relationships,
            'aggregated': aggregated_df,
            'knowledge_graph': knowledge_graph,
            'insights': insights
        }
    
    
    def _print_summary(self, insights: Dict):
        """Print analysis summary"""
        print("\n=== Analysis Summary ===")
        print(f"Total relationships found: {insights['summary_statistics']['total_relationships']}")
        print(f"Unique strains: {insights['summary_statistics']['unique_strains']}")
        print(f"Unique conditions: {insights['summary_statistics']['unique_conditions']}")
        print(f"\nStudy Type Distribution:")
        print(f"  - Randomized Controlled Trials: {insights['summary_statistics']['total_rcts']}")
        print(f"  - Systematic Reviews: {insights['summary_statistics']['total_systematic_reviews']}")
        print(f"  - Observational Studies: {insights['summary_statistics']['total_observational_studies']}")
        
        print("\nTop Therapeutic Relationships:")
        for rel in insights['top_therapeutic_relationships'][:5]:
            print(f"  {rel['strain']} -> {rel['condition']}: "
                  f"score={rel['outcome_score']:.2f}, evidence={rel['evidence']}, "
                  f"study_type={rel['predominant_study_type']}")
        
        print("\nTop Neutral Relationships (No Effect):")
        for rel in insights['top_neutral_relationships'][:5]:
            print(f"  {rel['strain']} -> {rel['condition']}: "
                  f"neutral_outcomes={rel['neutral_outcomes']}, evidence={rel['evidence']}, "
                  f"study_type={rel['predominant_study_type']}")
        
        print("\nHighest Quality Evidence (RCTs/Systematic Reviews):")
        for rel in insights['highest_quality_evidence'][:5]:
            print(f"  {rel['strain']} + {rel['condition']}: "
                  f"RCTs={rel['rct_count']}, Reviews={rel['systematic_review_count']}, "
                  f"relationship={rel['relationship']}")
        
        print("\nMost Studied Pairs:")
        for pair in insights['most_studied_pairs'][:5]:
            print(f"  {pair['strain']} + {pair['condition']}: {pair['papers']} papers "
                  f"(RCTs={pair['rct_count']}, Reviews={pair['systematic_review_count']})")
        
         

            

        
            
                    
                
            
        
