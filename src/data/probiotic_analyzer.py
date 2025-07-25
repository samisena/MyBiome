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

class ProbioticAnalyzer:
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
        
        self.pos_embeddings = self.sentence_model.encode(positive_refs)
        self.neg_embeddings = self.sentence_model.encode(negative_refs)
    
    
    def prepare_data_from_db(self) -> pd.DataFrame:
        """Loads data from SQLite database as a pandas DataFrame"""
        with sqlite3.connect(self.db_path) as conn:
            query = """
                SELECT
                    paper_id,
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
        """Uses BioBERT NER to extract strains and conditions"""
        
        #* Detects named entities:
        entities = self.ner_pipeline(text[:512])  # BioBERT token limit
        
        categorized_entities = {
            'strains': [],
            'conditions': [],
            'outcomes': []
        }
        
        strain_keywords = ['bacterium', 'bacillus', 'coccus', 'bifidum', 'lactis', 
                          'lactobacillus', 'streptococcus', 'saccharomyces']
        
        for entity in entities:
            if entity['entity_group'] in ['GENE', 'PROTEIN']:
                if any(term in entity['word'].lower() for term in strain_keywords):
                    categorized_entities['strains'].append(entity)
            elif entity['entity_group'] in ['DISEASE', 'PHENOTYPE']:
                categorized_entities['conditions'].append(entity)
        
        categorized_entities['outcomes'] = self._extract_outcomes(text)
        return categorized_entities
    
    
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
            
            if pos_sim > 0.7 or neg_sim > 0.7:
                outcomes.append({
                    'text': sentence.strip(),
                    'type': 'positive' if pos_sim > neg_sim else 'negative',
                    'confidence': max(pos_sim, neg_sim)
                })
        
        return outcomes
    
    
    def extract_strain_condition_relationships(self, paper_data: Dict) -> List[Dict]:
        """Extract relationships between strains and conditions in a paper"""
        text = paper_data['full_text']
        entities = self.extract_entities(text)
        relationships = []
        
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
                        'paper_id': paper_data.get('paper_id', 'unknown')
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
            'associated': "The probiotic strain is associated with the condition",
            'no_effect': "The probiotic strain has no effect on the condition"
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
                    'type': outcome['type'],
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
                'paper_id': rel.get('paper_id')
            }
            knowledge_graph['edges'].append(edge)
        
        return knowledge_graph
    
    
    def create_semantic_index(self, df: pd.DataFrame) -> Tuple[faiss.IndexFlatIP, List[Dict]]:
        """Create FAISS index for semantic search"""
        embeddings = []
        metadata = []
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Encoding Papers"):
            text = row['full_text'][:1000]
            embedding = self.sentence_model.encode(text)
            embeddings.append(embedding)
            metadata.append({
                'paper_id': row['paper_id'],
                'title': row['title'],
                'text_snippet': text[:200]
            })
        
        # Create FAISS index after processing all papers
        embeddings_array = np.array(embeddings).astype('float32')
        
        # Normalize embeddings for cosine similarity
        embeddings_array = embeddings_array / np.linalg.norm(embeddings_array, axis=1, keepdims=True)
        
        dimension = embeddings_array.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings_array)
        
        return index, metadata
    
    
    def semantic_search(self, query: str, index: faiss.IndexFlatIP, 
                       metadata: List[Dict], k: int = 10) -> List[Dict]:
        """Search for similar papers"""
        query_embedding = self.sentence_model.encode(query)
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        
        similarities, indices = index.search(
            query_embedding.reshape(1, -1).astype('float32'), k
        )
        
        results = []
        for sim, idx in zip(similarities[0], indices[0]):
            result = metadata[idx].copy()
            result['similarity'] = float(sim)
            results.append(result)
        
        return results
    
    
    def aggregate_relationships(self, all_relationships: List[Dict]) -> pd.DataFrame:
        """Aggregate relationships by strain-condition pair"""
        pair_data = defaultdict(lambda: {
            'papers': [],
            'confidence_scores': [],
            'relationship_types': [],
            'positive_outcomes': 0,
            'negative_outcomes': 0,
            'total_outcomes': 0
        })
        
        for rel in all_relationships:
            key = (rel['strain'], rel['condition'])
            
            pair_data[key]['papers'].append(rel.get('paper_id', 'unknown'))
            pair_data[key]['confidence_scores'].append(rel['confidence'])
            pair_data[key]['relationship_types'].append(rel['relationship_type'])
            
            for outcome in rel.get('outcomes', []):
                if outcome['type'] == 'positive':
                    pair_data[key]['positive_outcomes'] += 1
                else:
                    pair_data[key]['negative_outcomes'] += 1
                pair_data[key]['total_outcomes'] += 1
        
        aggregated_data = []
        for (strain, condition), data in pair_data.items():
            avg_confidence = np.mean(data['confidence_scores'])
            paper_count = len(set(data['papers']))
            
            type_counts = defaultdict(int)
            for rel_type in data['relationship_types']:
                type_counts[rel_type] += 1
            
            primary_type = max(type_counts.items(), key=lambda x: x[1])[0]
            
            if data['total_outcomes'] > 0:
                outcome_score = (data['positive_outcomes'] - data['negative_outcomes']) / data['total_outcomes']
            else:
                outcome_score = 0
            
            aggregated_data.append({
                'strain': strain,
                'condition': condition,
                'paper_count': paper_count,
                'avg_confidence': avg_confidence,
                'primary_relationship': primary_type,
                'positive_outcomes': data['positive_outcomes'],
                'negative_outcomes': data['negative_outcomes'],
                'outcome_score': outcome_score,
                'evidence_strength': paper_count * avg_confidence
            })
        
        return pd.DataFrame(aggregated_data).sort_values('evidence_strength', ascending=False)
    
    
    def generate_insights_report(self, aggregated_df: pd.DataFrame, 
                                knowledge_graph: Dict) -> Dict:
        """Generate insights from analysis"""
        insights = {
            'summary_statistics': {
                'total_relationships': len(aggregated_df),
                'unique_strains': aggregated_df['strain'].nunique(),
                'unique_conditions': aggregated_df['condition'].nunique()
            },
            'top_therapeutic_relationships': [],
            'most_studied_pairs': []
        }
        
        # Top therapeutic relationships
        therapeutic_df = aggregated_df[
            aggregated_df['primary_relationship'] == 'therapeutic'
        ].sort_values('outcome_score', ascending=False)
        
        for _, row in therapeutic_df.head(10).iterrows():
            insights['top_therapeutic_relationships'].append({
                'strain': row['strain'],
                'condition': row['condition'],
                'outcome_score': row['outcome_score'],
                'evidence': row['paper_count']
            })
        
        # Most studied pairs
        most_studied = aggregated_df.sort_values('paper_count', ascending=False)
        for _, row in most_studied.head(10).iterrows():
            insights['most_studied_pairs'].append({
                'strain': row['strain'],
                'condition': row['condition'],
                'papers': row['paper_count']
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
                    primary_relationship TEXT,
                    positive_outcomes INTEGER,
                    negative_outcomes INTEGER,
                    outcome_score REAL,
                    evidence_strength REAL,
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
        
        # Create semantic search index
        print("\nCreating semantic search index...")
        search_index, search_metadata = self.create_semantic_index(df)
        
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
            'insights': insights,
            'search_index': search_index,
            'search_metadata': search_metadata
        }
    
    
    def _print_summary(self, insights: Dict):
        """Print analysis summary"""
        print("\n=== Analysis Summary ===")
        print(f"Total relationships found: {insights['summary_statistics']['total_relationships']}")
        print(f"Unique strains: {insights['summary_statistics']['unique_strains']}")
        print(f"Unique conditions: {insights['summary_statistics']['unique_conditions']}")
        
        print("\nTop Therapeutic Relationships:")
        for rel in insights['top_therapeutic_relationships'][:5]:
            print(f"  {rel['strain']} -> {rel['condition']}: "
                  f"score={rel['outcome_score']:.2f}, evidence={rel['evidence']}")
        
        print("\nMost Studied Pairs:")
        for pair in insights['most_studied_pairs'][:5]:
            print(f"  {pair['strain']} + {pair['condition']}: {pair['papers']} papers")
        
         

            

        
            
                    
                
            
        
