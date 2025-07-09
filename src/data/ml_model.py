import sqlite3
from pathlib import Path
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
import torch    
from transformers import (                           #?Huggig Face imports
    AutoTokenizer,
    AutoModel,
    AutoModelForTokenClassification,
    AutoModelForSequenceClassification,
    pipeline
)
from sentence_transformers import SentenceTransformer  #?converts sentences to vector embeddings
import faiss                                           #?Facebook AI Similarity Search
from sklearn.metrics.pairwise import cosine_similarity #? computes cosine similarity between vectors
from collections import defaultdict
import json
import os
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')  #ignore redundant warnings

from src.data.models import Author, Paper
from src.data.database_manager import DatabaseManager
from src.data.paper_parser import PubmedParser


project_root = Path(__file__).parent.parent.parent


class ProbioticAnakyzer:
    """_summary_
    """
    
    def __init__(self):
        self.db_path = project_root / "data" / "processed" / 'pubmed_research.db'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        #* Initializes LLM models as defined in the method below
        self._initiliaze_models()
        
    
    def _initialize_models(self):
        """ Initalizes our LLMs
        """
        
        print("Loading models...")
        
        #* 1. Named Entity Recognition using BioBERT
        #? Loads BioBERT for the purpose of NER
        self.ner_pipeline = pipeline(
            "ner",
            model = "dmis-lab/biobert-base-cased-v1.2",
            aggregation_strategy = "simple",
            device=0 if torch.cuda.is_available() else -1 #uses gpu or cpu
        )
        
        #* 2. Semantic embeddings using PubMedBERT
        self.sentence_model = SentenceTransformer('pritamdeka/S-PubMedBert-MS-MARCO')
        
        #* 3. Relation extraction using BioBERT
        #? Loads BioBERT's tokenizer
        self.relation_tokenizer = AutoTokenizer.from_pretrained(  
            "dmis-lab/biobert-base-cased-v1.2"
        )    
        self.relation_model = AutoModel.from_pretrained(
            "dmis-lab/biobert-base-cased-v1.2"
            ).to(self.device)
        
        #* 4. PubMedBERT for sentiment/outcome classification
        self.outcome_tokenizer = AutoTokenizer.from_pretrained(
            "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"
        )
        self.outcome_model = AutoModel.from_pretrained(
            "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"
            ).to(self.device)
        
        print("All models loaded successfully!")
        
    
    def prepare_data_from_db(self) -> pd.DataFrame:
        """ Loads data from our SQLite database
        Returns:
            pd.DataFrame: _description_
        """
        
        #* Retrieve paper abstracts from the SQL database
        conn = sqlite3.connect(self.db_path)
        query = """
                SELECT
                    paper_id,
                    title,
                    abstract,
                    keywords,
                    publication_date,
                    strain,
                    condition
                FROM papers
                WHERE abstract IS NOT NULL AND abstract != ''
        """
        
        #* SQl to pandas DataFrame
        try:
            df = pd.read_sql_query(query, conn)  #special pandas function
            #* Combine the title with the abstract
            df['full_text'] = df['title'].fillna('') + ' ' + df['abstract'].fillna('')
            print(f"Loaded {len(df)} papers from database")
            return df
        finally:
            conn.close()
            
            
    def extract_entities(self, text: str) -> Dict[str: List[Dict]] :
        """Uses our BioBERT NER to extract strains, conditions, outcomes and mechanisms
        """
        
        entities = self.ner_pipeline(text[:512])  #BioBERT has a token limit of 512
        
        categorized_entities = {
            'strains': [],
            'conditons': [],
            'outcomes': [],
            'mechanisms': []
        }
        
        strain_labels = ['GENE', 'PROTEIN']  #bacteria are often tagged as genes or proteins
        condition_labels = ['DISEASE', 'PHENOTYPE']
        
        #* Going through the entities BioBERT extracted
        for entity in entities:
            #?
            entity_info = {
                'text': entity['word'],   #the entity 
                'score': entity['score'], #confidence score
                'start': entity['start'], #start of the word in the text
                'end': entity['end']      #end of the word in the text
            }
            
            #* Checking if BioBERT NER found a probiotic strain in the text
            if entity['entity_group'] in strain_labels:
                #* Returns True if one of these terms is found:
                if any(term in entity['word'].lower() for term in 
                       ['bacterium', 'bacillus', 'coccus', 'bididum', 'lactis']):
                    categorized_entities['strains'].append(entity_info)
            elif entity['entity_group'] in condition_labels:
                    categorized_entities['conditions'].append(entity_info)
                    
        #* Defines whether the abstract discussed a positive or negative outcome
        categorized_entities['outcomes'] = self._extract_outcomes(text)
        
        return categorized_entities
    
    def _extract_outcomes(self, text:str) -> List[Dict]:
        """Breaks down the Abstract text into individual sentences. Then assess
            wheter they are discussing a positive or negative outcome.
        """
        
        outcomes = []
        
        #* Split sentences:
        sentences = text.split('.')
        
        #* First 20 sentences of the abstract:
        for sentence in sentences[:20]:
            if len(sentence.strip()) <20:  #checks the number of charcters excluding white spaces
                continue
            
            #* Converts text to embedding using PubMedBERT: 
            embedding = self.sentence_model.encode(sentence)  #shape (768,) np array
            
            #* Clear examples of positive and negative outcomes
            positive_refs = [
                "The treatment was effective and imporved symptoms",
                "Siginificant improvement was observed",
                "The intervention successfully reduced disease markers"
            ]
            
            negative_refs = [
                "The treatment was ineffective",
                "No improvement was observed",
                "Adverse effects were reported"
            ]
            
            #* Calling PubMedBERT for Semantic Embeddings
            pos_embeddings = self.sentence_model.encode(positive_refs)
            neg_embeddings = self.sentence_model.encode(negative_refs)
            
            #* Calculates sentence similarity to the reference positives and negatives
            pos_sim = np.mean([cosine_similarity([embedding], [ref_emb])[0][0]
                              for ref_emb in pos_embeddings])
            
            neg_sim = np.mean([cosine_similarity([embedding], [ref_emb])[0][0]
                              for ref_emb in neg_embeddings])
            
            #* Filter low correlation outcomes:
            if pos_sim > 0.7 or neg_sim > 0.7:
                outcomes.append({
                    'text': sentence.strip(),
                    'type': 'positive' if pos_sim > neg_sim else 'negative',
                    'confidence': max(pos_sim, neg_sim),
                    'embedding': embedding
                })
        
        return outcomes
    
    
    def strain_condition_relationships(self, paper_data: Dict) -> List[Dict]:
        """ ?
        """
        
        #* Store the full text (abstract) as 'full_text' key in paper_data dictionary
        text = paper_data['full_text']
        
        #* Extract strain and condition entities from the text using BioBERT
        entities = self.extract_entities(text)
        relationships = []
        
        #* For every strain and every condition possible combination:
        for strain in entities['strains']:
            for condition in entities['conditions']:
                #* Checks if the 2 entities are within 500 characters of one another
                #* and extracts a context window of text
                context = self._extract_context_window(
                    text, strain['start'], strain['end'],
                    condition['start'], condition['end']
                )
                #* 
                if context:
                    relationship_info = self._analyze_relationship(
                        context, strain['text'], condition['text']
                    )
                    
                    relevant_outcomes = self._find_relevant_outcomes(
                        strain, condition, entities['outcomes'], text
                    )
                    
                    relationships.append(
                        {
                        'strain': strain['text'],
                        'condition': condition['text'],
                        'confidence': relationship_info['confidence'],
                        'relationship_type': relationship_info['type'],
                        'context': context,
                        'outcomes': relevant_outcomes,
                        'paper_id': paper_data.get('paper_id', 'unknown')
                        }
                    )
                    
        if paper_data.get('strain') and paper_data.get('condition'):
            full_relationship = self._analyze_full_text_relationship(
                text, paper_data['strain'], paper_data['condition'], entities['outcomes']
            )
            relationships.append(full_relationship)


    def _extract_context_window(self, text: str, start1: int, end1: int,
                                start2:int, end2: int, window:int = 200 
                                ) -> Optional[str]:
        
        """ Checks if 2 named entities are within 500 characters of one another. If so the
        function extracts a 'context window' of 200 characters before and after each
        entity
        """
        
        #* 200 charcters before the start of the first named entity 
        context_start = max(0, min(start1, start2) - window)
        
        #* 200 characters after the end of the second named entity
        context_end = min(len(text), max(end1, end2) + window)
        
        #* if entities are further than 500 characters from each other:
        if abs(start1 - start2) >500:
            return None
        
        #* returns the context window
        return text[context_start:context_end]  # truncates the abstract 
                                                # to the context window
                                                
    def _analyze_relationship(self, context: str, strain: str, condition:str
                              ) -> Dict:
        """ 
        
        """
        
        #* Marks the beginning and end of named entities within the context window text
        marked_context = context.replace(strain, f"[STRAIN]{strain}[/STRAIN]")
        marked_context = marked_context.replace(condition, f"[CONDITION]{condition}[/CONDITION]")
        
        #* BioBERT Tokenizer breaks the text into individual Tokens that BioBERT recognizes - returns a Dictionary
        inputs = self.relation_tokenizer(
            marked_context,
            return_tensors = "pt",
            max_length=512,
            padding=True, 
            truncation=True,
            ).to(self.device)
        
        with torch.no_grad():  #since we are inferring not training we don't need to track gradients
            outputs = self.relation_model(**inputs) #? Dictionary unpacking
            #* BioBERT stores the semantic embedding of the text in it's first taken [CLS]
            relationship_embedding = outputs.last_hidden_state[:,0, :].cpu().numpy()[0]
            
        #* Frame of reference for relationships between strain and conditions
        relationship_types = {
            'therapeutic': "The probiotic strain treats and improves the condition",
            'preventive': "The probiotic strain prevents the condition",
            'associated': "The probiotic strain is associated with the condition",
            'no_effect': "The probiotic strain has no effect on the condition"            
        }
        
        #* This dictionary will contain the sementic embedding vector of each one of the reference sentences
        type_embeddings = {}
        
        #* for every reference relationship_types dictionary
        for rel_type, description in relationship_types.items():
            
            #* BioBERT tokenization of each reference sentence - returns a Dictionary
            inputs = self.relation_tokenizer(
                description,
                return_tensor='pt', #returns PyTorch
                max_length=512,
                truncation=True, #cut_off at character 512 to avoid errors
            ).to(self.device)
            
            #* Sementic embedding of each reference sentence 
            with torch.no_grad():
                outputs = self.relation_model(**inputs) #? applies dictionary unpacking and outputs
                                                        #? semantic embedding 
                #* 1st token [CLS]: [batch_size, sequence_length, 768] -> [1,768]                                       
                type_embeddings[rel_type] = outputs.last_hidden_state[:, 0, :].cpu().numpy() 
                
        #* Dictionary of similarities between reference sentences and text inputs (relationship_embedding)       
        similarities = {}
        
        #* Iterates over type_embeddings dictionary with the vector embeddings of each reference sentence:
        for rel_type, type_emb in type_embeddings.items():
            #* geometric cosine similarity between the 2 vector embedding representation (1D matrices)
            sim = cosine_similarity([relationship_embedding], [type_emb])[0][0]
            similarities[rel_type] = sim
            
        #* Picks the most simalar reference sentence
        best_type = max(similarities.items(), key = lambda x: x[1])
        
        return {
            'type': best_type[0],
            'confidence': best_type[1],
            'all_scores': similarities
        }

            

        
            
                    
                
            
        
