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
                    
        #* ?
        categorized_entities['outcomes'] = self._extract_outcomes(text)
        
        return categorized_entities
    
    def _extract_outcomes(self, text:str) -> List[Dict]:
        """_summary_
        """
        
        outcomes = []
        
        #* Split sentences:
        sentences = text.split('.')
        
        #* First 20 sentences of the abstract:
        for sentence in sentences[:20]:
            if len(sentence.strip()) <20:  #checks the number of charcters excluding white spaces
                continue
            
            #* Gets the PubMedBERT for semenatic embedding's output
            embedding = self.get_sentence_embedding(sentence)
            
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
            
        
            
            
        
        
            
                    
                
            
        
