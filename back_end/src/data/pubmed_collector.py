from ast import Dict
import os  #system navigation
import time  #time stamps
import json  
import requests  #to make api requests
from dotenv import load_dotenv  #used to load from .env files
from pathlib import Path  #better than os for file naviagation
from typing import List  #add context to function documenation

from src.data.database_manager import DatabaseManager  #Creates and manages SQLite database
from src.data.paper_parser import PubmedParser  #Parses XML files to JSON
from src.data.fulltext_retriever import FullTextRetriever #Retrieves the full paper if available

# my_biome/src/data/script.py -> my_biome/
project_root = Path(__file__).parent.parent.parent

env_path = project_root / '.env'  #navigate to the .env file
load_dotenv(env_path)  #loads the envirionemnt variables notably API keys to this file

class PubMedCollector:
    """
    A collection of methods to query and collect papers relation to probiotics and health conditions
    using pubmed's API.
    """

    def __init__(self):

        #* Define pubmed api's base url:
        self.base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"

        self.api_key = os.getenv("NCBI_API_KEY") #our API key from .env

        #* Directory path where we will store papers temporarily
        self.data_dir = project_root/ "data" / "raw" / "papers"

        #* Directory path where we will temporarily store metadata 
        self.metadata_dir = project_root / "data" / "raw" / "metadata"

        self.db_manager = DatabaseManager()  #initiate a databasemanager object
        self.parser = PubmedParser()  #initiate a paper parser object
        self.full_text_retriever = FullTextRetriever() #initiate a full text retriever object
        
        # Create directories if they don't exist
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)


    def search_papers(self, query, min_year=2000, max_results=100) -> List:
        """
        This method queries the pubmed API via a get request according to 
        our search query and criteria filters.

        Args:
            min_year (int): the minimum release year of papers,set by default to 2000
            max_results (int): max results ouptput of the get request, set by default to 100
        Returns:
            pmid_list: a list of pubmed ids that match the search criteria
        """

        #* Will tell the API that both the query and min_year must be true
        query = f"{query} AND {min_year}[PDAT]:3000[PDAT]"

        #* Search endpoint (URL) format
        search_url = f"{self.base_url}esearch.fcgi"

        #* Parameters specific to the pubmed API
        params = {
            'db': 'pubmed', #pubmed database
            'term': query,
            'retmax':max_results,
            'retmode':'json', #response in JSON format
            'sort':'relevance',
            'api_key': self.api_key
        }

        #* Requests data from the API according to our search query and filters
        response = requests.get(search_url, params=params)

        if response.status_code != 200:  #if .status_code is different than 200 its an error
            print(f"Error searching PubMed: {response.status_code}")
            return []  #Interrupts search_papers() early and returns and empty list

        #* Converts JSON to python dictionary
        search_results = response.json()  #.json() is a method of requests.get()

        #* Gets the paper IDs 
        pmid_list = search_results["esearchresult"]["idlist"]

        if not pmid_list: #if id_list is empty == False
            print(f'No results found matching query criteria: {query}')
            return []

        print(f'Found {len(pmid_list)} papers from query: {query}')  #Number of papers found
        return pmid_list  


    def fetch_papers(self, pmid_list):
        """ 
        fetch papers from pubmed using their ids, then returns and saves the response data

        Args:
            pmid_list: the list of pubmed ids

        Returns: 
            A metadata file with the title, authors, abstarct etc 
        """

        if not pmid_list:  #if the list of paper ids is empty
            return []   #interrupts fetch_papers() 

        #* Join the ids into one string seprated by commas
        pmids = ",".join(pmid_list) 

        #* Gets paper details from paper IDs
        fetch_url = f"{self.base_url}efetch.fcgi"  #Fetch endpoint URL format

        params = {
            'db':'pubmed',
            'id': pmids,  #string of ids separated by commas
            'retmode':'xml',  #XML format
            'api_key': self.api_key
        }

        response = requests.get(fetch_url, params = params) #Gets the paper details

        if response.status_code != 200:  #if .status_code is different than 200 its an error
            print(f"Error fetching papers: {requests.status_code}")
            return []           

        #* Path to save the response as a file in data/raw/metadata/pubmed_batch_X.xml
        metadata_file = self.metadata_dir / f'pubmed_batch_{int(time.time())}.xml'
        
        #* Creates the file at the above location
        with open(metadata_file, 'w', encoding='utf-8') as f:  #create the file 
            f.write(response.text)  #writes the response to this file

        return metadata_file


    def collect_probiotics_by_condition(self, condition, min_year=2000, max_results=100) -> Dict:
        """
        Search for papers investigating probiotics for a given health condition

        Args:
            condition: the health condition we wish to investigate
        
        Returns:
            result: a dictionary with the search results
        """

        #* Search query in Pubmed's format
        query = f'''("{condition}"[Title/Abstract] OR "{condition}"[MeSH Terms])
            AND (probiotic*[Title/Abstract] OR "Probiotics"[MeSH Terms] OR 
            lactobacillus[Title/Abstract] OR bifidobacterium[Title/Abstract] OR 
            "lactic acid bacteria"[Title/Abstract] OR
            saccharomyces[Title/Abstract] OR
            synbiotic*[Title/Abstract])'''  # () because of Pubmed search format 
        
        #* Method that retrieves 1st 100 paper IDs (by default) according to search query
        pmid_list = self.search_papers(query, min_year, max_results)

        #* Return output if no paper is found
        if not pmid_list:  #if the no paper was retrieved (list is empty)
            return {
                "condition": condition,
                "paper_cout": 0,
                "status": "no_results"
            }

        #* Get the paper details from their IDs
        metadata_file = self.fetch_papers(pmid_list)

        #* Parse the XML output to a pyhton dictionary
        papers = self.parser.parse_metadata_file(metadata_file)

        #* Checks if the papers are available in full text
        fulltext_stats = None
        if papers:   #if the list is not empty == True
            print(f"Checking for full text availability for {len(papers)} papers...")
            self.full_text_retriever.create_directories() # '?'
            # for every entry in the papers dictionary retrieves pmc_id and doi entries
            papers_for_fulltext = [p for p in papers if p.get('pmc_id') or p.get('doi')] #.get() to avoi
                                                                                # crashes
            if papers_for_fulltext:  #if at least one paper has full text available in PMC
                                     # or a doi == True
                print(f"Found {len(papers_for_fulltext)} papers with PMC IDs")
                #* ?
                fulltext_stats = self.full_text_retriever.process_papers_batch(
                    papers_for_fulltext #list of PMC ids
                )
            else:
                print("No papers with PMC IDs or DOIs found")


        result = {
            "condition": condition,
            "paper_count": len(papers),
            "metadata_file": str(metadata_file),
            "status": "success"
        }

        #* Adds papers that have full text to the dictionary output
        if fulltext_stats:
            result["fulltext_stats"] = fulltext_stats

        return result

    
    def run_probiotic_collection_for_condition_list(self, conditions: List[str],
        max_results:int = 100) -> List[Dict]:
        """
        Returns and saves the results of a probiotic search given a list of health conditions.

        Args:
            conditions: the list of health conditions we want to investigate

        Returns:
            For each paper:
            {
            "condition": the medical condition,
            "paper_count": number of papers found that fit the search query,
            "metadata_file": file containing the paper details,
            "status": success or failure
            /Optional/
            "fulltext_stats": information about the full text if available
            }
        """

        collection_results = []

        total_queries = len(conditions)  #nbre of queries we will perform
        query_count = 0

        for condition in conditions:
            query_count += 1
            print(f"""\n[{query_count}/{total_queries}] Collecting probiotic papers
            for {condition}...""")
            result = self.collect_probiotics_by_condition(condition, max_results) #returns metatdata
                                                    # and checks if full text is available for free
            collection_results.append(result)  #adds this to the output
            time.sleep(1)  #For the API

        #* Saves the collection results in data/raw/metadata_dir/condition_collection_results_X
        results_file = self.metadata_dir / f"condition_collection_results_{int(time.time())}"
        with open(results_file, 'w') as f:
            #*Converts data to JSON before saving it
            json.dump(collection_results, f, indent =2)

        return collection_results  





                
        




