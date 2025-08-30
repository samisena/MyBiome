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
load_dotenv(dotenv=env_path)  #loads the envirionemnt variables notably API keys to this file

class PubMedCollector:
    """
    """

    def __init__(self):

        #* Define pubmed api's base url:
        self.base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"

        self.api_key = os.getenv("NCBI_API_KEY") #our API key from .env

        #* Directory path where we will store papers temporarily
        self.data_dir = project_root/ "data" / "raw" / "papers"

        #* Directory path where we will temporarily store metadata 
        self.metadata_dir = project_root / "data" / "raw" / "metadata"

        self.db_manager = DatabaseManager  #initiate a databasemanager object
        self.parser = PubmedParser()  #initiate a paper parser object
        self.full_text_retriever = FullTextRetriever #initiate a full text retriever object


    def search_papers(self, min_year=2000, max_results=100) -> List:
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
        




