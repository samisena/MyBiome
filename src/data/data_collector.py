import os
import time
import json
import requests   #? Used to communicate with APIs
from dotenv import load_dotenv  #? used to load from .env
from pathlib import Path

# Create a path to the project root (3 levels up from your script)
# project/src/data/script.py -> project/
project_root = Path(__file__).parent.parent.parent

# Load the .env file from the project root
env_path = project_root / '.env'
#* Using the dotenv module to load our variables form .env=
load_dotenv(dotenv_path=env_path) 

class PubMedCollector:
    """The class interacts with PubMed's E-utilities API to:
        1.Search for paper IDs matching specific criteria (strains and conditions)
        2.Fetch paper details in XML format from those IDs
        3.Save this metadata to files with timestamps
    """ 
    def __init__(self):
        #Base URL for PubMed API  
        self.base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
        #Our personal Access Key
        self.api_key = os.getenv("NCBI_API_KEY")
        #Setting up the data save paths 
        self.data_dir = project_root / "data" / "raw" / "papers"
        self.metadata_dir = project_root / "data" /"raw" /"metadata"
        
    def search_papers(self, query, max_results=100, min_year=2000):
        """Search for papers that match the query criterias

        Args:
            query (_type_): _description_
            max_results (int, optional): _description_. Defaults to 100.
            min_year (int, optional): _description_. Defaults to 2000.

        Returns:
            _type_: _description_
        """
        #API search query format - both query and min_year must be TRUE
        query = f"{query} AND ({min_year}[PDAT]:3000[PDAT])"
        #Search endpoint format
        search_url = f"{self.base_url}esearch.fcgi"
        #Search parameters
        params ={
            "db": "pubmed", #pubmed database
            "term": query,
            "retmax": max_results,
            "retmode": "json", #asking for a JSON in return of a GET request
            "sort": "relevance",
            "api_key": self.api_key
        }
        #Requests data from pubmed using our params dictionary
        response = requests.get(search_url, params=params)
        #! Error handling:
        #A status_code response other than 200 means an error happened
        if response.status_code != 200:
            print(f"Error searching PuMed: {response.status_code}")
            #! early return
            return []
        #Convert JSON to Python dictionary for handling
        search_results = response.json()
        #Get the IDs of the papers
        id_list = search_results["esearchresult"]["idlist"]
        #! Error handling:
        #if not id_list is TRUE - meaning id_list is FALSE - meaning empty
        if not id_list:
            print(f'No results found matching query criteria: {query}')
            #! early return
            return []
        print(f"Found {len(id_list)} papers from query: {query}")
        return id_list
            
            
    def fetch_paper_details(self, pmid_list):
        """Returns the paper details of a paper given its ID

        Args:
            pmid_list (list): list of pubmed paper IDs

        Returns:
            _type_: _description_
        """
        #! Error Handling:
        if not pmid_list:   #if pmed_list is empty
            return []   #early return
        pmids = ",".join(pmid_list)   #turns the list into a string seprated with commas
        fetch_url = f"{self.base_url}efetch.fcgi"  # fetch endpoint format
        params = {
            "db" : 'pubmed',
            "id" : pmids,
            "retmode" : "xml",  #papers in XML format
            "api_key": self.api_key
        }
        response = requests.get(fetch_url, params=params)
        #! Error Handling:
        if response.status_code != 200:
            print(f"Error fetching papers: {requests.status_code}")
            return []
        # Stores the paper details in XML format
        metadata_file = self.metadata_dir / f"pubmed_batch_{int(time.time())}.xml"
        with open(metadata_file, 'w', encoding="utf-8") as f:  #opens the file at the path we just created
            f.write(response.text)   #writes the contents of the response to the file 
        return metadata_file
    
    def collect_by_strain_and_condition(self, strain, condition, max_results=20):
        """Searchs for papers that investigate a probiotic strain and a health condition

        Args:
            strain (str)
            condition (str)
            max_results (int, optional): Defaults to 20.

        Returns:
            _dict: a dictionary with the metadata info
        """
        #Query combining a probiotic strain and health condition/outcome
        query = f'"{strain}"[Title/Abstract] AND "{condition}"[Title/Abstract] AND "clinical trial"[Publication Type]'
        #Gets the IDs of papers matching the query
        paper_ids = self.search_papers(query, max_results)
        #! Error handling:
        if not paper_ids:
            return []
        #Gets the paper details
        metadata_file = self.fetch_paper_details(paper_ids)
        return {
            "strain" : strain,
            "condition" : condition,
            "paper_count" : len(paper_ids),
            "metadata_file" : str(metadata_file)
        }
        
    def run_collection_for_list(self, strains, conditions, results_per_query=10):
        collection_results = []
        for strain in strains:
            for condition in conditions:
                print(f"Collecting papers for {strain} and {condition}...")
                result = self.collect_by_strain_and_condition(strain, condition, results_per_query)
                if result:  #if result is not empty
                    collection_results.append(result)
                time.sleep(1)  #pause between requests
        # save record of searches
        results_file = self.metadata_dir / "collection_results.json"
        with open(results_file, 'w') as f:
            json.dump(collection_results, f, indent=2) #converts JSON to Python
        return collection_results
            


#! Sample data for testing
test_strains = [
    "Lactobacillus acidophilus",
    "Bifidobacterium bifidum"
]

test_conditions = [
    "irritable bowel syndrome",
    "antibiotic-associated diarrhea"
]

if __name__ == "__main__":
    # Check for API key
    if not os.getenv("NCBI_API_KEY"):
        print("Error: No NCBI API key found. Make sure you have a .env file with NCBI_API_KEY defined.")
        exit(1)
    
    # Initialize the collector
    collector = PubMedCollector()
    
    print(f"Running collection for {len(test_strains)} strains and {len(test_conditions)} conditions...")
    print(f"This will make {len(test_strains) * len(test_conditions)} API calls")
    
    # Run the collection
    results = collector.run_collection_for_list(test_strains, test_conditions, results_per_query=5)
    
    if results:
        print("\nCollection complete!")
        print(f"Collected data for {len(results)} strain-condition combinations")
        
        # Display results
        print("\nResults summary:")
        for result in results:
            print(f"- {result['strain']} + {result['condition']}: {result['paper_count']} papers")
            print(f"  Metadata saved to: {result['metadata_file']}")
        
        print("\nFull collection results saved to: data/raw/metadata/collection_results.json")
    else:
        print("No results were collected.")
