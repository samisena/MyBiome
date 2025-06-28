"""
Database management for the PubMed collection system.
This module handles all SQLite operations.
"""

import sqlite3
import json
from pathlib import Path
from typing import List, Dict, Optional
from contextlib import contextmanager  #? context for "with" statements
import logging

# Import the shared data models
from src.models import Author, Paper

# Get project root 
project_root = Path(__file__).parent.parent.parent

class DatabaseManager:
    """ ?
    """
    
    def __init__(self, db_name: str = 'pubmed_research.db'):
        """ Initiates the database
        
        Args:
            db_name (str, optional): Defaults to 'pubmed_research.db'.
        """
        
        #* Setting up the database
        self.db_path = project_root/"data"/"processed"/db_name
        self.db_path.parent.mkdir(parents=True, exist_ok=True) #Creates the data directory 
                                                # if it doesn't exist
                                                
        #* Setting up logging:
        logging.basicConfig(level=logging.INFO) #console will show INFO level logs
        self.logger = logging.getLogger(__name__) #console will show wich class/function
                                                #the log is coming from
                                    #? __name__ takes the value of he current module's name
        
        #* ?                            
        self.create_tables()
        
        
    @contextmanager  
    def get_connection(self):
         """ We define how the database interacts with Python with statements """
         
         #* Connect to the SQLite database
         conn = sqlite3.connect(self.db_path)  #path to the database
         conn.row_factory = sqlite3.Row #Returns rows as as iteratble dictionary-like
                        # objects instead of tuples
         try:
             yield conn  # the code stops here untill code interrupts
         finally:  #when the code interupts:
            conn.close()
            
            
    def create_tables(self):
        """ """
        
        #* Connect to the database using get_connection() method
        with self.get_connection() as conn:
            cursor = conn.cursor  #cursor keeps track of where we are in the database
            
            #* We create the Papers table:
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS papers (
                    pmid TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    abstract TEXT,
                    journal TEXT,
                    publication_date TEXT,
                    doi TEXT,
                    keywords TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                ''')
            
            #* We create the Author's table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS authors (
                    author_id INTEGER PRIMARY KEY AUTOINCREMENT, 
                    -- SQL generates the ids
                    last_name TEXT NOT NULL,
                    first_name TEXT,
                    initials TEXT,
                    affiliations TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(last_name, first_name, initials, affiliations) 
                    -- The combination of these columns must be unique (composite unique constraint)
                )
                ''')
            
            #* Junction table for papers-authors relationship
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS papers_authors (
                    paper_pmid TEXT,
                    author_id INTEGER,
                    author_order INTEGER,
                    PRIMARY KEY (paper_pmid, author_id),
                    -- This combination of columns is the primary key and must be unique 
                    FOREIGN KEY (paper_pmid) REFERENCES papers(pmid),
                    FOREIGN KEY (author_id) REFERENCES authors(author_id)
                )
                ''')
            
            #* Search history table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS search_history (
                    search_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    strain TEXT,
                    condition TEXT,
                    query TEXT,
                    result_count INTEGER,
                    search_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP   
                )
                ''')
            
            #* Search to paper linking table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS search_results(
                    search_id INTEGER,
                    paper_pmid TEXT,
                    PRIMARY KEY (searhc_id, paper_pmid) 
                    -- The junction of the 2 tables is the primary key
                    FOREIGN KEY (search_id) REFERENCES search_history(search_id),
                    FOREIGN KEY (paper_pmid) REFERENCES papers(pmid)
                )
                ''')
            
            #* Indexes for easy querying
            cursor.execute('''CREATE INDEX IF NOT EXISTS idx_papers_date 
                           ON papers(publication_date)
                           ''')
            cursor.execute('''CREATE INDEX IF NOT EXISTS idx_authors_name
                           ON authors(last_name, first_name)
                           ''')
            
            conn.commit()
            self.logger.info(f"Database tables created at {self.db_path}")  
            #? Will output in the console that the database was created
            
    
    
    def insert_papers(self, paper: Paper, search_id: Optional[int]=None) -> bool:
        """Inserts a paper and its authors into the database. This method handles
        the complexity of the many-to-many relationship automatically.

        Args:
            paper (Paper): takes the data class Paper
            search_id (Optional[int], optional): Defaults to None.

        Returns:
            bool: True if the paper was newly inserted, False if it already existed.
        """
        
        with self.get_connection as conn: #? connects to the database
            cursor = conn.cursor()
            
            #* Tries to insert a paper's data to the database
            try:
                cursor.execute('''
                    INSERT OR IGNORE INTO papers  
                    -- ignores if pmid is a duplicate
                    (pmid, title, abstract, journal, publication_date, doi, keywords)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    ''', (
                    paper.pmid,              # goes to pmid column
                    paper.title,             # title column  
                    paper.abstract,          # abstract column
                    paper.journal,           # journal column
                    paper.publication_date,  # publication_date column
                    paper.doi,               # doi column
                    json.dumps(paper.keywords) if paper.keywords else None #keywords 
                            #? converts python list to JSON string (SQL TEXT) 
                    ))
                
                was_new_paper = cursor.rowcount > 0  #? returns TRUE if the paper got added
                                                    #? meaning it wasn't a duplicate
                #* Add the authors
                for order, author in enumerate(paper.authors):
                    cursor.execute('''
                                   INSERT OR IGNORE INTO authors
                                   (last_name, first_name, initials, affiliations)
                                   VALUES (?, ?, ?, ?)
                                   ''', (
                                    author.last_name,
                                    author.first_name,
                                    author.intials,
                                    author.affiliations
                                   ))
                    
                cursor.execute('''
                               SELECT author_id FROM authors
                               WHERE last_name = ? AND first_name = ?
                               AND initials = ? AND affiliations IS ?
                               ''',(
                                    author.last_name,
                                    author.first_name,
                                    author.intials,
                                    author.affiliations
                               ))
                
                #* Retrieve the result of the SELECT statement above 
                author_id = cursor.fetchtone()[0]
                
                
                #* Filling the papers_authors table:
                cursor.execute('''
                    INSER OR IGNORE INTO papers_authors
                    (paper_pmid, author_id, author_order)
                    VALUES (?,?,?) 
                    ''', (paper.pmid, author_id, order))
                
                
                #* Filling the search_results table:
                if search_id is not None:
                    cursor.execute('''
                        INSERT OR IGNORE INTO search_results
                        (search_id, paper_pmid)
                        VALUES (?, ?)
                        ''', (search_id, paper.pmid))
                
                conn.commit()
                
                
                if was_new_paper():   #if a new paper was added to the database
                    self.logger.info(f"Inserted new paper: {paper.title}")
                else:
                    self.logger.debug(f"Paper already exists: {paper.title}")
                    
                return was_new_paper  #Returns True or False (Boolean)
            
            except Exception as e:  
                conn.rollback()  # like ctr+Z for databases - if something goes wrong 
                                #undo the previous operations to not leave 
                                #the database in an inconsitent state
                self.logger.error(f"Error inserting paper {paper.pmid}: {e}")
                raise
            
    def record_search(self, strain: str, condition: str, query: str, 
                      result_count: int):
        """Records a search in the search_hisotry table and returns the search_id

        Args:
            strain (str): probiotic strain
            condition (str): healht condition
            query (str): full query
            results_count (int): number of results papers
        """
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO search_history (strain, condition, query, result_count)
                VALUES (?, ?, ?, ?)
                ''', (strain, condition, query, result_count))
            conn.commit()
            return cursor.lastrowid  #the row id of the new entry
        
    def get_papers_by_strain(self, strain: str):
        """
        
        
        Returns:
        list: a list of dictionaries where each dictionary represent a row - column names
        as keys
        """
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT DISTINCT p.* FROM papers p 
                -- p stand for papers and p.* means select all columns
                JOIN search_results sr ON p.pmid = sr.paper_pmid
                JOIN search_results sh ON sr.search_id = sh.search_id
                WHERE sh.strain = ?
                ORDER BY p.publication_date DESC
                ''', (strain)) #comma to create a tuple object
            return [dict(row) for row in cursor.fetchall()]   #returns a list of dictionary pairs
                                            #converts to Python dict to avoid errors
                                            
                                            
    def get_papers_by_author(self, last_name: str, first_name: Optional[str] = None 
                             -> List[Dict]):
        """Find all papers by a specific author. Notice how the junction table
        allows us to navigate the many-to-many relationship easily.
        """
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            if first_name:   #if the author has a first name cited
                query = '''
                           SELECT p.* FROM papers p 
                           JOIN paper_authors pa ON p.pmid = pa.paper_pmid 
                           JOIN authors a ON pa.author_id = a.author_id
                           WHERE a.last_name = ? AND a.first_name = ?
                           ORDER BY p.publication_date DESC
                           '''
                params = (last_name, first_name)
                
            else:
                query = '''
                            SELECT p.* FROM papers p
                            JOIN paper_authors pa ON p.pmid = pa.paper_pmid
                            JOIN authors a ON pa.author_id = a.author_id
                            WHERE a.last_name = ?
                            ORDER BY p.publication_date DESC
                        '''
                params = (last_name,)
                
            cursor.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]
        
    def get_database_stats(self) -> Dict:
        """
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            stats = {}  #empty dictionary
            
            #* count the total number of papers
            cursor.execute('SELECT COUNT(*) FROM papers') 
            stats['total_papers'] = conn.fetchone()[0]  
                            # or = conn.fetchall()[0][0]
            
            #* count the number of authors
            cursor.execute('SELECT COUNT(*) FROM authors')
            stats['total_authors'] = conn.fetchone()[0]
            
            #* count the nbre of searches
            cursor.execute('SELECT COUNT(*) FROM search_history')
            stats['total_searches'] = conn.fetchone()[0]
            
            #* date range of publications
            cursor.execute('''SELECT MIN(publication_date), MAX(publication_date)
                           FROM papers
                           ''')
            min_date, max_date = cursor.fetchone()
                # or min_date = fetchone()[0]
                #    max_date = fetchone()[1]
            stats['date_range'] = f"{min_date} to {max_date}"
            
            return stats
            
        
                
                    
                    
        