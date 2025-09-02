Conda virtual environment is called 'venv'. (conda activate venv)

PubMedCollector (from pubmed_collector.py) collects papers that ivestigate probiotics and health
conditions. The Class checks if a paper has full text available for free in PMC (PubMed Central)
a free archive of biomedical litterature, or if it has a digital online identifier number (doi) so we can check other resources such as Unpaywall.

PaperParser (paper_parser.py) converts the PubMed API data that was temporary storred in XML format at src/data/raw/metadata to Python format, which is then storred in the SQLite database at src/data/processed as pubmed_research.db

FulltextRetriever (fulltext_retriever.py) checks if collected papers have free full text available in PMC (Pubmed Central) or Unpaywall and downloads the paper if a free link is found.

retrieve_fulltext.py is a file that when run uses FulltextRetriever to retrieve full text for papers in the database.

DatabaseManager (database_manager.py) this function has 19 different methods to handle everyhting related to the database. The database itself has 2 main tables: papers and correlations. The methods of this class include ways to insert papers and correlations to the database, query papers by condition/probiotic strain etc, delete papers from the database, update information, get general information about the database and more.

explore_database() function (database_explorer) uses many methods of the DatabaseManager class to output an overview that includes data such as unique probiotic strains, numbre of correlations, top journals etc

ProbioticAnalyzer and LLMConfig Classes (probiotic_analyzer.py) are two classes that define how we plan to setup the our LLM to extract condition-probiotic correlations from either the abstract only or full text if available.
