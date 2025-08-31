Conda virtual environment is called 'venv'. (conda activate venv)

PubMedCollector (from pubmed_collector.py) collects papers that ivestigate probiotics and health
conditions. The Class checks if a paper has full text available for free in PMC (PubMed Central)
a free archive of biomedical litterature, or if it has a digital online identifier number (doi) so we can check other resources such as Unpaywall.

PaperParser (paper_parser.py) converts the PubMed API data that was temporary storred in XML format at src/data/raw/metadata to Python format, which is then storred in the SQLite database at src/data/processed as pubmed_research.db

FulltextRetriever (fulltext_retriever.py) checks if collected papers have free full text available in PMC (Pubmed Central) or Unpaywall by checking a list of papers.
