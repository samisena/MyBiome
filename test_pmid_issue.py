#!/usr/bin/env python3
"""Test if PubMed returns same papers for different conditions."""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from back_end.src.data_collection.pubmed_collector import PubMedCollector

collector = PubMedCollector()

conditions = ["diabetes", "hypertension", "cancer", "heart failure"]

for condition in conditions:
    # Search for papers
    query = collector._build_intervention_query(condition, include_study_filter=True)
    pmids = collector.search_papers(query, min_year=2020, max_results=3)
    print(f"\n{condition}:")
    for pmid in pmids:
        print(f"  {pmid}")