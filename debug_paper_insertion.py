"""
Debug script to test paper insertion and identify why papers aren't being saved.
"""
import sqlite3
import json
from pathlib import Path

from back_end.src.data_collection.database_manager import database_manager
from back_end.src.data_collection.paper_parser import PubmedParser

def test_paper_insertion():
    print("=" * 80)
    print("PAPER INSERTION DEBUG TEST")
    print("=" * 80)
    print()

    # Find a sample XML file to test
    metadata_dir = Path("back_end/data/raw/metadata")
    xml_files = list(metadata_dir.glob("pubmed_batch_*.xml"))

    if not xml_files:
        print("ERROR: No XML files found in", metadata_dir)
        return

    test_file = xml_files[0]
    print(f"Testing with XML file: {test_file.name}")
    print()

    # Parse the XML file
    parser = PubmedParser(database_manager)
    print("Parsing XML file...")
    papers = parser.parse_metadata_file(str(test_file))

    print(f"Parsed {len(papers)} paper(s)")
    print()

    if not papers:
        print("ERROR: No papers parsed from XML")
        return

    # Test first paper
    test_paper = papers[0]
    print("TEST PAPER DATA:")
    print("-" * 80)
    for key, value in test_paper.items():
        if isinstance(value, str) and len(value) > 100:
            print(f"  {key}: {value[:100]}...")
        else:
            print(f"  {key}: {value}")
    print()

    # Test validation
    print("VALIDATION TEST:")
    print("-" * 80)
    from back_end.src.data.validators import validation_manager

    validation_result = validation_manager.validate_paper(test_paper)

    print(f"Valid: {validation_result.is_valid}")
    print()

    if not validation_result.is_valid:
        print("VALIDATION ERRORS:")
        for issue in validation_result.errors:
            print(f"  - {issue.field}: {issue.message} (value: {issue.value})")
        print()

    if validation_result.warnings:
        print("VALIDATION WARNINGS:")
        for issue in validation_result.warnings:
            print(f"  - {issue.field}: {issue.message}")
        print()

    # Test database insertion
    print("DATABASE INSERTION TEST:")
    print("-" * 80)

    # Check if paper already exists
    conn = sqlite3.connect('back_end/data/processed/intervention_research.db')
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM papers WHERE pmid = ?", (test_paper['pmid'],))
    exists_before = cursor.fetchone()[0]
    print(f"Paper exists before insert: {exists_before > 0}")

    # Try to insert
    print("Attempting insertion...")
    result = database_manager.insert_paper(test_paper)

    print(f"Insert result: {result}")
    print()

    # Verify in database
    cursor.execute("SELECT COUNT(*) FROM papers WHERE pmid = ?", (test_paper['pmid'],))
    exists_after = cursor.fetchone()[0]
    print(f"Paper exists after insert: {exists_after > 0}")

    if exists_after > 0:
        cursor.execute("""
            SELECT pmid, title, abstract, processing_status
            FROM papers
            WHERE pmid = ?
        """, (test_paper['pmid'],))
        row = cursor.fetchone()
        print()
        print("PAPER IN DATABASE:")
        print(f"  PMID: {row[0]}")
        print(f"  Title: {row[1][:60]}...")
        print(f"  Abstract: {row[2][:60] if row[2] else 'NULL'}...")
        print(f"  Status: {row[3]}")
    else:
        print()
        print("ERROR: Paper NOT in database after insertion!")

    conn.close()

    print()
    print("=" * 80)

if __name__ == "__main__":
    test_paper_insertion()
