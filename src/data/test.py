import os
import time
import json
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

print("=" * 60)
print("PubMed Pipeline Test - COMPLETE VERSION")
print("=" * 60)

# Step 1: Check prerequisites
print("\n1. Checking prerequisites...")
if not os.getenv("NCBI_API_KEY"):    
    print("❌ Error: No NCBI API key found.")
    print("   Make sure you have a .env file with NCBI_API_KEY defined.")
    exit(1)
else:
    print("✓ API key found")

# Step 2: Test the Collector
print("\n2. Testing PubMedCollector...")
print("-" * 40)

# Import and initialize the collector
from src.data.data_collector import *
collector = PubMedCollector()

# Test 1: Search for papers
print("\nTest 1: Searching for papers...")
test_query = "Lactobacillus acidophilus AND Obesity"
print(f"Query: {test_query}")

try:
    paper_ids = collector.search_papers(test_query, max_results=5)
    if paper_ids:
        print(f"✓ Found {len(paper_ids)} papers")
    else:
        print("⚠ No papers found (this might be okay if the query is very specific)")
except Exception as e:
    print(f"❌ Error during search: {e}")
    exit(1)

# Test 2: Fetch paper details
if paper_ids:
    print("\nTest 2: Fetching paper details...")
    try:
        test_ids = paper_ids[:2]
        metadata_file = collector.fetch_paper_details(test_ids)
        
        if metadata_file and Path(metadata_file).exists():
            file_size = Path(metadata_file).stat().st_size
            print(f"✓ Metadata saved to: {metadata_file}")
            print(f"  File size: {file_size} bytes")
        else:
            print("❌ Metadata file not created")
    except Exception as e:
        print(f"❌ Error fetching details: {e}")

# Test 3: Test the complete collection method
print("\nTest 3: Testing complete collection for strain-condition pair...")
try:
    result = collector.collect_by_strain_and_condition(
        strain="Bifidobacterium",
        condition="constipation",
        max_results=3
    )
    if result:
        print(f"✓ Collection successful")
        print(f"  Strain: {result['strain']}")
        print(f"  Condition: {result['condition']}")
        print(f"  Papers found: {result['paper_count']}")
    else:
        print("⚠ No results returned")
except Exception as e:
    print(f"❌ Error in collection: {e}")

# Small pause before parser tests
time.sleep(1)

# Step 3: Test the Parser
print("\n\n3. Testing PubmedParser...")
print("-" * 40)

# Import and initialize the parser
from src.data.paper_parser import *
parser = PubmedParser()

# Test 4: Check if we have XML files to parse
print("\nTest 4: Looking for XML files to parse...")
xml_files = list(Path("data/raw/metadata").glob("pubmed_batch_*.xml"))

if xml_files:
    print(f"✓ Found {len(xml_files)} XML files")
    
    # Test parsing a single file
    print("\nTest 5: Parsing a single XML file...")
    test_file = xml_files[0]
    print(f"Testing with: {test_file.name}")
    
    try:
        papers = parser.parse_metadata_file(test_file)
        
        if papers:
            print(f"✓ Parsed {len(papers)} papers from file")
            
            # Show details of first paper as verification
            first_paper = papers[0]
            print("\nFirst paper details:")
            print(f"  PMID: {first_paper.pmid}")
            print(f"  Title: {first_paper.title[:60]}...")
            print(f"  Authors: {len(first_paper.authors)} author(s)")
            if first_paper.authors:
                print(f"  First author: {first_paper.authors[0].last_name}, {first_paper.authors[0].first_name}")
            print(f"  Journal: {first_paper.journal}")
            print(f"  Year: {first_paper.publication_date[:4]}")
        else:
            print("⚠ No papers parsed from file")
            
    except Exception as e:
        print(f"❌ Error parsing XML: {e}")
        import traceback
        traceback.print_exc()
else:
    print("⚠ No XML files found to parse")
    print("  Run the collector tests first to generate XML files")

# Test 6: Test the COMPLETE parsing pipeline - THIS IS THE KEY TEST
if xml_files:
    print("\nTest 6: Testing COMPLETE parsing pipeline...")
    try:
        # This is the method that creates the JSON file
        all_papers = parser.parse_all_metadata()
        
        output_file = project_root / 'data' / 'processed' / 'all_papers.json'
        if output_file.exists():
            print(f"✓ Successfully created JSON output")
            print(f"  Total papers parsed: {len(all_papers)}")
            print(f"  Output file: {output_file}")
            
            # Check the JSON file is valid and show some stats
            with open(output_file, 'r') as f:
                json_data = json.load(f)
                print(f"  JSON file contains {len(json_data)} entries")
                
                # Show sample data
                if json_data:
                    sample_paper = json_data[0]
                    print(f"  Sample paper PMID: {sample_paper.get('pmid', 'N/A')}")
                    print(f"  Sample paper title: {sample_paper.get('title', 'N/A')[:50]}...")
        else:
            print("❌ JSON output file not created")
            print(f"  Expected location: {output_file}")
            
    except Exception as e:
        print(f"❌ Error in parsing pipeline: {e}")
        import traceback
        traceback.print_exc()

# Step 4: Integration test - Full pipeline
print("\n\n4. Integration Test - Full Pipeline")
print("-" * 40)
print("Testing the complete workflow: search → fetch → parse → save JSON")

# Simple test case
test_strain = "Saccharomyces boulardii"
test_condition = "diarrhea"

print(f"\nSearching for: {test_strain} + {test_condition}")

try:
    # Collect papers
    collection_result = collector.collect_by_strain_and_condition(
        test_strain, 
        test_condition, 
        max_results=2
    )
    
    if collection_result and collection_result['paper_count'] > 0:
        print(f"✓ Collection successful: {collection_result['paper_count']} papers")
        
        # Parse the individual file (as before)
        time.sleep(1)
        metadata_file = Path(collection_result['metadata_file'])
        
        if metadata_file.exists():
            papers = parser.parse_metadata_file(metadata_file)
            print(f"✓ Individual file parsing successful: {len(papers)} papers parsed")
            
            # NOW ALSO TEST THE COMPLETE PIPELINE
            print("\nTesting complete JSON generation...")
            all_papers = parser.parse_all_metadata()
            
            json_file = project_root / 'data' / 'processed' / 'all_papers.json'
            if json_file.exists() and len(all_papers) > 0:
                print("✅ COMPLETE INTEGRATION TEST PASSED!")
                print(f"   Individual parsing: ✓")
                print(f"   JSON generation: ✓")
                print(f"   Total papers in JSON: {len(all_papers)}")
            else:
                print("⚠ Individual parsing works but JSON generation failed")
        else:
            print("❌ Metadata file not found")
    else:
        print("⚠ No papers found for this search")
        
except Exception as e:
    print(f"❌ Integration test failed: {e}")
    import traceback
    traceback.print_exc()

# Step 5: Summary with detailed file check
print("\n" + "=" * 60)
print("DETAILED Test Summary")
print("=" * 60)

# Check what files were created
metadata_dir = project_root / "data" / "raw" / "metadata"
processed_dir = project_root / "data" / "processed"

xml_files = list(metadata_dir.glob("pubmed_batch_*.xml"))
json_file = processed_dir / "all_papers.json"
collection_results_file = metadata_dir / "collection_results.json"

print(f"\nFile Status:")
print(f"  XML files in metadata: {len(xml_files)}")
for xml_file in xml_files[-3:]:  # Show last 3 files
    size = xml_file.stat().st_size
    print(f"    - {xml_file.name}: {size:,} bytes")

print(f"  JSON output exists: {'✓ YES' if json_file.exists() else '❌ NO'}")
if json_file.exists():
    size = json_file.stat().st_size
    print(f"    - Size: {size:,} bytes")
    # Quick count of papers in JSON
    try:
        with open(json_file, 'r') as f:
            json_data = json.load(f)
            print(f"    - Papers in JSON: {len(json_data)}")
    except:
        print(f"    - Error reading JSON file")

print(f"  Collection results: {'✓ YES' if collection_results_file.exists() else '❌ NO'}")
if collection_results_file.exists():
    try:
        with open(collection_results_file, 'r') as f:
            collection_data = json.load(f)
            print(f"    - Collection runs recorded: {len(collection_data)}")
    except:
        print(f"    - Error reading collection results")

# Final status
if json_file.exists() and len(xml_files) > 0:
    print("\n🎉 SUCCESS: Complete pipeline is working!")
    print("   ✓ Data collection working")
    print("   ✓ XML parsing working") 
    print("   ✓ JSON generation working")
else:
    print("\n⚠ PARTIAL SUCCESS:")
    if len(xml_files) > 0:
        print("   ✓ Data collection working")
    if json_file.exists():
        print("   ✓ JSON generation working")
    else:
        print("   ❌ JSON generation needs debugging")

print("\nNext steps:")
print("1. Check the generated JSON file for data quality")
print("2. Test with larger datasets")
print("3. Add error handling for edge cases")
print("4. Consider adding data validation")