#!/usr/bin/env python3
"""Test script to verify API connectivity and paper collection."""

import os
import sys
import requests
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_ncbi_api():
    """Test NCBI API connectivity."""
    print("Testing NCBI API connectivity...")

    api_key = os.getenv("NCBI_API_KEY")
    email = os.getenv("EMAIL", "samisena@outlook.com")

    print(f"API Key: {'***' + api_key[-4:] if api_key else 'Not set'}")
    print(f"Email: {email}")

    # Simple NCBI API test
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    params = {
        "db": "pubmed",
        "term": "diabetes",
        "retmax": 1,
        "retmode": "json",
        "email": email
    }

    if api_key:
        params["api_key"] = api_key

    try:
        print("Making API request...")
        response = requests.get(base_url, params=params, timeout=30)
        print(f"Response status: {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            print(f"Search successful: {data}")
            return True
        else:
            print(f"API error: {response.text}")
            return False

    except Exception as e:
        print(f"Request failed: {e}")
        return False

def test_paper_collector():
    """Test the paper collector directly."""
    print("\nTesting paper collector...")

    try:
        # Add the project root to Python path
        sys.path.insert(0, str(Path(__file__).parent))

        from back_end.src.orchestration.rotation_collection_integrator import RotationCollectionIntegrator
        from back_end.src.orchestration.rotation_session_manager import session_manager

        print("Creating collection integrator...")
        integrator = RotationCollectionIntegrator(session_manager)

        print("Testing simple collection...")
        result = integrator.paper_collector.collect_condition_papers(
            condition="diabetes",
            target_count=1,
            min_year=2023,
            max_year=None,
            use_s2_enrichment=False  # Disable enrichment for faster testing
        )

        print(f"Collection result: {result}")
        return result.get('success', False)

    except Exception as e:
        print(f"Paper collector test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=== API Connectivity Test ===")
    api_success = test_ncbi_api()

    print("\n=== Paper Collection Test ===")
    collector_success = test_paper_collector()

    print(f"\n=== Results ===")
    print(f"API Test: {'PASS' if api_success else 'FAIL'}")
    print(f"Collector Test: {'PASS' if collector_success else 'FAIL'}")

    if api_success and collector_success:
        print("\n✅ All tests passed! The pipeline should work now.")
    else:
        print("\n❌ Some tests failed. Check the output above for details.")