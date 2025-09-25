#!/usr/bin/env python3
"""
Simple test of LLM entity matching to debug issues
"""

import sqlite3
import sys
import os
import json

# Add the src directory to the path so we can import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from entity_normalizer import EntityNormalizer


def test_llm_simple():
    """Simple test of LLM functionality"""

    # Connect to the database
    db_path = "data/processed/intervention_research.db"
    conn = sqlite3.connect(db_path)
    normalizer = EntityNormalizer(conn)

    print("=== SIMPLE LLM TEST ===\n")

    # Test the LLM prompt building
    candidate_canonicals = ["Gastroesophageal Reflux Disease", "irritable bowel syndrome", "migraine"]
    prompt = normalizer._build_llm_prompt("acid reflux", candidate_canonicals, "condition")

    print("--- LLM Prompt ---")
    print(prompt)
    print("\n" + "="*50 + "\n")

    # Test direct LLM call
    print("--- Direct LLM Call ---")
    try:
        if hasattr(normalizer, 'llm_client') and normalizer.llm_client:
            response = normalizer.llm_client.generate(prompt, temperature=0.1)
            print("Raw LLM Response:")
            print(response['content'])
            print("\n" + "="*50 + "\n")

            # Try parsing the response
            llm_content = response['content'].strip()
            print("--- JSON Parsing Test ---")
            print(f"Content to parse: '{llm_content}'")

            try:
                parsed = json.loads(llm_content)
                print("Successfully parsed JSON:")
                print(json.dumps(parsed, indent=2))
            except json.JSONDecodeError as e:
                print(f"JSON parsing failed: {e}")

                # Try to extract JSON from response
                import re
                json_match = re.search(r'\{.*\}', llm_content, re.DOTALL)
                if json_match:
                    json_str = json_match.group()
                    print(f"Extracted JSON: '{json_str}'")
                    try:
                        parsed = json.loads(json_str)
                        print("Successfully parsed extracted JSON:")
                        print(json.dumps(parsed, indent=2))
                    except json.JSONDecodeError as e2:
                        print(f"Extracted JSON parsing also failed: {e2}")
                else:
                    print("No JSON structure found in response")

        else:
            print("LLM client not available")

    except Exception as e:
        print(f"Error in direct LLM call: {e}")

    # Test the entity normalizer's find_by_llm method
    print("\n--- Entity Normalizer LLM Test ---")
    try:
        # Use existing entities for testing
        result = normalizer.find_by_llm("probiotics", "intervention")
        if result:
            print("LLM matching found result:")
            print(f"  Match: {result['canonical_name']}")
            print(f"  Confidence: {result['confidence']}")
            print(f"  Reasoning: {result.get('reasoning', 'N/A')}")
        else:
            print("LLM matching returned no result")

    except Exception as e:
        print(f"Error in entity normalizer LLM test: {e}")
        import traceback
        traceback.print_exc()

    conn.close()
    print("\n[SUCCESS] Simple LLM test completed")


if __name__ == "__main__":
    test_llm_simple()