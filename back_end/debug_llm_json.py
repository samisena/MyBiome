#!/usr/bin/env python3
"""
Debug LLM JSON parsing issue
"""

import sqlite3
import sys
import os
import json

# Add the src directory to the path so we can import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from entity_normalizer import EntityNormalizer


def debug_llm_json():
    """Debug the JSON parsing issue in find_by_llm"""

    # Connect to the database
    db_path = "data/processed/intervention_research.db"
    conn = sqlite3.connect(db_path)
    normalizer = EntityNormalizer(conn)

    print("=== DEBUGGING LLM JSON PARSING ===\n")

    # Test with a simple known case
    term = "acid reflux"
    entity_type = "condition"

    # Get candidate canonicals manually
    cursor = conn.cursor()
    cursor.execute("""
        SELECT canonical_name FROM canonical_entities
        WHERE entity_type = ?
        LIMIT 5
    """, (entity_type,))
    candidates = [row[0] for row in cursor.fetchall()]

    print(f"Testing term: '{term}' ({entity_type})")
    print(f"Candidates: {candidates}")

    # Build the prompt
    prompt = normalizer._build_llm_prompt(term, candidates, entity_type)
    print(f"\nPrompt:\n{prompt}")

    # Get LLM response
    if hasattr(normalizer, 'llm_client') and normalizer.llm_client:
        try:
            response = normalizer.llm_client.generate(prompt, temperature=0.1)
            llm_content = response['content'].strip()

            print(f"\nRaw LLM Response:\n'{llm_content}'")
            print(f"\nResponse type: {type(llm_content)}")
            print(f"Response length: {len(llm_content)}")

            # Try parsing exactly as in the method
            parsed = None
            try:
                parsed = json.loads(llm_content)
                print(f"\nDirect JSON parsing: SUCCESS")
                print(f"Parsed: {parsed}")
            except json.JSONDecodeError as e:
                print(f"\nDirect JSON parsing failed: {e}")

                # Try to extract JSON from response
                import re
                json_match = re.search(r'\{.*\}', llm_content, re.DOTALL)
                if json_match:
                    json_str = json_match.group()
                    print(f"\nExtracted JSON string: '{json_str}'")
                    try:
                        parsed = json.loads(json_str)
                        print(f"Extracted JSON parsing: SUCCESS")
                        print(f"Parsed: {parsed}")
                    except json.JSONDecodeError as e2:
                        print(f"Extracted JSON parsing also failed: {e2}")
                else:
                    print("\nNo JSON structure found in response")

            # Check what the method checks
            print(f"\nValidation checks:")
            print(f"  parsed is not None: {parsed is not None}")
            print(f"  isinstance(parsed, dict): {isinstance(parsed, dict) if parsed else 'N/A'}")

            if parsed and isinstance(parsed, dict):
                print(f"  parsed.get('match'): {parsed.get('match')}")
                print(f"  parsed.get('confidence'): {parsed.get('confidence')}")
                print(f"  parsed.get('reasoning'): {parsed.get('reasoning')}")

                match_name = parsed.get('match')
                confidence = float(parsed.get('confidence', 0.0))

                print(f"\nMatch validation:")
                print(f"  match_name: '{match_name}'")
                print(f"  confidence: {confidence}")
                print(f"  confidence > 0.3: {confidence > 0.3}")

                if match_name and confidence > 0.3:
                    print(f"  Would return match: YES")
                else:
                    print(f"  Would return match: NO")
            else:
                print("  Validation: FAILED - Invalid format")

        except Exception as e:
            print(f"Error getting LLM response: {e}")
    else:
        print("LLM client not available")

    conn.close()
    print("\n[SUCCESS] Debug completed")


if __name__ == "__main__":
    debug_llm_json()