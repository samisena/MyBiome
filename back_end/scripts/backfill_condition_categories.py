"""
Backfill condition_category for all interventions in the database.

This script uses LLM to classify each health condition into one of 18 categories.
Processes in batches of 20 for efficiency.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import json
import time
from typing import List, Dict
from back_end.src.data_collection.database_manager import database_manager
from back_end.src.llm_processing.llm_client import get_llm_client
from back_end.src.conditions.taxonomy import ConditionType

def create_classification_prompt(conditions: List[str]) -> str:
    """Create prompt for batch condition classification."""

    conditions_list = "\n".join([f"{i+1}. {cond}" for i, cond in enumerate(conditions)])

    return f"""You are a medical condition classification expert. Classify each health condition below into ONE of these 18 categories:

CATEGORIES:
- cardiac: Cardiovascular (heart failure, hypertension, coronary artery disease, STEMI, MI)
- neurological: Brain & nervous system (Alzheimer's, Parkinson's, stroke, epilepsy, ADHD, dementia)
- digestive: GI system (GERD, IBD, IBS, cirrhosis, H. pylori infection, peptic ulcer)
- pulmonary: Respiratory (COPD, asthma, pneumonia, lung cancer, respiratory failure)
- endocrine: Hormones & metabolism (diabetes, thyroid disorders, obesity, PCOS, metabolic syndrome)
- renal: Kidney & urinary (chronic kidney disease, kidney stones, glomerulonephritis, renal failure)
- oncological: Cancer (breast cancer, colorectal cancer, leukemia, any malignancy or tumor)
- rheumatological: Autoimmune & rheumatic (rheumatoid arthritis, lupus, vasculitis, gout)
- psychiatric: Mental health (depression, anxiety, bipolar, schizophrenia, PTSD)
- musculoskeletal: Bones, muscles, joints (osteoarthritis, fractures, back pain, ligament injury)
- dermatological: Skin (acne, psoriasis, eczema, skin cancer, dermatitis)
- infectious: Infections (HIV, tuberculosis, COVID-19, sepsis, hepatitis, influenza)
- immunological: Allergies & immune disorders (food allergies, immunodeficiency, hypersensitivity)
- hematological: Blood disorders (anemia, clotting disorders, sickle cell disease, thrombocytopenia)
- nutritional: Nutrient deficiencies (vitamin D deficiency, malnutrition, iron deficiency)
- toxicological: Poisoning & toxicity (drug toxicity, overdose, heavy metal poisoning)
- parasitic: Parasitic infections (malaria, helminth infections, toxoplasmosis)
- other: Conditions that don't fit above categories

CONDITIONS TO CLASSIFY:
{conditions_list}

Return ONLY valid JSON array (no markdown, no explanations):
[
    {{"condition_number": 1, "category": "endocrine"}},
    {{"condition_number": 2, "category": "cardiac"}},
    ...
]

IMPORTANT:
- Return exactly ONE category per condition
- Use exact category names from the list above
- Be precise: hypertension=cardiac, diabetes=endocrine, depression=psychiatric
- If truly uncertain, use "other"
"""

def classify_conditions_batch(conditions: List[str], llm_client) -> Dict[str, str]:
    """
    Classify a batch of conditions using LLM.

    Returns:
        Dict mapping condition -> category
    """
    prompt = create_classification_prompt(conditions)

    try:
        response = llm_client.chat.completions.create(
            model="qwen2.5:14b",
            messages=[
                {"role": "system", "content": "You are a medical classification expert. Return only valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=2000
        )

        response_text = response.choices[0].message.content.strip()

        # Parse JSON
        # Remove markdown code blocks if present
        if response_text.startswith("```"):
            lines = response_text.split("\n")
            response_text = "\n".join([line for line in lines if not line.startswith("```")])

        classifications = json.loads(response_text)

        # Build result dict
        result = {}
        for item in classifications:
            condition_idx = item["condition_number"] - 1  # Convert to 0-indexed
            if 0 <= condition_idx < len(conditions):
                condition = conditions[condition_idx]
                category = item["category"]

                # Validate category
                try:
                    ConditionType(category)  # Will raise ValueError if invalid
                    result[condition] = category
                except ValueError:
                    print(f"  Warning: Invalid category '{category}' for '{condition}', using 'other'")
                    result[condition] = "other"

        return result

    except json.JSONDecodeError as e:
        print(f"  Error parsing JSON response: {e}")
        print(f"  Response: {response_text[:200]}...")
        # Return "other" for all conditions as fallback
        return {cond: "other" for cond in conditions}
    except Exception as e:
        print(f"  Error classifying batch: {e}")
        return {cond: "other" for cond in conditions}


def backfill_condition_categories(batch_size: int = 20, dry_run: bool = False):
    """
    Backfill condition_category for all interventions.

    Args:
        batch_size: Number of conditions to classify per LLM call
        dry_run: If True, don't actually update database
    """
    print("="  * 70)
    print("BACKFILLING CONDITION CATEGORIES")
    print("=" * 70)

    # Get LLM client
    llm_client = get_llm_client("qwen2.5:14b")

    # Get all unique conditions that need classification
    with database_manager.get_connection() as conn:
        cursor = conn.cursor()

        cursor.execute('''
            SELECT DISTINCT health_condition
            FROM interventions
            WHERE condition_category IS NULL
            ORDER BY health_condition
        ''')

        conditions = [row[0] for row in cursor.fetchall()]

    print(f"\nFound {len(conditions)} unique conditions to classify\n")

    if len(conditions) == 0:
        print("No conditions need classification!")
        return

    # Process in batches
    total_batches = (len(conditions) + batch_size - 1) // batch_size
    all_classifications = {}

    for batch_idx in range(total_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(conditions))
        batch = conditions[start_idx:end_idx]

        print(f"Batch {batch_idx + 1}/{total_batches}: Classifying {len(batch)} conditions...")

        # Classify batch
        batch_results = classify_conditions_batch(batch, llm_client)
        all_classifications.update(batch_results)

        # Show sample results
        for i, (cond, cat) in enumerate(list(batch_results.items())[:3]):
            print(f"  {cond[:50]}... → {cat}")

        if len(batch_results) > 3:
            print(f"  ... and {len(batch_results) - 3} more")

        print()

        # Small delay to avoid overwhelming the LLM
        time.sleep(0.5)

    print("=" * 70)
    print(f"CLASSIFICATION COMPLETE: {len(all_classifications)} conditions classified")
    print("=" * 70)

    # Show category distribution
    category_counts = {}
    for category in all_classifications.values():
        category_counts[category] = category_counts.get(category, 0) + 1

    print("\nCategory distribution:")
    for category, count in sorted(category_counts.items(), key=lambda x: -x[1]):
        print(f"  {category}: {count}")

    if dry_run:
        print("\nDRY RUN - No database updates performed")
        return

    # Update database
    print("\nUpdating database...")
    update_count = 0

    with database_manager.get_connection() as conn:
        cursor = conn.cursor()

        for condition, category in all_classifications.items():
            cursor.execute('''
                UPDATE interventions
                SET condition_category = ?
                WHERE health_condition = ?
                AND condition_category IS NULL
            ''', (category, condition))

            update_count += cursor.rowcount

        conn.commit()

    print(f"Updated {update_count} intervention records")

    # Verify
    with database_manager.get_connection() as conn:
        cursor = conn.cursor()

        cursor.execute('SELECT COUNT(*) FROM interventions WHERE condition_category IS NULL')
        remaining_null = cursor.fetchone()[0]

        cursor.execute('SELECT COUNT(*) FROM interventions WHERE condition_category IS NOT NULL')
        total_with_cat = cursor.fetchone()[0]

        cursor.execute('SELECT COUNT(*) FROM interventions')
        total = cursor.fetchone()[0]

    print("\n" + "=" * 70)
    print("FINAL STATUS:")
    print("=" * 70)
    print(f"Total interventions: {total}")
    print(f"With condition_category: {total_with_cat}")
    print(f"Without condition_category: {remaining_null}")

    if remaining_null == 0:
        print("\n✓✓✓ SUCCESS! All interventions now have condition_category! ✓✓✓")
    else:
        print(f"\n⚠ WARNING: {remaining_null} interventions still missing condition_category")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Backfill condition categories")
    parser.add_argument('--batch-size', type=int, default=20, help="Conditions per LLM call")
    parser.add_argument('--dry-run', action='store_true', help="Don't update database")

    args = parser.parse_args()

    backfill_condition_categories(batch_size=args.batch_size, dry_run=args.dry_run)
