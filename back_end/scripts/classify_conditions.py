"""
Standalone script to classify health conditions in the database.

This is a separate classification step that runs AFTER intervention extraction.
It uses a focused prompt to classify each unique health_condition into one of 18 categories.
"""
import sqlite3
import json
import time
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from openai import OpenAI
except ImportError:
    print("Error: openai library not installed. Run: pip install openai")
    sys.exit(1)

# 18 condition categories
CONDITION_CATEGORIES = [
    "cardiac", "neurological", "digestive", "pulmonary", "endocrine", "renal",
    "oncological", "rheumatological", "psychiatric", "musculoskeletal",
    "dermatological", "infectious", "immunological", "hematological",
    "nutritional", "toxicological", "parasitic", "other"
]

def get_llm_client():
    """Get Ollama client."""
    return OpenAI(
        base_url='http://localhost:11434/v1',
        api_key='ollama'
    )

def create_classification_prompt(conditions):
    """Create focused prompt for condition classification."""
    conditions_list = "\n".join([f"{i+1}. {cond}" for i, cond in enumerate(conditions)])

    return f"""Classify each health condition into ONE category.

CATEGORIES:
cardiac, neurological, digestive, pulmonary, endocrine, renal, oncological,
rheumatological, psychiatric, musculoskeletal, dermatological, infectious,
immunological, hematological, nutritional, toxicological, parasitic, other

CONDITIONS:
{conditions_list}

Return ONLY JSON array:
[
    {{"number": 1, "category": "endocrine"}},
    {{"number": 2, "category": "cardiac"}},
    ...
]

Examples:
- hypertension → cardiac
- diabetes → endocrine
- depression → psychiatric
- HIV → infectious
- anemia → hematological
"""

def classify_batch(conditions, client):
    """Classify a batch of conditions."""
    prompt = create_classification_prompt(conditions)

    try:
        response = client.chat.completions.create(
            model="qwen2.5:14b",
            messages=[
                {"role": "system", "content": "You are a medical classifier. Return only valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=1000
        )

        text = response.choices[0].message.content.strip()

        # Clean response
        if "```" in text:
            lines = [l for l in text.split("\n") if not l.strip().startswith("```")]
            text = "\n".join(lines)

        classifications = json.loads(text)

        # Build result dict
        result = {}
        for item in classifications:
            idx = item["number"] - 1
            if 0 <= idx < len(conditions):
                cond = conditions[idx]
                cat = item["category"].lower().strip()
                if cat in CONDITION_CATEGORIES:
                    result[cond] = cat
                else:
                    result[cond] = "other"

        return result

    except Exception as e:
        print(f"  Error: {e}")
        return {cond: "other" for cond in conditions}


def main(batch_size=20, dry_run=False):
    """Main classification function."""
    print("=" * 70)
    print("CONDITION CLASSIFICATION")
    print("=" * 70)

    # Connect to database
    db_path = Path(__file__).parent.parent / "data" / "processed" / "intervention_research.db"
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Get unique conditions needing classification
    cursor.execute('''
        SELECT DISTINCT health_condition
        FROM interventions
        WHERE condition_category IS NULL
        ORDER BY health_condition
    ''')
    conditions = [row[0] for row in cursor.fetchall()]

    print(f"\nConditions to classify: {len(conditions)}\n")

    if len(conditions) == 0:
        print("No conditions need classification!")
        conn.close()
        return

    # Get LLM client
    client = get_llm_client()

    # Process in batches
    total_batches = (len(conditions) + batch_size - 1) // batch_size
    all_results = {}

    for batch_idx in range(total_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, len(conditions))
        batch = conditions[start:end]

        print(f"Batch {batch_idx + 1}/{total_batches} ({len(batch)} conditions)...")

        results = classify_batch(batch, client)
        all_results.update(results)

        # Show samples
        for i, (cond, cat) in enumerate(list(results.items())[:3]):
            print(f"  {cond[:50]}... -> {cat}")

        time.sleep(0.5)

    # Show distribution
    print("\n" + "=" * 70)
    print("CLASSIFICATION RESULTS")
    print("=" * 70)

    counts = {}
    for cat in all_results.values():
        counts[cat] = counts.get(cat, 0) + 1

    for cat, count in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {count}")

    if dry_run:
        print("\nDRY RUN - no database updates")
        conn.close()
        return

    # Update database
    print("\nUpdating database...")
    updated = 0

    for condition, category in all_results.items():
        cursor.execute('''
            UPDATE interventions
            SET condition_category = ?
            WHERE health_condition = ?
            AND condition_category IS NULL
        ''', (category, condition))
        updated += cursor.rowcount

    conn.commit()

    print(f"Updated {updated} records")

    # Final stats
    cursor.execute('SELECT COUNT(*) FROM interventions WHERE condition_category IS NULL')
    null_count = cursor.fetchone()[0]

    cursor.execute('SELECT COUNT(*) FROM interventions WHERE condition_category IS NOT NULL')
    with_cat = cursor.fetchone()[0]

    cursor.execute('SELECT COUNT(*) FROM interventions')
    total = cursor.fetchone()[0]

    print("\n" + "=" * 70)
    print("FINAL STATS")
    print("=" * 70)
    print(f"Total interventions: {total}")
    print(f"With condition_category: {with_cat}")
    print(f"Without condition_category: {null_count}")

    if null_count == 0:
        print("\n✓ SUCCESS! All interventions classified!")
    else:
        print(f"\n⚠ {null_count} interventions still unclassified")

    conn.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=20)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    main(batch_size=args.batch_size, dry_run=args.dry_run)
