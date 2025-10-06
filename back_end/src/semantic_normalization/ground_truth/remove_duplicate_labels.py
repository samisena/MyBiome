"""
Remove Duplicate Labels from Session File

Removes duplicate pair labels, keeping only the first occurrence.
Creates a backup before modification.
"""

import json
import shutil
from pathlib import Path
from datetime import datetime


def remove_duplicates_from_session(session_file: Path):
    """Remove duplicate pair labels from session file."""

    # Load session
    with open(session_file, 'r', encoding='utf-8') as f:
        session = json.load(f)

    labeled_pairs = session.get('labeled_pairs', [])
    print(f"Original labeled pairs: {len(labeled_pairs)}")

    # Find duplicates
    seen_pairs = set()
    unique_pairs = []
    duplicates = []

    for i, label in enumerate(labeled_pairs):
        key = tuple(sorted([label['intervention_1'], label['intervention_2']]))

        if key not in seen_pairs:
            seen_pairs.add(key)
            unique_pairs.append(label)
        else:
            duplicates.append({
                'index': i,
                'pair': key,
                'label': label
            })

    if not duplicates:
        print("\n[OK] No duplicates found!")
        return

    print(f"\nFound {len(duplicates)} duplicate labels:")
    for dup in duplicates:
        print(f"  [{dup['index']}] {dup['pair'][0]} vs {dup['pair'][1]}")
        rel_type = dup['label'].get('relationship', {}).get('type_code', 'Unknown')
        print(f"    Label: {rel_type}")

    # Ask for confirmation
    response = input(f"\nRemove {len(duplicates)} duplicate labels? (y/n): ")
    if response.lower() != 'y':
        print("Cancelled.")
        return

    # Create backup
    backup_file = session_file.with_suffix('.json.backup')
    shutil.copy(session_file, backup_file)
    print(f"\n[OK] Backup created: {backup_file}")

    # Update session
    session['labeled_pairs'] = unique_pairs
    session['progress']['labeled'] = len(unique_pairs)
    session['progress']['percentage'] = round((len(unique_pairs) / session['progress']['total']) * 100, 1)
    session['metadata']['last_updated'] = datetime.now().isoformat()

    # Save cleaned session
    with open(session_file, 'w', encoding='utf-8') as f:
        json.dump(session, f, indent=2, ensure_ascii=False)

    print(f"\n[OK] Removed {len(duplicates)} duplicate labels")
    print(f"[OK] Unique pairs remaining: {len(unique_pairs)}")
    print(f"[OK] Session saved: {session_file}")


def main():
    """Find and clean session files."""

    # Check both locations
    locations = [
        Path("c:/Users/samis/Desktop/MyBiome/back_end/experiments/semantic_normalization/data/ground_truth"),
        Path("c:/Users/samis/Desktop/MyBiome/back_end/src/semantic_normalization/ground_truth/data/ground_truth")
    ]

    session_files = []

    for loc in locations:
        if loc.exists():
            files = [f for f in loc.glob("labeling_session_*.json")
                    if 'candidates' not in f.name and 'BACKUP' not in f.name]
            session_files.extend(files)

    if not session_files:
        print("No session files found.")
        return

    print(f"Found {len(session_files)} session file(s):\n")
    for i, f in enumerate(session_files, 1):
        print(f"{i}. {f.name}")
        print(f"   {f.parent}")

    # Process most recent
    latest = max(session_files, key=lambda p: p.stat().st_mtime)
    print(f"\nProcessing most recent: {latest.name}\n")

    remove_duplicates_from_session(latest)


if __name__ == "__main__":
    main()
