#!/usr/bin/env python3
"""
Utility to analyze review files and export training data.
"""

import sys
import json
from pathlib import Path
from typing import Dict, List
from collections import Counter

# Add the src directory to Python path
src_dir = Path(__file__).parent.parent
sys.path.insert(0, str(src_dir))

try:
    from src.data.config import config
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running this from the back_end directory")
    sys.exit(1)

def load_review_files() -> List[Dict]:
    """Load all review files from the reviews directory."""
    reviews_dir = Path(config.database.path).parent / "reviews"
    if not reviews_dir.exists():
        print("No reviews directory found.")
        return []
    
    all_reviews = []
    review_files = list(reviews_dir.glob("review_session_*.jsonl"))
    
    if not review_files:
        print("No review files found.")
        return []
    
    print(f"Found {len(review_files)} review session files:")
    
    for file_path in review_files:
        print(f"  - {file_path.name}")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                session_reviews = []
                for line in f:
                    if line.strip():
                        review = json.loads(line)
                        session_reviews.append(review)
                all_reviews.extend(session_reviews)
                print(f"    Loaded {len(session_reviews)} reviews")
        except Exception as e:
            print(f"    Error loading file: {e}")
    
    return all_reviews

def analyze_reviews(reviews: List[Dict]) -> None:
    """Analyze the loaded reviews."""
    if not reviews:
        print("No reviews to analyze.")
        return
    
    print(f"\n=== REVIEW ANALYSIS ===")
    print(f"Total reviews: {len(reviews)}")
    
    # Action breakdown
    actions = [r['human_review']['action'] for r in reviews]
    action_counts = Counter(actions)
    
    print(f"\nAction breakdown:")
    for action, count in action_counts.items():
        percentage = (count / len(reviews)) * 100
        print(f"  {action}: {count} ({percentage:.1f}%)")
    
    # Accuracy metrics
    correct_reviews = [r for r in reviews if r['corrected_extraction']['is_correct']]
    accuracy = (len(correct_reviews) / len(reviews)) * 100 if reviews else 0
    print(f"\nOverall accuracy: {accuracy:.1f}%")
    
    # Most common corrections
    corrections = []
    for r in reviews:
        if r['human_review']['action'] == 'edit':
            for field, value in r['human_review']['corrections'].items():
                if field in ['probiotic_strain', 'health_condition']:
                    original = r['llm_extraction'][field]
                    corrections.append(f"{field}: '{original}' → '{value}'")
    
    if corrections:
        print(f"\nMost common corrections:")
        correction_counts = Counter(corrections)
        for correction, count in correction_counts.most_common(5):
            print(f"  {correction} ({count}x)")
    
    # Model performance
    models = [r['llm_extraction']['extraction_model'] for r in reviews]
    model_accuracy = {}
    
    for model in set(models):
        model_reviews = [r for r in reviews if r['llm_extraction']['extraction_model'] == model]
        model_correct = [r for r in model_reviews if r['corrected_extraction']['is_correct']]
        accuracy = (len(model_correct) / len(model_reviews)) * 100 if model_reviews else 0
        model_accuracy[model] = {
            'total': len(model_reviews),
            'accuracy': accuracy
        }
    
    print(f"\nModel performance:")
    for model, stats in model_accuracy.items():
        print(f"  {model}: {stats['accuracy']:.1f}% ({stats['total']} reviews)")

def export_training_data(reviews: List[Dict], output_file: str = None) -> None:
    """Export reviews as training data in different formats."""
    if not reviews:
        print("No reviews to export.")
        return
    
    reviews_dir = Path(config.database.path).parent / "reviews"
    
    if output_file is None:
        output_file = reviews_dir / "training_data.jsonl"
    else:
        output_file = Path(output_file)
    
    # Format for fine-tuning (input/output pairs)
    training_examples = []
    
    for review in reviews:
        # Create training example
        training_example = {
            "messages": [
                {
                    "role": "system",
                    "content": "Extract probiotic strain and health condition correlations from research abstracts. Return in JSON format."
                },
                {
                    "role": "user", 
                    "content": f"Title: {review['paper']['title']}\n\nAbstract: {review['paper']['abstract']}\n\nExtract probiotic-health correlations:"
                },
                {
                    "role": "assistant",
                    "content": json.dumps({
                        "probiotic_strain": review['corrected_extraction']['probiotic_strain'],
                        "health_condition": review['corrected_extraction']['health_condition'],
                        "correlation_type": review['corrected_extraction']['correlation_type']
                    })
                }
            ],
            "metadata": {
                "paper_id": review['paper']['pmid'],
                "review_action": review['human_review']['action'],
                "was_corrected": review['human_review']['action'] == 'edit',
                "original_extraction": review['llm_extraction']
            }
        }
        training_examples.append(training_example)
    
    # Save training data
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            for example in training_examples:
                f.write(json.dumps(example, ensure_ascii=False) + '\n')
        
        print(f"\nTraining data exported to: {output_file}")
        print(f"Total training examples: {len(training_examples)}")
        
    except Exception as e:
        print(f"Error exporting training data: {e}")

def main():
    """Main function."""
    print("MyBiome Review Analysis Tool")
    print("============================")
    
    # Load all review files
    reviews = load_review_files()
    
    if not reviews:
        return
    
    # Analyze reviews
    analyze_reviews(reviews)
    
    # Ask if user wants to export training data
    export_choice = input("\nExport training data? (y/n): ").lower().strip()
    if export_choice in ['y', 'yes']:
        export_training_data(reviews)

if __name__ == "__main__":
    main()