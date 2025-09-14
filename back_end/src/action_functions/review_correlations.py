#!/usr/bin/env python3
"""
Interactive correlation review program for manual validation and correction.
Allows users to review correlations extracted from abstracts only.
"""

import sys
import sqlite3
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime

# Add the src directory to Python path
src_dir = Path(__file__).parent.parent
sys.path.insert(0, str(src_dir))

try:
    from src.paper_collection.database_manager import database_manager
    from src.data.config import config
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running this from the back_end directory")
    sys.exit(1)

@dataclass
class CorrelationReview:
    """Data class for correlation review session."""
    correlation_id: int
    paper_id: str
    title: str
    abstract: str
    probiotic_strain: str
    health_condition: str
    correlation_type: str
    correlation_strength: float
    confidence_score: float
    supporting_quote: str
    extraction_model: str

class CorrelationReviewer:
    """Interactive correlation reviewer for manual validation."""
    
    def __init__(self):
        self.db_path = config.db_path
        self.current_index = 0
        self.correlations = []
        self.review_session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.reviews_dir = Path(config.db_path).parent / "reviews"
        self.reviews_dir.mkdir(exist_ok=True)
        self.session_file = self.reviews_dir / f"review_session_{self.review_session_id}.jsonl"
        self.ensure_review_columns()
    
    def ensure_review_columns(self):
        """Ensure the required review columns exist in the correlations table."""
        with database_manager.get_connection() as conn:
            cursor = conn.cursor()
            
            # Check if review columns exist and add them if they don't
            try:
                cursor.execute("SELECT human_reviewer FROM correlations LIMIT 1")
            except sqlite3.OperationalError:
                cursor.execute("ALTER TABLE correlations ADD COLUMN human_reviewer TEXT")
            
            try:
                cursor.execute("SELECT review_timestamp FROM correlations LIMIT 1")
            except sqlite3.OperationalError:
                cursor.execute("ALTER TABLE correlations ADD COLUMN review_timestamp TIMESTAMP")
            
            try:
                cursor.execute("SELECT review_notes FROM correlations LIMIT 1")
            except sqlite3.OperationalError:
                cursor.execute("ALTER TABLE correlations ADD COLUMN review_notes TEXT")
            
            conn.commit()
        
    def load_abstract_only_correlations(self) -> List[CorrelationReview]:
        """Load correlations from papers with abstracts only (no fulltext)."""
        with database_manager.get_connection() as conn:
            cursor = conn.cursor()
            
            # Get correlations from abstract-only papers
            cursor.execute('''
                SELECT 
                    c.id,
                    c.paper_id,
                    p.title,
                    p.abstract,
                    c.probiotic_strain,
                    c.health_condition,
                    c.correlation_type,
                    c.correlation_strength,
                    c.confidence_score,
                    c.supporting_quote,
                    c.extraction_model
                FROM correlations c
                JOIN papers p ON c.paper_id = p.pmid
                WHERE (p.has_fulltext = 0 OR p.has_fulltext = 'false' OR p.has_fulltext IS NULL)
                  AND c.human_reviewed = 0
                ORDER BY c.confidence_score DESC, c.id
            ''')
            
            correlations = []
            for row in cursor.fetchall():
                correlations.append(CorrelationReview(
                    correlation_id=row[0],
                    paper_id=row[1],
                    title=row[2],
                    abstract=row[3],
                    probiotic_strain=row[4],
                    health_condition=row[5],
                    correlation_type=row[6],
                    correlation_strength=row[7] or 0.0,
                    confidence_score=row[8] or 0.0,
                    supporting_quote=row[9] or "",
                    extraction_model=row[10]
                ))
            
            return correlations
    
    def display_correlation(self, correlation: CorrelationReview) -> None:
        """Display correlation details for review."""
        print("=" * 80)
        print(f"CORRELATION {self.current_index + 1} of {len(self.correlations)}")
        print("=" * 80)
        print(f"Paper ID: {correlation.paper_id}")
        print(f"Title: {correlation.title}")
        print()
        print("ABSTRACT:")
        print("-" * 40)
        # Display the full abstract
        print(correlation.abstract)
        
        # Check if abstract appears truncated
        abstract_text = correlation.abstract.strip()
        potential_truncation_indicators = [
            "particularly with",
            "showing",
            "demonstrated",
            "compared to",
            "resulted in",
            "significant",
            "indicate that",
            "suggest that",
            "found that"
        ]
        
        is_likely_truncated = (
            not abstract_text.endswith('.') or 
            any(abstract_text.endswith(indicator) for indicator in potential_truncation_indicators) or
            len(abstract_text) < 200
        )
        
        if is_likely_truncated:
            print()
            print("⚠️  WARNING: This abstract appears to be truncated!")
            print(f"   PMID: {correlation.paper_id} - You may want to check PubMed for the full abstract")
        
        print()
        
        print("EXTRACTED CORRELATION:")
        print("-" * 40)
        print(f"Probiotic Strain: {correlation.probiotic_strain}")
        print(f"Health Condition: {correlation.health_condition}")
        print(f"Correlation Type: {correlation.correlation_type}")
        print(f"Strength: {correlation.correlation_strength:.2f}")
        print(f"Confidence: {correlation.confidence_score:.2f}")
        print(f"Model: {correlation.extraction_model}")
        if correlation.supporting_quote:
            print(f"Supporting Quote: \"{correlation.supporting_quote}\"")
        print()
    
    def get_user_input(self) -> Tuple[str, Optional[Dict]]:
        """Get user input for correlation review."""
        print("REVIEW OPTIONS:")
        print("  [c] Correct - Mark as correct")
        print("  [w] Wrong - Mark as wrong")  
        print("  [e] Edit - Correct the strain or condition")
        print("  [t] Truncated - Mark as truncated abstract (skip review)")
        print("  [s] Skip - Skip this correlation")
        print("  [n] Next - Move to next correlation")
        print("  [p] Previous - Go back to previous correlation")
        print("  [q] Quit - Exit the review session")
        print()
        
        while True:
            choice = input("Your choice: ").lower().strip()
            
            if choice in ['c', 'correct']:
                return 'correct', None
            elif choice in ['w', 'wrong']:
                reason = input("Reason for marking wrong (optional): ").strip()
                return 'wrong', {'reason': reason} if reason else None
            elif choice in ['e', 'edit']:
                return self.get_edit_input()
            elif choice in ['t', 'truncated']:
                return 'truncated', {'reason': 'Abstract appears truncated - cannot review accurately'}
            elif choice in ['s', 'skip']:
                return 'skip', None
            elif choice in ['n', 'next']:
                return 'next', None
            elif choice in ['p', 'prev', 'previous']:
                return 'previous', None
            elif choice in ['q', 'quit']:
                return 'quit', None
            else:
                print("Invalid choice. Please try again.")
    
    def get_edit_input(self) -> Tuple[str, Dict]:
        """Get correction input from user."""
        print("\nEDIT CORRELATION:")
        print("Leave blank to keep current value")
        
        new_strain = input(f"Probiotic Strain [{self.correlations[self.current_index].probiotic_strain}]: ").strip()
        new_condition = input(f"Health Condition [{self.correlations[self.current_index].health_condition}]: ").strip()
        
        # Get correlation type if needed
        current_type = self.correlations[self.current_index].correlation_type
        print(f"Correlation Type options: positive, negative, neutral, inconclusive")
        new_type = input(f"Correlation Type [{current_type}]: ").strip()
        
        corrections = {}
        if new_strain:
            corrections['probiotic_strain'] = new_strain
        if new_condition:
            corrections['health_condition'] = new_condition
        if new_type and new_type in ['positive', 'negative', 'neutral', 'inconclusive']:
            corrections['correlation_type'] = new_type
        
        if corrections:
            return 'edit', corrections
        else:
            print("No changes made.")
            return 'skip', None
    
    def save_review(self, correlation_id: int, action: str, corrections: Optional[Dict] = None) -> None:
        """Save review result to database and JSON file."""
        correlation = self.correlations[self.current_index]
        
        # Save to database
        with database_manager.get_connection() as conn:
            cursor = conn.cursor()
            
            # Update the correlation based on action
            if action == 'correct':
                cursor.execute('''
                    UPDATE correlations 
                    SET human_reviewed = 1,
                        human_reviewer = 'manual_review',
                        review_timestamp = CURRENT_TIMESTAMP,
                        review_notes = 'Marked as correct'
                    WHERE id = ?
                ''', (correlation_id,))
                
            elif action == 'wrong':
                reason = corrections.get('reason', 'Marked as wrong') if corrections else 'Marked as wrong'
                cursor.execute('''
                    UPDATE correlations 
                    SET human_reviewed = 1,
                        human_reviewer = 'manual_review',
                        review_timestamp = CURRENT_TIMESTAMP,
                        review_notes = ?
                    WHERE id = ?
                ''', (reason, correlation_id))
                
            elif action == 'truncated':
                reason = corrections.get('reason', 'Abstract truncated - cannot review') if corrections else 'Abstract truncated - cannot review'
                cursor.execute('''
                    UPDATE correlations 
                    SET human_reviewed = 1,
                        human_reviewer = 'manual_review',
                        review_timestamp = CURRENT_TIMESTAMP,
                        review_notes = ?
                    WHERE id = ?
                ''', (reason, correlation_id))
                
            elif action == 'edit' and corrections:
                # Build update query dynamically based on corrections
                update_fields = []
                update_values = []
                
                for field, value in corrections.items():
                    update_fields.append(f"{field} = ?")
                    update_values.append(value)
                
                # Add review fields
                update_fields.extend([
                    "human_reviewed = 1",
                    "human_reviewer = 'manual_review'",
                    "review_timestamp = CURRENT_TIMESTAMP",
                    "review_notes = ?"
                ])
                
                correction_notes = f"Corrected: {', '.join(f'{k}: {v}' for k, v in corrections.items())}"
                update_values.append(correction_notes)
                update_values.append(correlation_id)
                
                query = f"UPDATE correlations SET {', '.join(update_fields)} WHERE id = ?"
                cursor.execute(query, update_values)
            
            conn.commit()
        
        # Save to JSON file
        self.save_review_to_file(correlation, action, corrections)
    
    def save_review_to_file(self, correlation: CorrelationReview, action: str, corrections: Optional[Dict] = None) -> None:
        """Save individual review to JSONL file for training data."""
        
        # Create the review record
        review_record = {
            "review_id": f"{self.review_session_id}_{correlation.correlation_id}",
            "timestamp": datetime.now().isoformat(),
            "session_id": self.review_session_id,
            
            # Paper information
            "paper": {
                "pmid": correlation.paper_id,
                "title": correlation.title,
                "abstract": correlation.abstract
            },
            
            # Original LLM extraction
            "llm_extraction": {
                "probiotic_strain": correlation.probiotic_strain,
                "health_condition": correlation.health_condition,
                "correlation_type": correlation.correlation_type,
                "correlation_strength": correlation.correlation_strength,
                "confidence_score": correlation.confidence_score,
                "supporting_quote": correlation.supporting_quote,
                "extraction_model": correlation.extraction_model
            },
            
            # Human review
            "human_review": {
                "action": action,  # 'correct', 'wrong', 'edit'
                "reviewer": "manual_review",
                "corrections": corrections or {},
                "reason": corrections.get('reason') if corrections else None
            },
            
            # Final corrected values (for training)
            "corrected_extraction": {
                "probiotic_strain": corrections.get('probiotic_strain', correlation.probiotic_strain) if corrections else correlation.probiotic_strain,
                "health_condition": corrections.get('health_condition', correlation.health_condition) if corrections else correlation.health_condition,
                "correlation_type": corrections.get('correlation_type', correlation.correlation_type) if corrections else correlation.correlation_type,
                "is_correct": action == 'correct' or action == 'edit',
                "is_reviewable": action != 'truncated'
            }
        }
        
        # Append to JSONL file
        try:
            with open(self.session_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(review_record, ensure_ascii=False) + '\n')
        except Exception as e:
            print(f"Warning: Could not save review to file: {e}")
    
    def show_progress(self) -> None:
        """Show current progress."""
        total = len(self.correlations)
        current = self.current_index + 1
        percentage = (current / total) * 100 if total > 0 else 0
        
        print(f"\nProgress: {current}/{total} ({percentage:.1f}%)")
        
        # Show review statistics
        with database_manager.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT 
                    SUM(CASE WHEN human_reviewed = 1 AND review_notes LIKE '%correct%' THEN 1 ELSE 0 END) as correct,
                    SUM(CASE WHEN human_reviewed = 1 AND review_notes LIKE '%wrong%' THEN 1 ELSE 0 END) as wrong,
                    SUM(CASE WHEN human_reviewed = 1 AND review_notes LIKE '%Corrected%' THEN 1 ELSE 0 END) as edited,
                    COUNT(*) as total_reviewed
                FROM correlations c
                JOIN papers p ON c.paper_id = p.pmid
                WHERE (p.has_fulltext = 0 OR p.has_fulltext = 'false' OR p.has_fulltext IS NULL)
                  AND c.human_reviewed = 1
            ''')
            
            stats = cursor.fetchone()
            if stats:
                print(f"Reviewed: {stats[3]} | Correct: {stats[0]} | Wrong: {stats[1]} | Edited: {stats[2]}")
        print()
    
    def run_review_session(self) -> None:
        """Run the interactive review session."""
        print("MyBiome Correlation Review Tool")
        print("===============================")
        print("Reviewing correlations extracted from abstracts only")
        print("This tool helps create training examples for fine-tuning")
        print()
        print(f"Reviews will be saved to:")
        print(f"  Database: {self.db_path}")
        print(f"  JSON File: {self.session_file}")
        print()
        
        # Load correlations
        self.correlations = self.load_abstract_only_correlations()
        
        if not self.correlations:
            print("No correlations found that need review!")
            return
        
        print(f"Loaded {len(self.correlations)} correlations for review")
        print()
        
        # Review loop
        while self.current_index < len(self.correlations):
            correlation = self.correlations[self.current_index]
            
            # Clear screen (works on both Windows and Unix)
            import os
            os.system('cls' if os.name == 'nt' else 'clear')
            
            self.show_progress()
            self.display_correlation(correlation)
            
            action, data = self.get_user_input()
            
            if action == 'quit':
                print("Exiting review session...")
                break
            elif action == 'next':
                if self.current_index < len(self.correlations) - 1:
                    self.current_index += 1
                else:
                    print("Already at the last correlation.")
                    input("Press Enter to continue...")
            elif action == 'previous':
                if self.current_index > 0:
                    self.current_index -= 1
                else:
                    print("Already at the first correlation.")
                    input("Press Enter to continue...")
            elif action == 'skip':
                self.current_index += 1
            elif action in ['correct', 'wrong', 'edit', 'truncated']:
                try:
                    self.save_review(correlation.correlation_id, action, data)
                    print(f"Review saved! Correlation marked as {action}.")
                    input("Press Enter to continue...")
                    self.current_index += 1
                except Exception as e:
                    print(f"Error saving review: {e}")
                    input("Press Enter to continue...")
        
        print("\nReview session completed!")
        self.show_final_statistics()
    
    def show_final_statistics(self) -> None:
        """Show final review statistics."""
        print("\n" + "=" * 50)
        print("FINAL REVIEW STATISTICS")
        print("=" * 50)
        
        with database_manager.get_connection() as conn:
            cursor = conn.cursor()
            
            # Get comprehensive stats
            cursor.execute('''
                SELECT 
                    COUNT(*) as total_abstract_correlations,
                    SUM(CASE WHEN human_reviewed = 1 THEN 1 ELSE 0 END) as reviewed,
                    SUM(CASE WHEN human_reviewed = 1 AND review_notes LIKE '%correct%' THEN 1 ELSE 0 END) as correct,
                    SUM(CASE WHEN human_reviewed = 1 AND review_notes LIKE '%wrong%' THEN 1 ELSE 0 END) as wrong,
                    SUM(CASE WHEN human_reviewed = 1 AND review_notes LIKE '%Corrected%' THEN 1 ELSE 0 END) as edited
                FROM correlations c
                JOIN papers p ON c.paper_id = p.pmid
                WHERE (p.has_fulltext = 0 OR p.has_fulltext = 'false' OR p.has_fulltext IS NULL)
            ''')
            
            stats = cursor.fetchone()
            
            print(f"Total Abstract-based Correlations: {stats[0]}")
            print(f"Reviewed: {stats[1]} ({(stats[1]/stats[0]*100):.1f}%)")
            print(f"  - Correct: {stats[2]}")
            print(f"  - Wrong: {stats[3]}")  
            print(f"  - Edited: {stats[4]}")
            print(f"Remaining: {stats[0] - stats[1]}")
        
        print("\nThis data can be used for:")
        print("- Creating training examples for fine-tuning")
        print("- Evaluating extraction model performance")
        print("- Identifying common extraction errors")
        print()
        print("Review files saved:")
        print(f"- Database: {self.db_path}")
        print(f"- JSON Training Data: {self.session_file}")
        if self.session_file.exists():
            print(f"  ({self.session_file.stat().st_size} bytes)")
        print()

def main():
    """Main function to run the correlation reviewer."""
    reviewer = CorrelationReviewer()
    reviewer.run_review_session()

if __name__ == "__main__":
    main()