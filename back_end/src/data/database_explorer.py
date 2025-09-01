import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add your project root to path
sys.path.append(str(Path(__file__).parent))

from src.data.database_manager import DatabaseManager

def explore_database():
    """Explore and display all data in the database"""
    
    # Initialize database manager
    db_manager = DatabaseManager()
    
    print("=" * 80)
    print("DATABASE EXPLORATION REPORT")
    print("=" * 80)
    
    # 1. Get overall statistics
    print("\n📊 DATABASE STATISTICS:")
    print("-" * 40)
    stats = db_manager.get_database_stats()
    
    print(f"Total Papers: {stats['total_papers']}")
    print(f"Total Correlations: {stats['total_correlations']}")
    print(f"Unique Probiotic Strains: {stats['unique_strains']}")
    print(f"Unique Health Conditions: {stats['unique_conditions']}")
    print(f"Date Range: {stats['date_range']}")
    print(f"Papers Added Last Week: {stats['papers_added_last_week']}")
    
    # 2. Top journals
    if stats['top_journals']:
        print(f"\n📚 TOP JOURNALS:")
        print("-" * 40)
        for journal, count in stats['top_journals'][:5]:
            print(f"  {journal}: {count} papers")
    
    # 3. Correlation type distribution
    if stats['correlation_types']:
        print(f"\n🔍 CORRELATION TYPES:")
        print("-" * 40)
        for corr_type, count in stats['correlation_types'].items():
            print(f"  {corr_type}: {count}")
    
    # 4. Top strain-condition pairs
    if stats['top_strain_condition_pairs']:
        print(f"\n🦠 TOP STRAIN-CONDITION PAIRS:")
        print("-" * 40)
        for pair in stats['top_strain_condition_pairs'][:5]:
            print(f"  {pair['strain']} → {pair['condition']}: {pair['papers']} papers")
    
    # 5. Show sample papers
    print(f"\n📄 SAMPLE PAPERS (First 5):")
    print("-" * 40)
    papers = db_manager.get_all_papers(limit=5)
    for i, paper in enumerate(papers, 1):
        print(f"\n{i}. PMID: {paper['pmid']}")
        print(f"   Title: {paper['title'][:100]}...")
        print(f"   Journal: {paper.get('journal', 'Unknown')}")
        print(f"   Date: {paper.get('publication_date', 'Unknown')}")
        
        # Get correlations for this paper
        correlations = db_manager.get_correlations_by_paper(paper['pmid'])
        if correlations:
            print(f"   Correlations found: {len(correlations)}")
            for corr in correlations[:2]:  # Show first 2 correlations
                print(f"     • {corr['probiotic_strain']} → {corr['health_condition']} ({corr['correlation_type']})")
    
    # 6. Aggregated correlations (evidence across multiple papers)
    print(f"\n🔬 STRONGEST EVIDENCE (Multiple Papers):")
    print("-" * 40)
    aggregated = db_manager.aggregate_correlations(min_papers=1)  # Set to 1 since you may have limited data
    for agg in aggregated[:5]:
        print(f"\n  {agg['probiotic_strain']} → {agg['health_condition']}")
        print(f"    Papers: {agg['paper_count']}")
        print(f"    Avg Strength: {agg['avg_strength']:.2f}" if agg['avg_strength'] else "    Avg Strength: N/A")
        print(f"    Types: {agg['correlation_types']}")
        print(f"    Results: {agg['positive_count']} positive, {agg['neutral_count']} neutral, {agg['negative_count']} negative")
    
    # 7. Verification status
    print(f"\n✅ VERIFICATION STATUS:")
    print("-" * 40)
    with db_manager.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT validation_status, COUNT(*) as count
            FROM correlations
            GROUP BY validation_status
        ''')
        for row in cursor.fetchall():
            print(f"  {row[0]}: {row[1]} correlations")
    
    # 8. Show specific examples of correlations
    print(f"\n💊 EXAMPLE CORRELATIONS (First 10):")
    print("-" * 40)
    with db_manager.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT c.*, p.title 
            FROM correlations c
            JOIN papers p ON c.paper_id = p.pmid
            LIMIT 10
        ''')
        for i, row in enumerate(cursor.fetchall(), 1):
            print(f"\n{i}. {row['probiotic_strain']} → {row['health_condition']}")
            print(f"   Paper: {row['title'][:80]}...")
            print(f"   Type: {row['correlation_type']}")
            print(f"   Strength: {row['correlation_strength']}" if row['correlation_strength'] else "   Strength: N/A")
            print(f"   Confidence: {row['confidence_score']}" if row['confidence_score'] else "   Confidence: N/A")
            if row['supporting_quote']:
                print(f"   Quote: \"{row['supporting_quote'][:100]}...\"")
    
    # 9. Check for unprocessed papers
    print(f"\n📋 UNPROCESSED PAPERS:")
    print("-" * 40)
    unprocessed = db_manager.get_unprocessed_papers(
        extraction_model="llama-3.3-70b-versatile",
        limit=5
    )
    print(f"Found {len(unprocessed)} unprocessed papers")
    if unprocessed:
        for paper in unprocessed[:3]:
            print(f"  • {paper['pmid']}: {paper['title'][:60]}...")
    
    print("\n" + "=" * 80)
    print("END OF REPORT")
    print("=" * 80)

if __name__ == "__main__":
    explore_database()