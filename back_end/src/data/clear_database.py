import sqlite3
import os
from pathlib import Path
from contextlib import contextmanager

@contextmanager
def get_db_connection(db_path):
    """Context manager for database connections"""
    conn = sqlite3.connect(str(db_path))
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

def clear_database_directly():
    """Direct database clearing using context managers properly"""
    
    # Build the database path directly
    project_root = Path(__file__).parent.parent.parent
    db_path = project_root / "data" / "processed" / "pubmed_research.db"
    
    print(f"Database path: {db_path}")
    
    if not db_path.exists():
        print("❌ Database file doesn't exist!")
        return
    
    print(f"Database size: {os.path.getsize(db_path) / 1024:.2f} KB")
    
    try:
        # Use context manager for database connection
        with get_db_connection(db_path) as conn:
            cursor = conn.cursor()
            
            # Check current counts
            cursor.execute('SELECT COUNT(*) FROM papers')
            papers_count = cursor.fetchone()[0]
            cursor.execute('SELECT COUNT(*) FROM correlations')
            correlations_count = cursor.fetchone()[0]
            
            print(f"\nCurrent database contents:")
            print(f"  Papers: {papers_count}")
            print(f"  Correlations: {correlations_count}")
            
            if papers_count == 0 and correlations_count == 0:
                print("\n✨ Database is already empty!")
                return
        
        # Ask for confirmation (outside the connection context)
        response = input("\n⚠️  Delete all data? (yes/no): ").strip().lower()
        if response != 'yes':
            print("Cancelled.")
            return
        
        # Clear the data in a new connection context
        with get_db_connection(db_path) as conn:
            cursor = conn.cursor()
            
            print("\nClearing database...")
            cursor.execute('DELETE FROM correlations')
            corr_deleted = cursor.rowcount
            cursor.execute('DELETE FROM papers')
            papers_deleted = cursor.rowcount
            
            # Reset autoincrement
            cursor.execute("DELETE FROM sqlite_sequence WHERE name='correlations'")
            
            print(f"✓ Deleted {papers_deleted} papers")
            print(f"✓ Deleted {corr_deleted} correlations")
        
        # Verify in another connection context
        with get_db_connection(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT COUNT(*) FROM papers')
            papers_after = cursor.fetchone()[0]
            cursor.execute('SELECT COUNT(*) FROM correlations')
            corr_after = cursor.fetchone()[0]
            
            if papers_after == 0 and corr_after == 0:
                print("\n✅ Database cleared successfully!")
            else:
                print(f"\n⚠️  Some data remains: {papers_after} papers, {corr_after} correlations")
        
    except sqlite3.OperationalError as e:
        if "locked" in str(e):
            print("\n❌ Database is locked! Another process is using it.")
            print("Solutions:")
            print("1. Close any other Python scripts accessing the database")
            print("2. Close any database viewers (DB Browser, etc.)")
            print("3. Restart your Python kernel/terminal")
            print("4. Check if OneDrive is syncing the file (pause sync temporarily)")
            print("5. If all else fails, restart your computer")
        else:
            print(f"❌ Database error: {e}")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    clear_database_directly()