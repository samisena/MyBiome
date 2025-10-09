"""
Select 16 papers for testing using best available sampling.
"""

import json
import sys
from pathlib import Path
from typing import List, Dict

sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from back_end.src.data.config import config, setup_logging
from back_end.src.data_collection.database_manager import database_manager

logger = setup_logging(__name__, 'dataset_selector.log')


class DatasetSelector:
    """Select sample of papers for experiments."""

    def __init__(self, num_papers: int = 16):
        """
        Initialize dataset selector.

        Args:
            num_papers: Total number of papers to select (default 16)
        """
        self.num_papers = num_papers

    def select_papers(self) -> List[Dict]:
        """
        Select up to 16 papers (or as many as available).

        Returns:
            List of paper dictionaries
        """
        papers = []

        with database_manager.get_connection() as conn:
            cursor = conn.cursor()

            # Check how many papers we have
            cursor.execute("""
                SELECT COUNT(*) FROM papers
                WHERE abstract IS NOT NULL AND abstract != ''
            """)
            total_available = cursor.fetchone()[0]
            logger.info(f"Total papers available: {total_available}")

            # Select papers prioritizing by citation count and fulltext availability
            target = min(self.num_papers, total_available)
            logger.info(f"Selecting {target} papers")

            cursor.execute("""
                SELECT pmid, title, abstract, has_fulltext, fulltext_path, citation_count
                FROM papers
                WHERE abstract IS NOT NULL
                    AND abstract != ''
                ORDER BY
                    CASE WHEN has_fulltext = 1 THEN 0 ELSE 1 END,
                    citation_count DESC NULLS LAST,
                    RANDOM()
                LIMIT ?
            """, (target,))

            papers = self._fetch_papers(cursor)
            logger.info(f"Selected {len(papers)} papers")

        return papers

    def _fetch_papers(self, cursor) -> List[Dict]:
        """Fetch papers from cursor and convert to dictionaries."""
        columns = [desc[0] for desc in cursor.description]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]

    def save_dataset(self, papers: List[Dict], output_path: str):
        """
        Save selected papers to JSON file.

        Args:
            papers: List of paper dictionaries
            output_path: Path to save JSON file
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(papers, f, indent=2, default=str)

        logger.info(f"Dataset saved to {output_path}")

    def load_dataset(self, input_path: str) -> List[Dict]:
        """
        Load dataset from JSON file.

        Args:
            input_path: Path to JSON file

        Returns:
            List of paper dictionaries
        """
        with open(input_path, 'r', encoding='utf-8') as f:
            papers = json.load(f)

        logger.info(f"Loaded {len(papers)} papers from {input_path}")
        return papers


if __name__ == "__main__":
    """Select and save test dataset."""

    selector = DatasetSelector(num_papers=16)
    papers = selector.select_papers()

    # Save to data directory
    output_path = Path(__file__).parent.parent / "data" / "test_dataset.json"
    selector.save_dataset(papers, str(output_path))

    print(f"\nDataset Selection Summary:")
    print(f"Total papers: {len(papers)}")
    print(f"Saved to: {output_path}")

    # Print summary statistics
    fulltext = sum(1 for p in papers if p.get('has_fulltext'))
    abstract = len(papers) - fulltext
    avg_citations = sum(p.get('citation_count') or 0 for p in papers) / len(papers) if papers else 0

    print(f"\nDataset Composition:")
    print(f"  Full-text: {fulltext}")
    print(f"  Abstract-only: {abstract}")
    print(f"  Average citations: {avg_citations:.1f}")
