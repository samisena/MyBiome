"""
Re-extract mechanisms for interventions that were processed before mechanism extraction was added.

This script:
1. Finds all interventions where mechanism IS NULL (579 interventions from Oct 2-4, 2025)
2. Gets their original papers (288 unique papers)
3. Re-runs LLM extraction with a focused mechanism-only prompt
4. Updates only the mechanism field (preserves all other data)
5. Supports resumable sessions for interruption recovery

Timeline context:
- Oct 2-4: 579 interventions extracted without mechanism field
- Oct 5: Mechanism extraction added (commit 7f3089d)
- Oct 6-7: 792 interventions extracted with mechanisms (100% coverage)
"""

import sqlite3
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass

from back_end.src.data.config import config, setup_logging
from back_end.src.data.api_clients import get_llm_client
from back_end.src.data.utils import parse_json_safely

logger = setup_logging(__name__, 'reextract_mechanisms.log')


@dataclass
class ReextractionSession:
    """Track re-extraction session state."""
    session_id: str
    started_at: str
    total_papers: int
    processed_papers: int
    updated_interventions: int
    failed_papers: List[str]
    last_processed_pmid: Optional[str] = None


class MechanismReextractor:
    """Re-extract mechanisms for interventions missing this field."""

    def __init__(self, db_path: str = None):
        """Initialize the re-extractor."""
        if db_path is None:
            db_path = Path(__file__).parent.parent.parent / "data" / "processed" / "intervention_research.db"

        self.db_path = Path(db_path)
        self.llm_client = get_llm_client('qwen3:14b')
        self.session_file = Path(__file__).parent.parent.parent / "data" / "mechanism_reextraction_session.json"

        logger.info("Initialized MechanismReextractor")

    def create_mechanism_extraction_prompt(self, paper: Dict, interventions: List[Dict]) -> str:
        """
        Create a focused prompt to extract ONLY mechanisms for specific interventions.

        Args:
            paper: Paper dict with title, abstract
            interventions: List of interventions needing mechanisms

        Returns:
            Formatted prompt string
        """
        # Format interventions list
        interventions_list = "\n".join([
            f"{i+1}. \"{intervention['intervention_name']}\" for {intervention['health_condition']}"
            for i, intervention in enumerate(interventions)
        ])

        prompt = f"""You are a biomedical research expert. Extract the mechanism of action for specific interventions from this paper.

PAPER:
Title: {paper['title']}
Abstract: {paper['abstract']}

INTERVENTIONS TO ANALYZE:
{interventions_list}

TASK: For each intervention above, extract HOW it works (mechanism of action).

MECHANISM GUIDELINES:
- Describe the biological, behavioral, or psychological pathway
- Be concise (1-2 sentences maximum)
- Focus on the ACTION, not the intervention name or condition
- Examples:
  * "gut microbiome modulation and reduced inflammation"
  * "improved insulin sensitivity through glucose uptake"
  * "enhanced cardiac function and hemodynamic stability"
  * "psychological restructuring and improved self-perception"

OUTPUT FORMAT:
Respond with valid JSON array ONLY. Start with [ and end with ]. NO markdown, NO explanations.

[
  {{
    "intervention_name": "exact name from list above",
    "health_condition": "exact condition from list above",
    "mechanism": "concise mechanism description" or null if not found
  }},
  ...
]

CRITICAL: If mechanism is not described in the paper, set mechanism to null. Do NOT invent mechanisms."""

        return prompt

    def get_papers_needing_reextraction(self) -> List[Dict]:
        """Get all papers with interventions missing mechanisms."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Get papers and their interventions without mechanisms
        cursor.execute("""
            SELECT
                i.paper_id,
                p.title,
                p.abstract,
                COUNT(*) as intervention_count
            FROM interventions i
            LEFT JOIN papers p ON i.paper_id = p.pmid
            WHERE i.mechanism IS NULL
            GROUP BY i.paper_id
            ORDER BY i.paper_id
        """)

        papers = []
        for row in cursor.fetchall():
            papers.append({
                'pmid': row['paper_id'],
                'title': row['title'] or '',
                'abstract': row['abstract'] or '',
                'intervention_count': row['intervention_count']
            })

        conn.close()
        return papers

    def get_interventions_for_paper(self, pmid: str) -> List[Dict]:
        """Get all interventions for a paper that need mechanisms."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute("""
            SELECT id, intervention_name, health_condition
            FROM interventions
            WHERE paper_id = ? AND mechanism IS NULL
        """, (pmid,))

        interventions = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return interventions

    def update_intervention_mechanisms(self, mechanisms: List[Dict]):
        """Update mechanism field for interventions."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        updated_count = 0
        for mech in mechanisms:
            if mech.get('mechanism'):
                cursor.execute("""
                    UPDATE interventions
                    SET mechanism = ?
                    WHERE intervention_name = ?
                      AND health_condition = ?
                      AND mechanism IS NULL
                """, (mech['mechanism'], mech['intervention_name'], mech['health_condition']))

                updated_count += cursor.rowcount

        conn.commit()
        conn.close()
        return updated_count

    def extract_mechanisms_for_paper(self, paper: Dict) -> Tuple[List[Dict], Optional[str]]:
        """
        Extract mechanisms for all interventions in a paper.

        Returns:
            (mechanisms, error_message)
        """
        # Get interventions needing mechanisms
        interventions = self.get_interventions_for_paper(paper['pmid'])

        if not interventions:
            return [], None

        # Create prompt
        prompt = self.create_mechanism_extraction_prompt(paper, interventions)

        # System message to suppress chain-of-thought (qwen3:14b optimization)
        system_message = "Provide only the final JSON output without showing your reasoning process or using <think> tags. Start your response immediately with the [ character."

        try:
            # Call LLM
            start_time = time.time()
            response = self.llm_client.chat(
                model='qwen3:14b',
                messages=[
                    {'role': 'system', 'content': system_message},
                    {'role': 'user', 'content': prompt}
                ],
                options={'temperature': 0.3}
            )

            extraction_time = time.time() - start_time

            # Parse response
            response_text = response['message']['content']

            # Strip <think> tags if present (qwen3 optimization)
            import re
            response_text = re.sub(r'<think>.*?</think>', '', response_text, flags=re.DOTALL)
            response_text = response_text.strip()

            # Parse JSON
            mechanisms = parse_json_safely(response_text)

            if not isinstance(mechanisms, list):
                return [], f"Invalid response format (expected list, got {type(mechanisms).__name__})"

            logger.info(f"Extracted {len(mechanisms)} mechanisms for PMID {paper['pmid']} in {extraction_time:.1f}s")
            return mechanisms, None

        except Exception as e:
            error_msg = f"Error extracting mechanisms: {str(e)}"
            logger.error(error_msg)
            return [], error_msg

    def save_session(self, session: ReextractionSession):
        """Save session state to JSON."""
        with open(self.session_file, 'w') as f:
            json.dump({
                'session_id': session.session_id,
                'started_at': session.started_at,
                'total_papers': session.total_papers,
                'processed_papers': session.processed_papers,
                'updated_interventions': session.updated_interventions,
                'failed_papers': session.failed_papers,
                'last_processed_pmid': session.last_processed_pmid
            }, f, indent=2)

    def load_session(self) -> Optional[ReextractionSession]:
        """Load session state from JSON."""
        if not self.session_file.exists():
            return None

        with open(self.session_file, 'r') as f:
            data = json.load(f)
            return ReextractionSession(**data)

    def run_reextraction(self, resume: bool = False, max_papers: Optional[int] = None):
        """
        Run the mechanism re-extraction process.

        Args:
            resume: Resume from previous session if available
            max_papers: Maximum papers to process (for testing)
        """
        # Try to load session if resuming
        session = None
        if resume:
            session = self.load_session()
            if session:
                logger.info(f"Resuming session {session.session_id} from {session.last_processed_pmid}")
                print(f"\nResuming previous session...")
                print(f"  Already processed: {session.processed_papers}/{session.total_papers} papers")
                print(f"  Already updated: {session.updated_interventions} interventions")

        # Get papers needing re-extraction
        all_papers = self.get_papers_needing_reextraction()

        # If resuming, skip already processed papers
        if session and session.last_processed_pmid:
            # Find index of last processed paper
            last_idx = next((i for i, p in enumerate(all_papers) if p['pmid'] == session.last_processed_pmid), -1)
            papers_to_process = all_papers[last_idx + 1:]
        else:
            papers_to_process = all_papers
            session = ReextractionSession(
                session_id=datetime.now().strftime("%Y%m%d_%H%M%S"),
                started_at=datetime.now().isoformat(),
                total_papers=len(all_papers),
                processed_papers=0,
                updated_interventions=0,
                failed_papers=[]
            )

        # Limit papers if testing
        if max_papers:
            papers_to_process = papers_to_process[:max_papers]

        print(f"\n{'='*70}")
        print(f"MECHANISM RE-EXTRACTION")
        print(f"{'='*70}")
        print(f"Total papers needing re-extraction: {len(all_papers)}")
        print(f"Papers to process this run: {len(papers_to_process)}")
        print(f"Interventions without mechanisms: {sum(p['intervention_count'] for p in all_papers)}")
        print(f"{'='*70}\n")

        # Process papers
        for i, paper in enumerate(papers_to_process, 1):
            print(f"[{session.processed_papers + i}/{session.total_papers}] Processing PMID {paper['pmid']}...")
            print(f"  Title: {paper['title'][:60]}...")
            print(f"  Interventions: {paper['intervention_count']}")

            # Extract mechanisms
            mechanisms, error = self.extract_mechanisms_for_paper(paper)

            if error:
                print(f"  ❌ Error: {error}")
                session.failed_papers.append(paper['pmid'])
            else:
                # Update database
                updated = self.update_intervention_mechanisms(mechanisms)
                session.updated_interventions += updated
                print(f"  ✓ Updated {updated} interventions")

            session.processed_papers += 1
            session.last_processed_pmid = paper['pmid']

            # Save session every 10 papers
            if i % 10 == 0:
                self.save_session(session)

            # Small delay to avoid overwhelming the LLM
            time.sleep(0.5)

        # Final save
        self.save_session(session)

        # Summary
        print(f"\n{'='*70}")
        print(f"RE-EXTRACTION COMPLETE")
        print(f"{'='*70}")
        print(f"Papers processed: {session.processed_papers}/{session.total_papers}")
        print(f"Interventions updated: {session.updated_interventions}")
        print(f"Failed papers: {len(session.failed_papers)}")
        if session.failed_papers:
            print(f"Failed PMIDs: {', '.join(session.failed_papers[:10])}")
        print(f"{'='*70}\n")


def main():
    """Main execution."""
    import argparse

    parser = argparse.ArgumentParser(description="Re-extract mechanisms for interventions")
    parser.add_argument('--resume', action='store_true', help='Resume from previous session')
    parser.add_argument('--test', action='store_true', help='Test mode: process only 5 papers')
    parser.add_argument('--max-papers', type=int, help='Maximum papers to process')

    args = parser.parse_args()

    reextractor = MechanismReextractor()

    if args.test:
        print("\n⚠️  TEST MODE: Processing only 5 papers\n")
        reextractor.run_reextraction(resume=args.resume, max_papers=5)
    else:
        reextractor.run_reextraction(resume=args.resume, max_papers=args.max_papers)


if __name__ == "__main__":
    main()