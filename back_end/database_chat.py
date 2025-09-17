#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Streamlined Database Chat - Browse abstracts and view extracted interventions
A simple terminal interface to browse research paper abstracts and their extracted interventions.
"""

import sys
from pathlib import Path
from typing import List, Dict, Optional, Any
import json
import re

# Set UTF-8 encoding for Windows console
if sys.platform == "win32":
    import codecs
    sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
    sys.stderr = codecs.getwriter("utf-8")(sys.stderr.detach())

# Add the src directory to Python path
src_dir = Path(__file__).parent / "src"
sys.path.insert(0, str(src_dir))

# Import database manager
try:
    from src.paper_collection.database_manager import database_manager
    print("✓ Database connected")
    DATABASE_AVAILABLE = True
except Exception as e:
    print(f"❌ Database not available: {e}")
    database_manager = None
    DATABASE_AVAILABLE = False


class DatabaseChatbot:
    """Simple chatbot for browsing abstracts and interventions."""

    def __init__(self) -> None:
        self.db = database_manager if DATABASE_AVAILABLE else None
        self.database_available = DATABASE_AVAILABLE
        self.last_papers_list: List[Dict] = []  # Store last paper list for navigation

    def start(self) -> None:
        """Start the interactive chat session."""
        print("\n" + "="*60)
        print("🧠 MyBiome LLM Performance Reviewer")
        print("="*60)
        print("Review LLM extraction performance: see abstracts + extracted interventions.")
        print("\nType 'help' for commands or 'quit' to exit.")
        self.main_loop()

    def main_loop(self) -> None:
        """Main interactive loop."""
        while True:
            try:
                user_input = input("\n🔍 > ").strip()

                if not user_input:
                    continue

                if user_input.lower() in ['quit', 'exit', 'bye']:
                    print("\n👋 Goodbye!")
                    break

                response = self.process_input(user_input)
                print(f"\n{response}")

            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except EOFError:
                print("\n\n👋 Session ended. Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")

    def process_input(self, user_input: str) -> str:
        """Process user input and return appropriate response."""
        input_lower = user_input.lower()

        # Help command
        if input_lower in ['help', 'h', '?']:
            return self.show_help()

        # Status
        if any(word in input_lower for word in ['status', 'stats']):
            return self.get_status()

        # Browse/List abstracts
        if any(word in input_lower for word in ['browse', 'list', 'papers']):
            return self.browse_papers()

        # Show specific paper by PMID
        if re.search(r'\b\d{8}\b', user_input):
            pmid = re.search(r'\b(\d{8})\b', user_input).group(1)
            return self.show_paper_details(pmid)

        # Show paper by number from list
        if re.match(r'^(abstract\s+)?(\d+)$', input_lower):
            match = re.match(r'^(?:abstract\s+)?(\d+)$', input_lower)
            paper_num = int(match.group(1))
            return self.show_paper_by_number(paper_num)

        # Search
        if any(word in input_lower for word in ['search', 'find']):
            # Extract search term
            for phrase in ['search', 'find']:
                if phrase in input_lower:
                    parts = user_input.lower().split(phrase, 1)
                    if len(parts) > 1:
                        search_term = parts[1].strip()
                        return self.search_papers(search_term)
            return "What would you like to search for? Try: 'search diabetes'"

        return self.get_default_response()

    def show_help(self) -> str:
        """Show available commands."""
        return """🧠 LLM Performance Review Commands:

📊 Status:
   • status - Database overview

📝 Review LLM Extractions:
   • browse, list - Show recent papers for review
   • [number] - Review LLM performance on paper N (e.g., "3")
   • [PMID] - Review specific paper by PMID (e.g., "12345678")

🔍 Search Papers to Review:
   • search [term] - Find papers containing term
   • find [intervention] - Search for specific interventions

Each paper view shows:
   1. Original abstract (full text)
   2. All LLM-extracted interventions/correlations
   3. Confidence scores and extraction details
   4. Review questions to guide your evaluation

Examples:
   • browse → see recent papers
   • 3 → review LLM performance on 3rd paper
   • search diabetes → find diabetes papers to review"""

    def get_status(self) -> str:
        """Get basic database status."""
        if not self.database_available:
            return "⚠️ Database is not available."

        try:
            stats = self.db.get_database_stats()
            return f"""📊 Database Status:

📄 Total Papers: {stats.get('total_papers', 0):,}
📚 Papers with Abstracts: {stats.get('papers_with_fulltext', 0):,}
💊 Extracted Interventions: {stats.get('total_interventions', 0):,}

Type 'browse' to start exploring papers."""

        except Exception as e:
            return f"❌ Error getting status: {e}"

    def browse_papers(self) -> str:
        """Browse recent papers with abstracts."""
        if not self.database_available:
            return "⚠️ Database not available."

        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()

                # Prioritize papers with interventions for review
                cursor.execute('''
                    SELECT p.pmid, p.title, p.abstract, p.journal, p.publication_date,
                           COUNT(i.id) as intervention_count
                    FROM papers p
                    LEFT JOIN interventions i ON p.pmid = i.paper_id
                    WHERE p.abstract IS NOT NULL AND p.abstract != ''
                    GROUP BY p.pmid, p.title, p.abstract, p.journal, p.publication_date
                    ORDER BY intervention_count DESC, p.created_at DESC
                    LIMIT 20
                ''')

                papers = cursor.fetchall()

                if not papers:
                    return "❌ No papers with abstracts found."

                # Store for navigation
                self.last_papers_list = papers

                response = ["🧠 Papers Available for LLM Performance Review:\n"]

                for i, paper in enumerate(papers, 1):
                    # Show preview of abstract
                    abstract_preview = paper['abstract'][:120] + "..." if len(paper['abstract']) > 120 else paper['abstract']

                    # Use intervention count from the query
                    intervention_count = paper['intervention_count']

                    # Add emoji indicator
                    if intervention_count > 0:
                        status_emoji = "🎯"
                        status_text = f"Extractions: {intervention_count}"
                    else:
                        status_emoji = "❌"
                        status_text = "No extractions yet"

                    response.append(f"{i:2d}. {status_emoji} PMID: {paper['pmid']} | {status_text}")
                    response.append(f"    Title: {paper['title']}")
                    response.append(f"    Journal: {paper['journal'] or 'Unknown'}")
                    response.append(f"    Preview: {abstract_preview}")
                    response.append("")

                response.append("🎯 Papers with extractions listed first for easy review")
                response.append("📋 Type a number (1-20) to see: Abstract + LLM Extractions + Review Questions")
                response.append("❌ Papers without extractions show why LLM found nothing")
                response.append("💡 Or type a PMID directly to review any specific paper")

                return "\n".join(response)

        except Exception as e:
            return f"❌ Error browsing papers: {e}"

    def show_paper_by_number(self, paper_num: int) -> str:
        """Show paper details by number from last list."""
        if not self.last_papers_list:
            return "❌ No paper list available. Use 'browse' first."

        if paper_num < 1 or paper_num > len(self.last_papers_list):
            return f"❌ Invalid number. Choose 1-{len(self.last_papers_list)}"

        paper = self.last_papers_list[paper_num - 1]
        return self.show_paper_details(str(paper['pmid']))

    def show_paper_details(self, pmid: str) -> str:
        """Show full paper details with abstract and extracted interventions."""
        if not self.database_available:
            return "⚠️ Database not available."

        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()

                # Get paper details
                cursor.execute('''
                    SELECT pmid, title, abstract, journal, publication_date, doi
                    FROM papers
                    WHERE pmid = ?
                ''', (pmid,))

                paper = cursor.fetchone()
                if not paper:
                    return f"❌ Paper PMID {pmid} not found."

                # Get extracted interventions with model information
                cursor.execute('''
                    SELECT intervention_name, intervention_category, health_condition,
                           correlation_type, correlation_strength, confidence_score,
                           intervention_details, extraction_model, sample_size, study_duration
                    FROM interventions
                    WHERE paper_id = ?
                    ORDER BY extraction_model, confidence_score DESC NULLS LAST
                ''', (pmid,))

                interventions = cursor.fetchall()

                # Build response for LLM performance review
                response = [f"🧠 LLM PERFORMANCE REVIEW - PMID: {paper['pmid']}\n"]
                response.append("=" * 80)
                response.append(f"Title: {paper['title']}")
                response.append(f"Journal: {paper['journal'] or 'Unknown'} | Date: {paper['publication_date'] or 'Unknown'}")
                if paper['doi']:
                    response.append(f"DOI: {paper['doi']}")

                # Show extraction stats first
                if interventions:
                    response.append(f"🎯 LLM Extracted: {len(interventions)} interventions")
                else:
                    response.append("⚠️ LLM Extracted: 0 interventions")

                response.append("\n" + "=" * 80)
                response.append("📝 SOURCE ABSTRACT")
                response.append("=" * 80)

                # Full abstract
                if paper['abstract']:
                    response.append(paper['abstract'])
                else:
                    response.append("⚠️ No abstract available")

                response.append("\n" + "=" * 80)
                response.append("🤖 LLM EXTRACTED CORRELATIONS")
                response.append("=" * 80)

                # Show extracted interventions for performance review
                if interventions:
                    for i, intervention in enumerate(interventions, 1):
                        # Correlation emoji with clearer meanings
                        corr_emoji = {
                            'positive': '🟢',  # Helps with condition
                            'negative': '🔴',  # Worsens condition
                            'neutral': '⚪',   # No significant effect
                            'inconclusive': '❓'  # Unclear/conflicting evidence
                        }.get(intervention['correlation_type'], '❓')

                        # Strength indicator
                        strength = intervention['correlation_strength']
                        strength_indicator = ""
                        if strength:
                            if strength >= 0.8:
                                strength_indicator = "🔴🔴🔴"
                            elif strength >= 0.6:
                                strength_indicator = "🔴🔴"
                            elif strength >= 0.4:
                                strength_indicator = "🔴"
                            else:
                                strength_indicator = "⚪"

                        # Map correlation types to clearer language
                        effect_description = {
                            'positive': 'helps with',
                            'negative': 'worsens',
                            'neutral': 'no significant effect on',
                            'inconclusive': 'unclear effect on'
                        }.get(intervention['correlation_type'], intervention['correlation_type'])

                        response.append(f"\n{i}. {corr_emoji} INTERVENTION: {intervention['intervention_name']}")
                        response.append(f"   CONDITION: {intervention['health_condition']}")
                        response.append(f"   CATEGORY: {intervention['intervention_category']}")
                        response.append(f"   EFFECT: {effect_description} {intervention['health_condition']} {strength_indicator}")

                        # Show confidence and extraction model (directly from the query)
                        conf_line = []
                        try:
                            if intervention['confidence_score']:
                                conf_line.append(f"Confidence: {intervention['confidence_score']:.3f}")
                        except (KeyError, TypeError):
                            pass

                        # Show which model extracted this (no extra query needed)
                        try:
                            if intervention['extraction_model']:
                                conf_line.append(f"Model: {intervention['extraction_model']}")
                        except (KeyError, TypeError):
                            pass

                        if conf_line:
                            response.append(f"   {' | '.join(conf_line)}")

                        # Show detailed extraction if available
                        if intervention['intervention_details']:
                            try:
                                details = json.loads(intervention['intervention_details'])
                                if isinstance(details, dict) and details:
                                    response.append("   DETAILS:")
                                    for key, value in details.items():
                                        response.append(f"     • {key}: {value}")
                            except (json.JSONDecodeError, TypeError):
                                response.append(f"   DETAILS: {intervention['intervention_details']}")

                        # Show sample size and study info if available
                        study_info = []
                        try:
                            if intervention['sample_size']:
                                study_info.append(f"Sample: {intervention['sample_size']}")
                        except (KeyError, TypeError):
                            pass
                        try:
                            if intervention['study_duration']:
                                study_info.append(f"Duration: {intervention['study_duration']}")
                        except (KeyError, TypeError):
                            pass

                        if study_info:
                            response.append(f"   STUDY: {' | '.join(study_info)}")
                else:
                    response.append("\n❌ LLM found no interventions in this abstract.")
                    response.append("\nPossible reasons:")
                    response.append("• Abstract contains no intervention data")
                    response.append("• Interventions were too subtle/complex for LLM")
                    response.append("• LLM extraction failed or had low confidence")

                response.append("\n" + "=" * 80)
                response.append("📊 REVIEW QUESTIONS FOR YOU:")
                response.append("• Did the LLM miss any obvious interventions?")
                response.append("• Are the extracted correlations logically consistent?")
                response.append("  (e.g., if gluten-free diet HELPS with IBS, then gluten should WORSEN it)")
                response.append("• Do both models agree on the same interventions?")
                response.append("• Are the confidence scores reasonable?")
                response.append("• Are intervention categories correctly classified?")
                response.append("• Are the correlation directions (helps vs worsens) accurate?")

                return "\n".join(response)

        except Exception as e:
            return f"❌ Error retrieving paper details: {e}"

    def search_papers(self, search_term: str) -> str:
        """Search for papers containing the search term."""
        if not self.database_available:
            return "⚠️ Database not available."

        if not search_term:
            return "❌ Please provide a search term."

        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()

                # Search in title and abstract, include intervention count
                cursor.execute('''
                    SELECT p.pmid, p.title, p.abstract, p.journal, p.publication_date,
                           COUNT(i.id) as intervention_count
                    FROM papers p
                    LEFT JOIN interventions i ON p.pmid = i.paper_id
                    WHERE (LOWER(p.title) LIKE LOWER(?) OR LOWER(p.abstract) LIKE LOWER(?))
                      AND p.abstract IS NOT NULL
                      AND p.abstract != ''
                    GROUP BY p.pmid, p.title, p.abstract, p.journal, p.publication_date
                    ORDER BY
                        intervention_count DESC,
                        CASE
                            WHEN LOWER(p.title) LIKE LOWER(?) THEN 1
                            ELSE 2
                        END,
                        p.publication_date DESC
                    LIMIT 15
                ''', (f'%{search_term}%', f'%{search_term}%', f'%{search_term}%'))

                papers = cursor.fetchall()

                if not papers:
                    return f"❌ No papers found containing '{search_term}'"

                # Store for navigation
                self.last_papers_list = papers

                response = [f"🔍 Found {len(papers)} papers for '{search_term}':\n"]

                for i, paper in enumerate(papers, 1):
                    # Highlight search term in title
                    title = paper['title']
                    if search_term.lower() in title.lower():
                        # Simple highlighting for terminal
                        title = re.sub(f'({re.escape(search_term)})', r'**\1**', title, flags=re.IGNORECASE)

                    # Abstract preview with search term context
                    abstract = paper['abstract']
                    preview = self.get_search_preview(abstract, search_term)

                    # Use intervention count from the query
                    intervention_count = paper['intervention_count']

                    # Add emoji indicator
                    if intervention_count > 0:
                        status_emoji = "🎯"
                        status_text = f"Extractions: {intervention_count}"
                    else:
                        status_emoji = "❌"
                        status_text = "No extractions yet"

                    response.append(f"{i:2d}. {status_emoji} PMID: {paper['pmid']} | {status_text}")
                    response.append(f"    Title: {title}")
                    response.append(f"    Journal: {paper['journal'] or 'Unknown'}")
                    response.append(f"    Preview: {preview}")
                    response.append("")

                response.append(f"💡 Type a number (1-{len(papers)}) to view full details")

                return "\n".join(response)

        except Exception as e:
            return f"❌ Error searching papers: {e}"

    def get_search_preview(self, text: str, search_term: str, context_chars: int = 100) -> str:
        """Get a preview of text around the search term."""
        text_lower = text.lower()
        term_lower = search_term.lower()

        pos = text_lower.find(term_lower)
        if pos == -1:
            return text[:150] + "..."

        # Get context around the search term
        start = max(0, pos - context_chars)
        end = min(len(text), pos + len(search_term) + context_chars)

        preview = text[start:end]
        if start > 0:
            preview = "..." + preview
        if end < len(text):
            preview = preview + "..."

        # Highlight the search term
        preview = re.sub(f'({re.escape(search_term)})', r'**\1**', preview, flags=re.IGNORECASE)

        return preview

    def get_default_response(self) -> str:
        """Default response for unrecognized input."""
        return """❓ I didn't understand that. Try:

• 'browse' - see papers for LLM performance review
• 'search [term]' - find specific papers to review
• [number] - review LLM performance on paper N
• 'help' - show all commands"""


def main():
    """Main entry point."""
    try:
        if not DATABASE_AVAILABLE:
            print("❌ Database not available. Please check your setup.")
            return

        chatbot = DatabaseChatbot()
        chatbot.start()

    except Exception as e:
        print(f"❌ Failed to start: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()