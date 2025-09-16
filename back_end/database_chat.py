#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Interactive Database Chat - Talk with your MyBiome database
A conversational terminal interface to explore database status, correlations, and health conditions.
"""

import sys
import os
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import json
import importlib.util

# Set UTF-8 encoding for Windows console
if sys.platform == "win32":
    import codecs
    sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
    sys.stderr = codecs.getwriter("utf-8")(sys.stderr.detach())

# Add the src directory to Python path
src_dir = Path(__file__).parent / "src"
sys.path.insert(0, str(src_dir))

# Import modules directly to avoid circular dependencies
def safe_import_module(module_path, module_name):
    """Safely import a module by file path."""
    try:
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[f'src.{module_name}'] = module
        spec.loader.exec_module(module)
        return module
    except Exception as e:
        print(f"Error importing {module_name}: {e}")
        return None

# Import required modules
try:
    # Load taxonomy first
    taxonomy_path = src_dir / "interventions" / "taxonomy.py"
    taxonomy_module = safe_import_module(taxonomy_path, "interventions.taxonomy")
    if not taxonomy_module:
        raise ImportError("Could not load taxonomy module")

    intervention_taxonomy = taxonomy_module.intervention_taxonomy
    InterventionType = taxonomy_module.InterventionType

    print("✓ Successfully loaded intervention system")

    # Try to load database manager
    try:
        from src.paper_collection.database_manager import database_manager
        print("✓ Successfully connected to database")
        DATABASE_AVAILABLE = True
    except Exception as e:
        print(f"⚠ Database manager not available: {e}")
        print("Running in limited mode - some features may not work")
        database_manager = None
        DATABASE_AVAILABLE = False

except ImportError as e:
    print(f"❌ Critical import error: {e}")
    print("Please make sure you're running this from the back_end directory")
    sys.exit(1)


class DatabaseChatbot:
    """Interactive chatbot for database exploration."""

    def __init__(self):
        self.db = database_manager if DATABASE_AVAILABLE else None
        self.taxonomy = intervention_taxonomy
        self.current_stats = None
        self.greeting_shown = False
        self.database_available = DATABASE_AVAILABLE

    def start(self):
        """Start the interactive chat session."""
        self.show_greeting()
        self.main_loop()

    def show_greeting(self):
        """Show the initial greeting and database status."""
        print("\n" + "="*60)
        print("🧬 MyBiome Database Chat Interface")
        print("="*60)
        print("Hello! I'm your database assistant. Let me check my current status...")

        try:
            self.current_stats = self.db.get_database_stats()
            self.show_status_summary()
        except Exception as e:
            print(f"❌ Oops! I'm having trouble accessing my data: {e}")
            print("But I'm still here to help with what I can!")

        print("\n💬 What would you like to know? Type 'help' for options or 'quit' to exit.")
        self.greeting_shown = True

    def show_status_summary(self):
        """Show a conversational status summary."""
        if not self.database_available:
            print("⚠ I'm running in limited mode - database features are not available.")
            print("I can still help you explore intervention categories and provide guidance!")
            return

        if not self.current_stats:
            print("❌ I can't access my current statistics right now.")
            return

        stats = self.current_stats

        print(f"\n📊 Here's what I'm currently storing:")
        print(f"   📄 {stats.get('total_papers', 0):,} research papers")
        print(f"   💊 {stats.get('total_interventions', 0):,} intervention correlations")
        print(f"   📚 {stats.get('papers_with_fulltext', 0):,} papers with full text")

        # Health conditions
        if 'top_health_conditions' in stats:
            top_conditions = stats['top_health_conditions'][:3]
            if top_conditions:
                print(f"\n🏥 My top health conditions by intervention count:")
                for i, condition in enumerate(top_conditions, 1):
                    print(f"   {i}. {condition['condition']} ({condition['interventions']} interventions)")

        # Validation status
        if 'validation_status' in stats:
            validation = stats['validation_status']
            total_pending = validation.get('pending', 0)
            total_verified = validation.get('verified', 0)
            if total_pending > 0:
                print(f"\n⚠️  I have {total_pending:,} interventions waiting for validation")
            if total_verified > 0:
                print(f"✅ {total_verified:,} interventions have been verified")

        # Quality issues
        quality_issues = stats.get('quality_issues', 0)
        if quality_issues > 0:
            print(f"⚠️  I've detected {quality_issues:,} potential data quality issues")

    def main_loop(self):
        """Main interactive loop."""
        while True:
            try:
                user_input = input("\n🤖 You: ").strip()

                if not user_input:
                    continue

                if user_input.lower() in ['quit', 'exit', 'bye']:
                    self.show_goodbye()
                    break

                response = self.process_input(user_input)
                print(f"\n🧬 Database: {response}")

            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except EOFError:
                print("\n\n👋 Session ended. Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Sorry, I encountered an error: {e}")
                # Add a safety break to prevent infinite loops
                import time
                time.sleep(0.1)

    def process_input(self, user_input: str) -> str:
        """Process user input and return appropriate response."""
        input_lower = user_input.lower()

        # Help command
        if input_lower in ['help', 'h', '?']:
            return self.show_help()

        # Status and statistics
        if any(word in input_lower for word in ['status', 'stats', 'statistics', 'summary']):
            return self.get_detailed_status()

        # Health condition queries
        if any(word in input_lower for word in ['condition', 'disease', 'illness', 'disorder']):
            return self.handle_condition_query(user_input)

        # Intervention categories
        if any(word in input_lower for word in ['intervention', 'treatment', 'therapy', 'category']):
            return self.handle_intervention_query(user_input)

        # Validation and quality
        if any(word in input_lower for word in ['validation', 'validate', 'quality', 'issues', 'problems']):
            return self.handle_validation_query(user_input)

        # Emerging categories
        if any(word in input_lower for word in ['emerging', 'new', 'novel', 'unclassified']):
            return self.handle_emerging_query(user_input)

        # Papers and research
        if any(word in input_lower for word in ['papers', 'research', 'studies', 'publications']):
            return self.handle_papers_query(user_input)

        # Correlations
        if any(word in input_lower for word in ['correlation', 'effect', 'relationship', 'connection']):
            return self.handle_correlation_query(user_input)

        # Default response
        return self.get_default_response(user_input)

    def show_help(self) -> str:
        """Show available commands and options."""
        help_text = """Here's what you can ask me about:

📊 Status & Statistics:
   • "status" or "stats" - Current database overview
   • "how many papers" - Paper counts and details

🏥 Health Conditions:
   • "what conditions" - List all health conditions I know
   • "diabetes" or "depression" - Explore specific conditions
   • "top conditions" - Most researched conditions

💊 Interventions:
   • "what interventions" - Browse intervention categories
   • "exercise" or "medication" - Explore specific intervention types
   • "new categories" - See emerging intervention types

✅ Validation & Quality:
   • "validation status" - Check data validation progress
   • "quality issues" - Any problems I've detected
   • "pending" - Items waiting for review

📚 Research Papers:
   • "latest papers" - Recently added research
   • "papers for [condition]" - Research on specific topics

🔗 Correlations:
   • "correlations" - Overview of intervention effects
   • "positive effects" - Beneficial interventions
   • "negative effects" - Harmful correlations

Type 'quit' to exit. Just ask naturally - I'll do my best to understand!"""
        return help_text

    def get_detailed_status(self) -> str:
        """Get detailed database status."""
        if not self.database_available:
            return "⚠ Database is not available. I can still help you with:\n" \
                   "• Intervention categories and taxonomy\n" \
                   "• Understanding intervention types\n" \
                   "• General guidance about health research"

        try:
            # Refresh stats
            self.current_stats = self.db.get_database_stats()
            stats = self.current_stats

            response = ["Let me give you a comprehensive overview:\n"]

            # Basic counts
            response.append(f"📄 Papers: {stats.get('total_papers', 0):,} total")
            response.append(f"   └─ {stats.get('papers_with_fulltext', 0):,} with full text available")

            response.append(f"\n💊 Interventions: {stats.get('total_interventions', 0):,} correlations extracted")

            # Category breakdown
            if 'category_breakdown' in stats:
                response.append(f"\n📋 Intervention Categories:")
                for category, count in stats['category_breakdown'].items():
                    response.append(f"   • {category.title()}: {count:,}")

            # Validation status
            if 'validation_status' in stats:
                response.append(f"\n✅ Validation Status:")
                for status, count in stats['validation_status'].items():
                    emoji = {"verified": "✅", "pending": "⏳", "conflicted": "⚠️", "failed": "❌"}.get(status, "📊")
                    response.append(f"   {emoji} {status.title()}: {count:,}")

            # Quality metrics
            quality_issues = stats.get('quality_issues', 0)
            if quality_issues > 0:
                response.append(f"\n⚠️  Quality Issues: {quality_issues:,} items need attention")
            else:
                response.append(f"\n✅ Data Quality: Looking good!")

            return "\n".join(response)

        except Exception as e:
            return f"Sorry, I'm having trouble accessing my current status: {e}"

    def handle_condition_query(self, query: str) -> str:
        """Handle health condition related queries."""
        try:
            stats = self.current_stats or self.db.get_database_stats()

            # Check if asking about specific condition
            query_lower = query.lower()

            # Extract potential condition name from query
            condition_name = self.extract_condition_from_query(query)

            if condition_name:
                return self.get_condition_details(condition_name)

            # General condition overview
            if 'top_health_conditions' in stats:
                conditions = stats['top_health_conditions']
                response = [f"I have research on {len(conditions)} different health conditions.\n"]
                response.append("🏥 Top conditions by intervention count:")

                for i, condition in enumerate(conditions[:10], 1):
                    response.append(f"   {i:2d}. {condition['condition']} ({condition['interventions']} interventions)")

                response.append(f"\nWant details on any condition? Just ask: 'tell me about diabetes' or 'depression research'")
                return "\n".join(response)
            else:
                return "I don't have condition statistics available right now."

        except Exception as e:
            return f"Sorry, I had trouble retrieving condition information: {e}"

    def extract_condition_from_query(self, query: str) -> Optional[str]:
        """Extract a potential health condition from the user's query."""
        # Common health conditions to look for
        common_conditions = [
            'diabetes', 'depression', 'anxiety', 'obesity', 'hypertension', 'cancer',
            'alzheimer', 'arthritis', 'asthma', 'cardiovascular', 'heart disease',
            'stroke', 'migraine', 'fibromyalgia', 'ibs', 'crohn', 'lupus'
        ]

        query_lower = query.lower()
        for condition in common_conditions:
            if condition in query_lower:
                return condition

        return None

    def get_condition_details(self, condition: str) -> str:
        """Get detailed information about a specific health condition."""
        try:
            # Query interventions for this condition
            interventions = self.query_interventions_by_condition(condition)

            if not interventions:
                return f"I don't have any specific research on '{condition}' yet. " \
                       f"Try searching for a broader term or check my top conditions with 'top conditions'."

            response = [f"🏥 Research on '{condition.title()}':\n"]
            response.append(f"Found {len(interventions)} intervention correlations:\n")

            # Group by intervention type and correlation
            positive_effects = []
            negative_effects = []
            neutral_effects = []

            for intervention in interventions[:20]:  # Limit to top 20
                effect_list = {
                    'positive': positive_effects,
                    'negative': negative_effects,
                    'neutral': neutral_effects,
                    'inconclusive': neutral_effects
                }.get(intervention.get('correlation_type', 'neutral'), neutral_effects)

                effect_list.append(intervention)

            # Show positive effects
            if positive_effects:
                response.append("✅ Beneficial interventions:")
                for intervention in positive_effects[:10]:
                    strength_indicator = self.get_strength_indicator(intervention.get('correlation_strength'))
                    response.append(f"   • {intervention['intervention_name']} ({intervention['intervention_category']}) {strength_indicator}")

            # Show negative effects
            if negative_effects:
                response.append("\n❌ Potentially harmful interventions:")
                for intervention in negative_effects[:5]:
                    strength_indicator = self.get_strength_indicator(intervention.get('correlation_strength'))
                    response.append(f"   • {intervention['intervention_name']} ({intervention['intervention_category']}) {strength_indicator}")

            # Show neutral/inconclusive
            if neutral_effects:
                response.append("\n⚪ Neutral or inconclusive interventions:")
                for intervention in neutral_effects[:5]:
                    response.append(f"   • {intervention['intervention_name']} ({intervention['intervention_category']})")

            if len(interventions) > 20:
                response.append(f"\n... and {len(interventions) - 20} more interventions")

            return "\n".join(response)

        except Exception as e:
            return f"Sorry, I couldn't retrieve details for {condition}: {e}"

    def query_interventions_by_condition(self, condition: str) -> List[Dict]:
        """Query interventions for a specific health condition."""
        if not self.database_available or not self.db:
            return []

        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()

                # Search for conditions containing the search term
                cursor.execute('''
                    SELECT intervention_category, intervention_name, health_condition,
                           correlation_type, correlation_strength, confidence_score,
                           sample_size, study_type, extraction_model
                    FROM interventions
                    WHERE LOWER(health_condition) LIKE LOWER(?)
                    ORDER BY
                        CASE correlation_type
                            WHEN 'positive' THEN 1
                            WHEN 'negative' THEN 2
                            ELSE 3
                        END,
                        correlation_strength DESC NULLS LAST,
                        confidence_score DESC NULLS LAST
                ''', (f'%{condition}%',))

                columns = [desc[0] for desc in cursor.description]
                return [dict(zip(columns, row)) for row in cursor.fetchall()]

        except Exception as e:
            print(f"Database query error: {e}")
            return []

    def get_strength_indicator(self, strength: Optional[float]) -> str:
        """Get a visual indicator for correlation strength."""
        if strength is None:
            return ""

        if strength >= 0.8:
            return "🔴🔴🔴"  # Very strong
        elif strength >= 0.6:
            return "🔴🔴"    # Strong
        elif strength >= 0.4:
            return "🔴"      # Moderate
        else:
            return "⚪"      # Weak

    def handle_intervention_query(self, query: str) -> str:
        """Handle intervention-related queries."""
        try:
            categories = list(InterventionType)

            response = ["I track these intervention categories:\n"]

            for i, category in enumerate(categories, 1):
                cat_def = self.taxonomy.get_category(category)
                response.append(f"   {i}. {cat_def.display_name} ({category.value})")
                response.append(f"      └─ Includes: {', '.join(cat_def.subcategories[:4])}{'...' if len(cat_def.subcategories) > 4 else ''}")

            response.append(f"\nI can tell you more about any category. Try: 'tell me about exercise' or 'medication interventions'")

            return "\n".join(response)

        except Exception as e:
            return f"Sorry, I had trouble retrieving intervention information: {e}"

    def handle_validation_query(self, query: str) -> str:
        """Handle validation and quality related queries."""
        try:
            stats = self.current_stats or self.db.get_database_stats()

            validation_status = stats.get('validation_status', {})
            quality_issues = stats.get('quality_issues', 0)

            response = ["Here's my validation and quality status:\n"]

            if validation_status:
                total_items = sum(validation_status.values())
                response.append(f"📊 Total interventions: {total_items:,}")

                for status, count in validation_status.items():
                    percentage = (count / total_items * 100) if total_items > 0 else 0
                    emoji = {"verified": "✅", "pending": "⏳", "conflicted": "⚠️", "failed": "❌"}.get(status, "📊")
                    response.append(f"   {emoji} {status.title()}: {count:,} ({percentage:.1f}%)")

            if quality_issues > 0:
                response.append(f"\n⚠️  I've detected {quality_issues:,} potential quality issues that need attention.")
                response.append("   These might include incomplete data, suspicious patterns, or inconsistencies.")
            else:
                response.append(f"\n✅ Data quality looks good! No major issues detected.")

            return "\n".join(response)

        except Exception as e:
            return f"Sorry, I couldn't retrieve validation information: {e}"

    def handle_emerging_query(self, query: str) -> str:
        """Handle emerging categories and novel interventions."""
        try:
            emerging_interventions = self.query_emerging_interventions()

            if not emerging_interventions:
                return "🎉 Great news! I don't have any unclassified interventions right now. " \
                       "All discovered interventions fit into my existing categories!"

            response = [f"🆕 I found {len(emerging_interventions)} novel interventions that need classification:\n"]

            # Group by proposed category
            by_category = {}
            for intervention in emerging_interventions:
                proposed = intervention.get('proposed_category', 'Unknown')
                if proposed not in by_category:
                    by_category[proposed] = []
                by_category[proposed].append(intervention)

            for proposed_category, interventions in by_category.items():
                response.append(f"📋 Proposed Category: '{proposed_category}' ({len(interventions)} items)")
                for intervention in interventions[:3]:  # Show first 3
                    response.append(f"   • {intervention['intervention_name']}")
                    if intervention.get('category_rationale'):
                        response.append(f"     Reason: {intervention['category_rationale'][:100]}...")
                if len(interventions) > 3:
                    response.append(f"   ... and {len(interventions) - 3} more")
                response.append("")

            response.append("💡 These might become new intervention categories if validated!")
            response.append("Human review needed to confirm classification.")

            return "\n".join(response)

        except Exception as e:
            return f"Sorry, I couldn't retrieve emerging category information: {e}"

    def query_emerging_interventions(self) -> List[Dict]:
        """Query interventions in the EMERGING category."""
        if not self.database_available or not self.db:
            return []

        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()

                cursor.execute('''
                    SELECT intervention_name, health_condition, intervention_details,
                           validation_status, confidence_score
                    FROM interventions
                    WHERE intervention_category = 'emerging'
                    ORDER BY confidence_score DESC NULLS LAST
                ''')

                columns = [desc[0] for desc in cursor.description]
                interventions = [dict(zip(columns, row)) for row in cursor.fetchall()]

                # Parse intervention_details to get emerging category info
                for intervention in interventions:
                    details = intervention.get('intervention_details')
                    if details:
                        try:
                            details_dict = json.loads(details) if isinstance(details, str) else details
                            intervention.update(details_dict)
                        except (json.JSONDecodeError, TypeError):
                            pass

                return interventions

        except Exception as e:
            print(f"Database query error: {e}")
            return []

    def handle_papers_query(self, query: str) -> str:
        """Handle paper and research related queries."""
        try:
            stats = self.current_stats or self.db.get_database_stats()

            response = ["Here's what I know about my research papers:\n"]
            response.append(f"📄 Total papers: {stats.get('total_papers', 0):,}")
            response.append(f"📚 With full text: {stats.get('papers_with_fulltext', 0):,}")

            if 'processing_status' in stats:
                response.append(f"\n📊 Processing Status:")
                for status, count in stats['processing_status'].items():
                    response.append(f"   • {status.title()}: {count:,}")

            return "\n".join(response)

        except Exception as e:
            return f"Sorry, I couldn't retrieve paper information: {e}"

    def handle_correlation_query(self, query: str) -> str:
        """Handle correlation and effectiveness queries."""
        try:
            stats = self.current_stats or self.db.get_database_stats()

            response = ["Let me tell you about the correlations I've found:\n"]
            response.append(f"🔗 Total correlations: {stats.get('total_interventions', 0):,}")

            # This would be enhanced to show correlation breakdowns by type
            response.append("\n📈 Correlation Types:")
            response.append("   (This feature is being developed to show you positive,")
            response.append("    negative, and neutral intervention effects)")

            return "\n".join(response)

        except Exception as e:
            return f"Sorry, I couldn't retrieve correlation information: {e}"

    def get_default_response(self, query: str) -> str:
        """Provide a helpful default response."""
        responses = [
            "I'm not sure I understand that question. Could you rephrase it?",
            "Hmm, that's not something I can help with yet. Try asking about my status, health conditions, or interventions.",
            "I didn't quite catch that. Type 'help' to see what I can assist you with!",
            "That's an interesting question! Right now I can tell you about my database status, health conditions, and interventions."
        ]

        # Simple response selection based on query length
        response_idx = len(query) % len(responses)
        base_response = responses[response_idx]

        return f"{base_response}\n\nType 'help' to see all available options, or try:\n" \
               f"• 'status' - Database overview\n" \
               f"• 'what conditions' - Health conditions I know\n" \
               f"• 'interventions' - Treatment categories"

    def show_goodbye(self):
        """Show goodbye message."""
        print("\n🧬 Database: Thanks for chatting with me! I'll keep processing research papers.")
        print("Come back anytime to see what new correlations I've discovered! 👋")


def main():
    """Main entry point."""
    try:
        chatbot = DatabaseChatbot()

        # Check if running non-interactively (for testing)
        if len(sys.argv) > 1 and sys.argv[1] == '--test':
            print("Running in test mode...")
            chatbot.show_greeting()
            print("\n🧬 Database: " + chatbot.process_input("status"))
            print("\n🧬 Database: " + chatbot.process_input("help"))
            print("\n👋 Test completed!")
            return

        chatbot.start()
    except Exception as e:
        print(f"❌ Failed to start database chat: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()