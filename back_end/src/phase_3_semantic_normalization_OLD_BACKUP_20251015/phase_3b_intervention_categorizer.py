"""
Group-Based Categorizer

Categorizes canonical groups instead of individual interventions.
Provides semantic context by including group members in the categorization prompt.
"""

import json
import logging
import sqlite3
import time
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from openai import OpenAI

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class GroupBasedCategorizer:
    """
    Categorizes canonical groups using LLM with semantic context.
    """

    # 13-category taxonomy (from InterventionType enum)
    CATEGORIES = [
        'exercise', 'diet', 'supplement', 'medication', 'therapy',
        'lifestyle', 'surgery', 'test', 'device', 'procedure',
        'biologics', 'gene_therapy', 'emerging'
    ]

    # Category descriptions (from rotation_llm_categorization.py)
    CATEGORY_DESCRIPTIONS = {
        'exercise': 'Physical exercise interventions (aerobic, resistance training, yoga, walking)',
        'diet': 'Dietary interventions (Mediterranean diet, ketogenic diet, intermittent fasting)',
        'supplement': 'Nutritional supplements taken orally (vitamins, minerals, probiotics, herbs, omega-3)',
        'medication': 'Small molecule pharmaceutical drugs (statins, metformin, antibiotics, antidepressants)',
        'therapy': 'Psychological/physical/behavioral therapies (CBT, physical therapy, massage, acupuncture)',
        'lifestyle': 'Behavioral changes (sleep hygiene, stress management, smoking cessation)',
        'surgery': 'Surgical procedures requiring incisions (bariatric surgery, cardiac surgery, transplant operations)',
        'test': 'Medical tests and diagnostics (blood tests, imaging, genetic testing, colonoscopy for diagnosis)',
        'device': 'Medical devices and implants (pacemakers, insulin pumps, CPAP, hearing aids)',
        'procedure': 'Non-surgical medical procedures (endoscopy, dialysis, blood transfusion, fecal transplant, radiation therapy, PRP injection)',
        'biologics': 'Biological drugs from living organisms (monoclonal antibodies, vaccines, immunotherapies, insulin)',
        'gene_therapy': 'Genetic and cellular interventions (CRISPR, CAR-T cell therapy, stem cell therapy)',
        'emerging': 'Novel interventions that don\'t fit existing categories'
    }

    def __init__(
        self,
        db_path: str,
        batch_size: int = 20,
        max_retries: int = 3,
        retry_delay: float = 2.0,
        include_members: bool = True,
        max_members_in_prompt: int = 10
    ):
        """
        Initialize the group-based categorizer.

        Args:
            db_path: Path to SQLite database
            batch_size: Number of groups per LLM call
            max_retries: Maximum retry attempts
            retry_delay: Initial delay between retries (exponential backoff)
            include_members: Include member names in prompt for context
            max_members_in_prompt: Maximum members to include in prompt
        """
        self.db_path = db_path
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.include_members = include_members
        self.max_members_in_prompt = max_members_in_prompt

        # Initialize LLM client
        from back_end.src.data.config import config
        self.client = OpenAI(
            base_url=config.llm_base_url,
            api_key="not-needed"
        )
        self.model = config.llm_model

        # Stats
        self.stats = {
            'total_groups': 0,
            'processed_groups': 0,
            'failed_groups': 0,
            'llm_calls': 0,
            'cache_hits': 0
        }

        logger.info(f"GroupBasedCategorizer initialized with batch_size={batch_size}")

    def get_canonical_groups(self) -> List[Dict]:
        """
        Load canonical groups from database.

        Returns:
            List of dicts with group info: {id, canonical_name, member_count, ...}
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Get groups that need categorization (layer_0_category is NULL)
        query = """
        SELECT
            id,
            canonical_name,
            entity_type,
            member_count,
            total_paper_count,
            layer_0_category
        FROM canonical_groups
        WHERE entity_type = 'intervention'
        AND (layer_0_category IS NULL OR layer_0_category = '')
        ORDER BY member_count DESC
        """

        cursor.execute(query)
        groups = [dict(row) for row in cursor.fetchall()]
        conn.close()

        logger.info(f"Found {len(groups)} canonical groups needing categorization")
        return groups

    def get_group_members(self, canonical_name: str, limit: Optional[int] = None) -> List[str]:
        """
        Get member intervention names for a canonical group.

        Args:
            canonical_name: Canonical group name
            limit: Optional limit on number of members to return

        Returns:
            List of member intervention names
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        query = """
        SELECT DISTINCT entity_name
        FROM semantic_hierarchy
        WHERE layer_1_canonical = ?
        AND entity_type = 'intervention'
        ORDER BY occurrence_count DESC
        """

        if limit:
            query += f" LIMIT {limit}"

        cursor.execute(query, (canonical_name,))
        members = [row[0] for row in cursor.fetchall()]
        conn.close()

        return members

    def categorize_group_batch(self, groups: List[Dict]) -> Dict[str, str]:
        """
        Categorize a batch of groups using LLM.

        Args:
            groups: List of group dicts

        Returns:
            Dict mapping canonical_name -> category
        """
        # Build prompt with group names and members
        group_list = []
        for i, group in enumerate(groups):
            canonical_name = group['canonical_name']

            if self.include_members:
                members = self.get_group_members(canonical_name, self.max_members_in_prompt)
                member_str = ', '.join(members[:self.max_members_in_prompt])
                if len(members) > self.max_members_in_prompt:
                    member_str += f" (+ {len(members) - self.max_members_in_prompt} more)"
                group_list.append(f"{i+1}. {canonical_name} (members: {member_str})")
            else:
                group_list.append(f"{i+1}. {canonical_name}")

        groups_text = "\n".join(group_list)

        # Build category descriptions
        category_desc = "\n".join([
            f"- {cat}: {desc}"
            for cat, desc in self.CATEGORY_DESCRIPTIONS.items()
        ])

        prompt = f"""Classify each intervention GROUP into ONE category from the list below.

CATEGORY DEFINITIONS:
{category_desc}

KEY DISTINCTIONS:
- Blood transfusion, fecal microbiota transplant, platelet injections → procedure (NOT medication/supplement/biologics)
- Probiotics in pill form → supplement; fecal transplant → procedure
- Insulin, vaccines, monoclonal antibodies → biologics (NOT medication)
- Small molecule drugs → medication; biological drugs → biologics

GROUPS TO CLASSIFY:
{groups_text}

IMPORTANT: Consider the group name AND its member interventions to determine the most appropriate category.
For example, a group named "probiotics" with members like "L. reuteri", "S. boulardii" should be classified as "supplement".

Return ONLY JSON array:
[
    {{"number": 1, "category": "supplement"}},
    {{"number": 2, "category": "medication"}},
    ...
]

No explanations. Just the JSON array."""

        # Call LLM with retry logic
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "Provide only the final JSON output without showing your reasoning process or using <think> tags. Start your response immediately with the [ character."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,
                    max_tokens=2000
                )

                response_text = response.choices[0].message.content.strip()

                # Strip think tags if present (qwen3:14b optimization)
                response_text = self._strip_think_tags(response_text)

                # Parse JSON response
                results = self._parse_response(response_text)

                # Map number -> category to canonical_name -> category
                category_map = {}
                for result in results:
                    number = result.get("number")
                    category = result.get("category")

                    if number and category and 1 <= number <= len(groups):
                        canonical_name = groups[number - 1]["canonical_name"]

                        # Validate category
                        if category.lower() in self.CATEGORIES:
                            category_map[canonical_name] = category.lower()
                        else:
                            logger.warning(f"Invalid category '{category}' for group '{canonical_name}'")

                self.stats['llm_calls'] += 1
                logger.info(f"Categorized batch of {len(groups)} groups ({len(category_map)} successful)")
                return category_map

            except Exception as e:
                if attempt < self.max_retries - 1:
                    delay = self.retry_delay * (2 ** attempt)
                    logger.warning(f"Batch categorization failed (attempt {attempt + 1}/{self.max_retries}): {e}")
                    logger.info(f"Retrying in {delay:.1f} seconds...")
                    time.sleep(delay)
                else:
                    logger.error(f"Batch categorization failed after {self.max_retries} attempts: {e}")
                    raise

    def categorize_all_groups(self) -> Dict[str, int]:
        """
        Categorize all canonical groups.

        Returns:
            Statistics dict
        """
        logger.info("Starting group-based categorization")

        groups = self.get_canonical_groups()

        if not groups:
            logger.info("No groups need categorization")
            return {'total': 0, 'processed': 0, 'failed': 0}

        self.stats['total_groups'] = len(groups)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Process in batches
        for i in range(0, len(groups), self.batch_size):
            batch = groups[i:i + self.batch_size]
            batch_num = (i // self.batch_size) + 1
            total_batches = (len(groups) + self.batch_size - 1) // self.batch_size

            logger.info(f"Processing group batch {batch_num}/{total_batches} ({len(batch)} groups)")

            try:
                categories = self.categorize_group_batch(batch)

                # Update database
                for canonical_name, category in categories.items():
                    cursor.execute(
                        "UPDATE canonical_groups SET layer_0_category = ? WHERE canonical_name = ? AND entity_type = 'intervention'",
                        (category, canonical_name)
                    )
                    self.stats['processed_groups'] += 1

                conn.commit()
                logger.info(f"Batch {batch_num} complete: {len(categories)} groups categorized")

            except Exception as e:
                logger.error(f"Batch {batch_num} failed: {e}")
                self.stats['failed_groups'] += len(batch)

        conn.close()

        result = {
            'total': self.stats['total_groups'],
            'processed': self.stats['processed_groups'],
            'failed': self.stats['failed_groups'],
            'llm_calls': self.stats['llm_calls']
        }

        logger.info(f"Group categorization complete: {self.stats['processed_groups']}/{self.stats['total_groups']} successful, {self.stats['llm_calls']} LLM calls")
        return result

    def propagate_to_interventions(self) -> Dict[str, int]:
        """
        Propagate categories from groups to interventions.

        Updates BOTH legacy column (intervention_category) AND junction table (intervention_category_mapping).

        Returns:
            Statistics dict
        """
        logger.info("Propagating categories from groups to interventions")

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Get intervention-category mappings from canonical groups
        cursor.execute("""
            SELECT DISTINCT
                i.id as intervention_id,
                i.intervention_name,
                cg.layer_0_category as category
            FROM interventions i
            JOIN semantic_hierarchy sh ON sh.entity_name = i.intervention_name
            JOIN canonical_groups cg ON cg.canonical_name = sh.layer_1_canonical
            WHERE sh.entity_type = 'intervention'
            AND cg.entity_type = 'intervention'
            AND cg.layer_0_category IS NOT NULL
            AND cg.layer_0_category != ''
        """)

        mappings = cursor.fetchall()
        updated_count = 0
        junction_count = 0

        for intervention_id, intervention_name, category in mappings:
            # Update legacy column (backward compatibility)
            cursor.execute("""
                UPDATE interventions
                SET intervention_category = ?
                WHERE id = ?
                AND (intervention_category IS NULL OR intervention_category = '')
            """, (category, intervention_id))

            if cursor.rowcount > 0:
                updated_count += 1

            # Insert into junction table (multi-category support)
            cursor.execute("""
                INSERT OR IGNORE INTO intervention_category_mapping
                (intervention_id, category_type, category_name, confidence, assigned_by, notes)
                VALUES (?, 'primary', ?, 1.0, 'group_propagation', 'Propagated from canonical group')
            """, (intervention_id, category))

            if cursor.rowcount > 0:
                junction_count += 1

        conn.commit()

        # Count remaining uncategorized (orphans)
        cursor.execute("""
            SELECT COUNT(DISTINCT intervention_name)
            FROM interventions
            WHERE intervention_category IS NULL OR intervention_category = ''
        """)
        orphan_count = cursor.fetchone()[0]

        conn.close()

        logger.info(f"Propagated categories: {updated_count} interventions (legacy column), {junction_count} entries (junction table)")
        logger.info(f"Remaining orphans: {orphan_count}")

        return {
            'updated': updated_count,
            'junction_entries': junction_count,
            'orphans': orphan_count
        }

    def categorize_orphan_interventions(self) -> Dict[str, int]:
        """
        Fallback categorization for interventions not in any canonical group.

        Returns:
            Statistics dict
        """
        logger.info("Categorizing orphan interventions (fallback)")

        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Find orphan interventions (not in semantic_hierarchy)
        query = """
        SELECT DISTINCT intervention_name
        FROM interventions
        WHERE (intervention_category IS NULL OR intervention_category = '')
        AND intervention_name NOT IN (
            SELECT entity_name
            FROM semantic_hierarchy
            WHERE entity_type = 'intervention'
        )
        ORDER BY intervention_name
        """

        cursor.execute(query)
        orphans = [row[0] for row in cursor.fetchall()]

        if not orphans:
            logger.info("No orphan interventions to categorize")
            return {'total': 0, 'processed': 0, 'failed': 0}

        logger.info(f"Found {len(orphans)} orphan interventions")

        # Use individual categorization (similar to Phase 2.5)
        success = 0
        failed = 0

        for i in range(0, len(orphans), self.batch_size):
            batch = orphans[i:i + self.batch_size]
            batch_items = [{'id': None, 'name': name} for name in batch]

            try:
                categories = self._categorize_individual_batch(batch_items)

                for name in batch:
                    category = categories.get(name)
                    if category:
                        # Update legacy column (backward compatibility)
                        cursor.execute(
                            "UPDATE interventions SET intervention_category = ? WHERE intervention_name = ?",
                            (category, name)
                        )

                        # Insert into junction table (multi-category support)
                        cursor.execute("""
                            INSERT OR IGNORE INTO intervention_category_mapping
                            (intervention_id, category_type, category_name, confidence, assigned_by, notes)
                            SELECT id, 'primary', ?, 1.0, 'orphan_fallback', 'Fallback categorization for orphan intervention'
                            FROM interventions
                            WHERE intervention_name = ?
                        """, (category, name))

                        success += 1
                    else:
                        failed += 1

                conn.commit()

            except Exception as e:
                logger.error(f"Failed to categorize orphan batch: {e}")
                failed += len(batch)

        conn.close()

        logger.info(f"Orphan categorization complete: {success}/{len(orphans)} successful")

        return {
            'total': len(orphans),
            'processed': success,
            'failed': failed
        }

    def _categorize_individual_batch(self, batch: List[Dict]) -> Dict[str, str]:
        """
        Categorize individual interventions (fallback for orphans).

        Uses same logic as rotation_llm_categorization.py.
        """
        intervention_list = "\n".join([
            f"{i+1}. {item['name']}"
            for i, item in enumerate(batch)
        ])

        category_desc = "\n".join([
            f"- {cat}: {desc}"
            for cat, desc in self.CATEGORY_DESCRIPTIONS.items()
        ])

        prompt = f"""Classify each health intervention into ONE category from the list below.

CATEGORY DEFINITIONS:
{category_desc}

INTERVENTIONS TO CLASSIFY:
{intervention_list}

Return ONLY JSON array:
[
    {{"number": 1, "category": "supplement"}},
    {{"number": 2, "category": "medication"}},
    ...
]

No explanations. Just the JSON array."""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "Provide only the final JSON output without showing your reasoning process or using <think> tags. Start your response immediately with the [ character."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=2000
        )

        response_text = response.choices[0].message.content.strip()
        response_text = self._strip_think_tags(response_text)
        results = self._parse_response(response_text)

        category_map = {}
        for result in results:
            number = result.get("number")
            category = result.get("category")

            if number and category and 1 <= number <= len(batch):
                name = batch[number - 1]["name"]
                if category.lower() in self.CATEGORIES:
                    category_map[name] = category.lower()

        self.stats['llm_calls'] += 1
        return category_map

    def _strip_think_tags(self, text: str) -> str:
        """Remove <think>...</think> tags from LLM response (qwen3:14b optimization)."""
        import re
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        return text.strip()

    def _parse_response(self, response_text: str) -> List[Dict]:
        """Parse LLM JSON response."""
        # Remove markdown code blocks if present
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0]
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0]

        response_text = response_text.strip()

        try:
            results = json.loads(response_text)
            if not isinstance(results, list):
                raise ValueError("Response is not a JSON array")
            return results
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON: {e}\nResponse: {response_text[:500]}")
            raise


if __name__ == "__main__":
    # Test the categorizer
    from back_end.src.data.config import config

    categorizer = GroupBasedCategorizer(
        db_path=config.db_path,
        batch_size=20
    )

    # Test group categorization
    stats = categorizer.categorize_all_groups()
    print(f"\nGroup categorization stats: {stats}")

    # Propagate to interventions
    propagate_stats = categorizer.propagate_to_interventions()
    print(f"\nPropagation stats: {propagate_stats}")

    # Handle orphans
    orphan_stats = categorizer.categorize_orphan_interventions()
    print(f"\nOrphan categorization stats: {orphan_stats}")
