"""
Condition Group-Based Categorizer

Categorizes canonical condition groups instead of individual conditions.
Provides semantic context by including group members in the categorization prompt.

Uses 18-category condition taxonomy from ConditionType enum.
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


class ConditionGroupBasedCategorizer:
    """
    Categorizes canonical condition groups using LLM with semantic context.
    """

    # 18-category condition taxonomy (from ConditionType enum)
    CATEGORIES = [
        'cardiac', 'neurological', 'digestive', 'pulmonary', 'endocrine', 'renal',
        'oncological', 'rheumatological', 'psychiatric', 'musculoskeletal',
        'dermatological', 'infectious', 'immunological', 'hematological',
        'nutritional', 'toxicological', 'parasitic', 'other'
    ]

    # Category descriptions (from ConditionTaxonomy in taxonomy.py)
    CATEGORY_DESCRIPTIONS = {
        'cardiac': 'Heart and blood vessel conditions (coronary artery disease, heart failure, hypertension, arrhythmias, MI)',
        'neurological': 'Brain, spinal cord, nervous system (stroke, Alzheimer\'s, Parkinson\'s, epilepsy, MS, dementia, neuropathy)',
        'digestive': 'Gastrointestinal system (GERD, IBD, IBS, cirrhosis, Crohn\'s, ulcerative colitis, H. pylori)',
        'pulmonary': 'Lungs and respiratory system (COPD, asthma, pneumonia, pulmonary embolism, respiratory failure)',
        'endocrine': 'Hormones and metabolism (diabetes, thyroid disorders, obesity, PCOS, metabolic syndrome)',
        'renal': 'Kidneys and urinary system (chronic kidney disease, acute kidney injury, kidney stones, glomerulonephritis)',
        'oncological': 'All cancers and malignant neoplasms (lung cancer, breast cancer, colorectal cancer, leukemia)',
        'rheumatological': 'Autoimmune and rheumatic diseases (rheumatoid arthritis, lupus, gout, vasculitis, fibromyalgia)',
        'psychiatric': 'Mental health conditions (depression, anxiety, bipolar disorder, schizophrenia, ADHD, PTSD)',
        'musculoskeletal': 'Bones, muscles, tendons, ligaments (fractures, osteoarthritis, back pain, ACL injury, tendinitis)',
        'dermatological': 'Skin, hair, nails (acne, psoriasis, eczema, atopic dermatitis, melanoma, rosacea)',
        'infectious': 'Bacterial, viral, fungal infections (HIV, tuberculosis, hepatitis, sepsis, COVID-19, influenza)',
        'immunological': 'Allergies and immune disorders (food allergies, allergic rhinitis, immunodeficiency, anaphylaxis)',
        'hematological': 'Blood cells and clotting (anemia, thrombocytopenia, hemophilia, sickle cell disease, thrombosis)',
        'nutritional': 'Nutrient deficiencies (vitamin D deficiency, B12 deficiency, iron deficiency, malnutrition)',
        'toxicological': 'Poisoning and drug toxicity (drug toxicity, heavy metal poisoning, overdose, carbon monoxide poisoning)',
        'parasitic': 'Parasitic infections (malaria, toxoplasmosis, giardiasis, schistosomiasis, helminth infections)',
        'other': 'Conditions that don\'t fit standard categories or are multisystem'
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
        Initialize the condition group-based categorizer.

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

        logger.info(f"ConditionGroupBasedCategorizer initialized with batch_size={batch_size}")

    def get_canonical_groups(self) -> List[Dict]:
        """
        Load canonical condition groups from database.

        Returns:
            List of dicts with group info: {id, canonical_name, member_count, ...}
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Get condition groups that need categorization (layer_0_category is NULL)
        query = """
        SELECT
            id,
            canonical_name,
            entity_type,
            member_count,
            total_paper_count,
            layer_0_category
        FROM canonical_groups
        WHERE entity_type = 'condition'
        AND (layer_0_category IS NULL OR layer_0_category = '')
        ORDER BY member_count DESC
        """

        cursor.execute(query)
        groups = [dict(row) for row in cursor.fetchall()]
        conn.close()

        logger.info(f"Found {len(groups)} canonical condition groups needing categorization")
        return groups

    def get_group_members(self, canonical_name: str, limit: Optional[int] = None) -> List[str]:
        """
        Get member condition names for a canonical group.

        Args:
            canonical_name: Canonical group name
            limit: Optional limit on number of members to return

        Returns:
            List of member condition names
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        query = """
        SELECT DISTINCT entity_name
        FROM semantic_hierarchy
        WHERE layer_1_canonical = ?
        AND entity_type = 'condition'
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
        Categorize a batch of condition groups using LLM.

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
                group_list.append(f"{i+1}. {canonical_name} (variants: {member_str})")
            else:
                group_list.append(f"{i+1}. {canonical_name}")

        groups_text = "\n".join(group_list)

        # Build category descriptions
        category_desc = "\n".join([
            f"- {cat}: {desc}"
            for cat, desc in self.CATEGORY_DESCRIPTIONS.items()
        ])

        prompt = f"""Classify each health CONDITION group into ONE category from the list below.

CATEGORY DEFINITIONS:
{category_desc}

KEY DISTINCTIONS:
- Type 2 diabetes, diabetic neuropathy, PCOS → endocrine (metabolic/hormonal)
- Diabetic foot ulcer, foot complications → infectious or dermatological (depending on context)
- Osteoarthritis in rheumatological context (autoimmune) → rheumatological
- Osteoarthritis as mechanical wear → musculoskeletal
- Heart failure, hypertension, atrial fibrillation → cardiac
- Depression, anxiety, bipolar disorder → psychiatric
- H. pylori infection, sepsis → infectious (NOT digestive/cardiac - it's the infection itself)
- IBS, IBD, GERD → digestive
- COPD, asthma → pulmonary

CONDITION GROUPS TO CLASSIFY:
{groups_text}

Return ONLY JSON array:
[
    {{"number": 1, "category": "endocrine"}},
    {{"number": 2, "category": "cardiac"}},
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
                results = self._parse_categorization_response(response_text)

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
                            logger.warning(f"Invalid condition category: {category}")

                self.stats['llm_calls'] += 1
                return category_map

            except Exception as e:
                if attempt < self.max_retries - 1:
                    delay = self.retry_delay * (2 ** attempt)
                    logger.warning(f"Condition group categorization failed (attempt {attempt + 1}/{self.max_retries}): {e}")
                    logger.info(f"Retrying in {delay:.1f} seconds...")
                    time.sleep(delay)
                else:
                    logger.error(f"Condition group categorization failed after {self.max_retries} attempts")
                    raise

        return {}

    def _strip_think_tags(self, text: str) -> str:
        """Remove <think>...</think> tags from LLM response."""
        import re
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        return text.strip()

    def _parse_categorization_response(self, response_text: str) -> List[Dict]:
        """Parse LLM response into structured data."""
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
            logger.error(f"Failed to parse JSON: {e}\nResponse: {response_text}")
            raise

    def categorize_all_groups(self) -> Dict:
        """
        Categorize all canonical condition groups.

        Returns:
            Statistics dict
        """
        groups = self.get_canonical_groups()
        self.stats['total_groups'] = len(groups)

        if not groups:
            logger.info("No condition groups need categorization")
            return self.stats

        logger.info(f"Categorizing {len(groups)} condition groups in batches of {self.batch_size}...")

        # Process in batches
        for i in range(0, len(groups), self.batch_size):
            batch = groups[i:i + self.batch_size]
            batch_num = (i // self.batch_size) + 1
            total_batches = (len(groups) + self.batch_size - 1) // self.batch_size

            logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch)} groups)")

            try:
                categories = self.categorize_group_batch(batch)

                # Update database
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()

                for canonical_name, category in categories.items():
                    cursor.execute(
                        "UPDATE canonical_groups SET layer_0_category = ? WHERE canonical_name = ? AND entity_type = 'condition'",
                        (category, canonical_name)
                    )
                    self.stats['processed_groups'] += 1

                conn.commit()
                conn.close()

                logger.info(f"Batch {batch_num} complete: {len(categories)} groups categorized")

            except Exception as e:
                logger.error(f"Batch {batch_num} failed: {e}")
                self.stats['failed_groups'] += len(batch)

        return self.stats

    def propagate_to_conditions(self) -> Dict:
        """
        Propagate categories from canonical groups to individual conditions.

        Updates BOTH legacy column (condition_category) AND junction table (condition_category_mapping).

        Returns:
            Statistics dict with update counts
        """
        logger.info("Propagating categories from condition groups to conditions...")

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Get condition-category mappings from canonical groups
        cursor.execute("""
            SELECT DISTINCT
                interventions.health_condition,
                cg.layer_0_category as category
            FROM interventions
            JOIN semantic_hierarchy sh ON sh.entity_name = interventions.health_condition
            JOIN canonical_groups cg ON sh.layer_1_canonical = cg.canonical_name
            WHERE sh.entity_type = 'condition'
            AND cg.entity_type = 'condition'
            AND cg.layer_0_category IS NOT NULL
            AND cg.layer_0_category != ''
        """)

        mappings = cursor.fetchall()
        updated_count = 0
        junction_count = 0

        for condition_name, category in mappings:
            # Update legacy column (backward compatibility)
            cursor.execute("""
                UPDATE interventions
                SET condition_category = ?
                WHERE health_condition = ?
                AND (condition_category IS NULL OR condition_category = '')
            """, (category, condition_name))

            if cursor.rowcount > 0:
                updated_count += cursor.rowcount

            # Insert into junction table (multi-category support)
            cursor.execute("""
                INSERT OR IGNORE INTO condition_category_mapping
                (condition_name, category_type, category_name, confidence, assigned_by, notes)
                VALUES (?, 'primary', ?, 1.0, 'group_propagation', 'Propagated from canonical group')
            """, (condition_name, category))

            if cursor.rowcount > 0:
                junction_count += 1

        conn.commit()

        # Count orphan conditions (not in semantic_hierarchy)
        orphan_query = """
        SELECT COUNT(DISTINCT health_condition)
        FROM interventions
        WHERE condition_category IS NULL
        AND health_condition IS NOT NULL
        AND health_condition != ''
        """

        cursor.execute(orphan_query)
        orphan_count = cursor.fetchone()[0]

        conn.close()

        logger.info(f"Propagated categories: {updated_count} interventions (legacy column), {junction_count} entries (junction table)")
        logger.info(f"Found {orphan_count} orphan conditions (not in groups)")

        return {
            'updated': updated_count,
            'junction_entries': junction_count,
            'orphans': orphan_count
        }

    def categorize_orphan_conditions(self) -> Dict:
        """
        Categorize orphan conditions that are not in semantic groups.

        Fallback categorization for conditions without group membership.

        Returns:
            Statistics dict
        """
        logger.info("Categorizing orphan conditions...")

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Get orphan conditions
        orphan_query = """
        SELECT DISTINCT health_condition
        FROM interventions
        WHERE condition_category IS NULL
        AND health_condition IS NOT NULL
        AND health_condition != ''
        AND health_condition NOT IN (
            SELECT entity_name
            FROM semantic_hierarchy
            WHERE entity_type = 'condition'
        )
        """

        cursor.execute(orphan_query)
        orphans = [row[0] for row in cursor.fetchall()]
        conn.close()

        if not orphans:
            logger.info("No orphan conditions to categorize")
            return {'total': 0, 'processed': 0, 'failed': 0}

        logger.info(f"Found {len(orphans)} orphan conditions")

        # Categorize orphans in batches
        processed = 0
        failed = 0

        for i in range(0, len(orphans), self.batch_size):
            batch = orphans[i:i + self.batch_size]

            try:
                # Create fake groups for orphans
                fake_groups = [
                    {'canonical_name': condition, 'entity_type': 'condition'}
                    for condition in batch
                ]

                categories = self.categorize_group_batch(fake_groups)

                # Update database directly (no group membership)
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()

                for condition, category in categories.items():
                    # Update legacy column (backward compatibility)
                    cursor.execute(
                        "UPDATE interventions SET condition_category = ? WHERE health_condition = ?",
                        (category, condition)
                    )

                    # Insert into junction table (multi-category support)
                    cursor.execute("""
                        INSERT OR IGNORE INTO condition_category_mapping
                        (condition_name, category_type, category_name, confidence, assigned_by, notes)
                        VALUES (?, 'primary', ?, 1.0, 'orphan_fallback', 'Fallback categorization for orphan condition')
                    """, (condition, category))

                    processed += 1

                conn.commit()
                conn.close()

            except Exception as e:
                logger.error(f"Failed to categorize orphan batch: {e}")
                failed += len(batch)

        return {
            'total': len(orphans),
            'processed': processed,
            'failed': failed,
            'llm_calls': self.stats['llm_calls']
        }
