"""
LLM-based categorization for interventions and conditions.
Separate categorization phase that runs after extraction.
"""

import json
import logging
import time
from typing import List, Dict, Optional, Tuple
from openai import OpenAI

from back_end.src.data.config import config, setup_logging
from back_end.src.phase_1_data_collection.database_manager import database_manager
from back_end.src.interventions.taxonomy import InterventionType
from back_end.src.conditions.taxonomy import ConditionType

logger = setup_logging(__name__)


class LLMCategorizationError(Exception):
    """Errors during LLM categorization."""
    pass


class RotationLLMCategorizer:
    """Categorizes interventions and conditions using LLM."""

    def __init__(self, batch_size: int = 20, max_retries: int = 3, retry_delay: float = 2.0):
        """
        Initialize categorizer.

        Args:
            batch_size: Number of items to categorize per LLM call
            max_retries: Maximum number of retry attempts for failed LLM calls
            retry_delay: Initial delay between retries in seconds (exponential backoff)
        """
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.client = OpenAI(
            base_url=config.llm_base_url,
            api_key="not-needed"
        )
        self.model = config.llm_model

    def categorize_interventions(self, limit: Optional[int] = None) -> Dict[str, int]:
        """
        Categorize all interventions missing intervention_category.

        Args:
            limit: Optional limit on number of interventions to process

        Returns:
            Dictionary with processing statistics
        """
        logger.info("Starting intervention categorization")

        # Get interventions needing categorization
        with database_manager.get_connection() as conn:
            cursor = conn.cursor()
            query = """
                SELECT id, intervention_name
                FROM interventions
                WHERE intervention_category IS NULL
                ORDER BY id
            """
            if limit:
                query += f" LIMIT {limit}"

            cursor.execute(query)
            interventions = [
                {"id": row[0], "name": row[1]}
                for row in cursor.fetchall()
            ]

        if not interventions:
            logger.info("No interventions need categorization")
            return {"total": 0, "processed": 0, "success": 0, "failed": 0}

        logger.info(f"Found {len(interventions)} interventions to categorize")

        # Process in batches
        total = len(interventions)
        success = 0
        failed = 0

        for i in range(0, total, self.batch_size):
            batch = interventions[i:i + self.batch_size]
            batch_num = (i // self.batch_size) + 1
            total_batches = (total + self.batch_size - 1) // self.batch_size

            logger.info(f"Processing intervention batch {batch_num}/{total_batches} ({len(batch)} items)")

            try:
                categories = self._categorize_intervention_batch_with_retry(batch)

                # Update database
                with database_manager.get_connection() as conn:
                    cursor = conn.cursor()
                    for item in batch:
                        category = categories.get(item["id"])
                        if category:
                            cursor.execute(
                                "UPDATE interventions SET intervention_category = ? WHERE id = ?",
                                (category, item["id"])
                            )
                            success += 1
                        else:
                            failed += 1
                    conn.commit()

                logger.info(f"Batch {batch_num} complete: {len([c for c in categories.values() if c])} categorized")

            except Exception as e:
                logger.error(f"Batch {batch_num} failed after all retries: {e}")
                failed += len(batch)

        result = {
            "total": total,
            "processed": success + failed,
            "success": success,
            "failed": failed
        }

        logger.info(f"Intervention categorization complete: {success}/{total} successful")
        return result

    def categorize_conditions(self, limit: Optional[int] = None) -> Dict[str, int]:
        """
        Categorize all conditions missing condition_category.

        Args:
            limit: Optional limit on number of conditions to process

        Returns:
            Dictionary with processing statistics
        """
        logger.info("Starting condition categorization")

        # Get unique conditions needing categorization
        with database_manager.get_connection() as conn:
            cursor = conn.cursor()
            query = """
                SELECT DISTINCT health_condition
                FROM interventions
                WHERE condition_category IS NULL AND health_condition IS NOT NULL
                ORDER BY health_condition
            """
            if limit:
                query += f" LIMIT {limit}"

            cursor.execute(query)
            conditions = [row[0] for row in cursor.fetchall()]

        if not conditions:
            logger.info("No conditions need categorization")
            return {"total": 0, "processed": 0, "success": 0, "failed": 0}

        logger.info(f"Found {len(conditions)} unique conditions to categorize")

        # Process in batches
        total = len(conditions)
        success = 0
        failed = 0

        for i in range(0, total, self.batch_size):
            batch = conditions[i:i + self.batch_size]
            batch_num = (i // self.batch_size) + 1
            total_batches = (total + self.batch_size - 1) // self.batch_size

            logger.info(f"Processing condition batch {batch_num}/{total_batches} ({len(batch)} items)")

            try:
                categories = self._categorize_condition_batch_with_retry(batch)

                # Update database for all interventions with these conditions
                with database_manager.get_connection() as conn:
                    cursor = conn.cursor()
                    for condition in batch:
                        category = categories.get(condition)
                        if category:
                            cursor.execute(
                                "UPDATE interventions SET condition_category = ? WHERE health_condition = ?",
                                (category, condition)
                            )
                            success += 1
                        else:
                            failed += 1
                    conn.commit()

                logger.info(f"Batch {batch_num} complete: {len([c for c in categories.values() if c])} categorized")

            except Exception as e:
                logger.error(f"Batch {batch_num} failed after all retries: {e}")
                failed += len(batch)

        result = {
            "total": total,
            "processed": success + failed,
            "success": success,
            "failed": failed
        }

        logger.info(f"Condition categorization complete: {success}/{total} successful")
        return result

    def categorize_all(self, intervention_limit: Optional[int] = None,
                      condition_limit: Optional[int] = None) -> Dict[str, Dict[str, int]]:
        """
        Categorize both interventions and conditions.

        Args:
            intervention_limit: Optional limit on interventions
            condition_limit: Optional limit on conditions

        Returns:
            Dictionary with statistics for both tasks
        """
        logger.info("Starting full categorization (interventions + conditions)")

        intervention_stats = self.categorize_interventions(intervention_limit)
        condition_stats = self.categorize_conditions(condition_limit)

        return {
            "interventions": intervention_stats,
            "conditions": condition_stats
        }

    def _categorize_intervention_batch_with_retry(self, batch: List[Dict]) -> Dict[int, str]:
        """
        Categorize a batch of interventions with retry logic.

        Args:
            batch: List of dicts with 'id' and 'name' keys

        Returns:
            Dictionary mapping intervention_id -> category
        """
        for attempt in range(self.max_retries):
            try:
                return self._categorize_intervention_batch(batch)
            except Exception as e:
                if attempt < self.max_retries - 1:
                    delay = self.retry_delay * (2 ** attempt)
                    logger.warning(f"Intervention batch categorization failed (attempt {attempt + 1}/{self.max_retries}): {e}")
                    logger.info(f"Retrying in {delay:.1f} seconds...")
                    time.sleep(delay)
                else:
                    logger.error(f"Intervention batch categorization failed after {self.max_retries} attempts")
                    raise

    def _categorize_intervention_batch(self, batch: List[Dict]) -> Dict[int, str]:
        """
        Categorize a batch of interventions using LLM.

        Args:
            batch: List of dicts with 'id' and 'name' keys

        Returns:
            Dictionary mapping intervention_id -> category
        """
        # Create prompt with detailed category descriptions
        intervention_list = "\n".join([
            f"{i+1}. {item['name']}"
            for i, item in enumerate(batch)
        ])

        prompt = f"""Classify each health intervention into ONE category from the list below.

CATEGORY DEFINITIONS:
- exercise: Physical exercise interventions (aerobic, resistance training, yoga, walking)
- diet: Dietary interventions (Mediterranean diet, ketogenic diet, intermittent fasting)
- supplement: Nutritional supplements taken orally (vitamins, minerals, probiotics, herbs, omega-3)
- medication: Small molecule pharmaceutical drugs (statins, metformin, antibiotics, antidepressants)
- therapy: Psychological/physical/behavioral therapies (CBT, physical therapy, massage, acupuncture)
- lifestyle: Behavioral changes (sleep hygiene, stress management, smoking cessation)
- surgery: Surgical procedures requiring incisions (bariatric surgery, cardiac surgery, transplant operations)
- test: Medical tests and diagnostics (blood tests, imaging, genetic testing, colonoscopy for diagnosis)
- device: Medical devices and implants (pacemakers, insulin pumps, CPAP, hearing aids)
- procedure: Non-surgical medical procedures and one-time/periodic interventions (endoscopy, dialysis, blood transfusion, fecal transplant, radiation therapy, platelet-rich plasma injection)
- biologics: Biological drugs from living organisms (monoclonal antibodies, vaccines, immunotherapies, insulin)
- gene_therapy: Genetic and cellular interventions (CRISPR, CAR-T cell therapy, stem cell therapy)
- emerging: Novel interventions that don't fit existing categories

KEY DISTINCTIONS:
- Blood transfusion, fecal microbiota transplant, platelet injections → procedure (NOT medication/supplement/biologics)
- Probiotics in pill form → supplement; fecal transplant → procedure
- Insulin, vaccines, monoclonal antibodies → biologics (NOT medication)
- Small molecule drugs → medication; biological drugs → biologics
- Pacemaker implantation surgery → surgery; using pacemaker → device
- Colonoscopy for diagnosis → test; colonoscopy for polyp removal → procedure

INTERVENTIONS TO CLASSIFY:
{intervention_list}

Return ONLY JSON array:
[
    {{"number": 1, "category": "diet"}},
    {{"number": 2, "category": "exercise"}},
    ...
]

No explanations. Just the JSON array."""

        # Call LLM
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

            # Map number -> category to id -> category
            category_map = {}
            for result in results:
                number = result.get("number")
                category = result.get("category")

                if number and category and 1 <= number <= len(batch):
                    intervention_id = batch[number - 1]["id"]

                    # Validate category
                    try:
                        InterventionType(category.lower())
                        category_map[intervention_id] = category.lower()
                    except ValueError:
                        logger.warning(f"Invalid intervention category: {category}")

            return category_map

        except Exception as e:
            logger.error(f"LLM categorization failed: {e}")
            raise LLMCategorizationError(f"Failed to categorize interventions: {e}")

    def _categorize_condition_batch_with_retry(self, batch: List[str]) -> Dict[str, str]:
        """
        Categorize a batch of conditions with retry logic.

        Args:
            batch: List of condition names

        Returns:
            Dictionary mapping condition_name -> category
        """
        for attempt in range(self.max_retries):
            try:
                return self._categorize_condition_batch(batch)
            except Exception as e:
                if attempt < self.max_retries - 1:
                    delay = self.retry_delay * (2 ** attempt)
                    logger.warning(f"Condition batch categorization failed (attempt {attempt + 1}/{self.max_retries}): {e}")
                    logger.info(f"Retrying in {delay:.1f} seconds...")
                    time.sleep(delay)
                else:
                    logger.error(f"Condition batch categorization failed after {self.max_retries} attempts")
                    raise

    def _categorize_condition_batch(self, batch: List[str]) -> Dict[str, str]:
        """
        Categorize a batch of conditions using LLM.

        Args:
            batch: List of condition names

        Returns:
            Dictionary mapping condition_name -> category
        """
        # Create prompt with detailed category descriptions
        condition_list = "\n".join([
            f"{i+1}. {condition}"
            for i, condition in enumerate(batch)
        ])

        prompt = f"""Classify each health condition into ONE category from the list below.

CATEGORY DEFINITIONS:
- cardiac: Heart and blood vessel conditions (coronary artery disease, heart failure, hypertension, arrhythmias, MI)
- neurological: Brain, spinal cord, nervous system (stroke, Alzheimer's, Parkinson's, epilepsy, MS, dementia, neuropathy)
- digestive: Gastrointestinal system (GERD, IBD, IBS, cirrhosis, Crohn's, ulcerative colitis, H. pylori)
- pulmonary: Lungs and respiratory system (COPD, asthma, pneumonia, pulmonary embolism, respiratory failure)
- endocrine: Hormones and metabolism (diabetes, thyroid disorders, obesity, PCOS, metabolic syndrome)
- renal: Kidneys and urinary system (chronic kidney disease, acute kidney injury, kidney stones, glomerulonephritis)
- oncological: All cancers and malignant neoplasms (lung cancer, breast cancer, colorectal cancer, leukemia)
- rheumatological: Autoimmune and rheumatic diseases (rheumatoid arthritis, lupus, gout, vasculitis, fibromyalgia)
- psychiatric: Mental health conditions (depression, anxiety, bipolar disorder, schizophrenia, ADHD, PTSD)
- musculoskeletal: Bones, muscles, tendons, ligaments (fractures, osteoarthritis, back pain, ACL injury, tendinitis)
- dermatological: Skin, hair, nails (acne, psoriasis, eczema, atopic dermatitis, melanoma, rosacea)
- infectious: Bacterial, viral, fungal infections (HIV, tuberculosis, hepatitis, sepsis, COVID-19, influenza)
- immunological: Allergies and immune disorders (food allergies, allergic rhinitis, immunodeficiency, anaphylaxis)
- hematological: Blood cells and clotting (anemia, thrombocytopenia, hemophilia, sickle cell disease, thrombosis)
- nutritional: Nutrient deficiencies (vitamin D deficiency, B12 deficiency, iron deficiency, malnutrition)
- toxicological: Poisoning and drug toxicity (drug toxicity, heavy metal poisoning, overdose, carbon monoxide poisoning)
- parasitic: Parasitic infections (malaria, toxoplasmosis, giardiasis, schistosomiasis, helminth infections)
- other: Conditions that don't fit standard categories or are multisystem

KEY DISTINCTIONS:
- Type 2 diabetes, diabetic neuropathy, PCOS → endocrine (metabolic/hormonal)
- Diabetic foot ulcer, foot complications → infectious or dermatological (depending on context)
- Osteoarthritis in rheumatological context (autoimmune) → rheumatological
- Osteoarthritis as mechanical wear → musculoskeletal
- Heart failure, hypertension, atrial fibrillation → cardiac
- Depression, anxiety, bipolar disorder → psychiatric
- H. pylori infection, sepsis → infectious (NOT digestive/cardiac - it's the infection itself)

CONDITIONS TO CLASSIFY:
{condition_list}

Return ONLY JSON array:
[
    {{"number": 1, "category": "endocrine"}},
    {{"number": 2, "category": "cardiac"}},
    ...
]

No explanations. Just the JSON array."""

        # Call LLM
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

            # Map number -> category to condition_name -> category
            category_map = {}
            for result in results:
                number = result.get("number")
                category = result.get("category")

                if number and category and 1 <= number <= len(batch):
                    condition_name = batch[number - 1]

                    # Validate category
                    try:
                        ConditionType(category.lower())
                        category_map[condition_name] = category.lower()
                    except ValueError:
                        logger.warning(f"Invalid condition category: {category}")

            return category_map

        except Exception as e:
            logger.error(f"LLM categorization failed: {e}")
            raise LLMCategorizationError(f"Failed to categorize conditions: {e}")

    def _strip_think_tags(self, text: str) -> str:
        """
        Remove <think>...</think> tags from LLM response.
        qwen3:14b optimization to suppress chain-of-thought reasoning.

        Args:
            text: Raw LLM response

        Returns:
            Text with think tags removed
        """
        import re
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        return text.strip()

    def _parse_categorization_response(self, response_text: str) -> List[Dict]:
        """
        Parse LLM response into structured data.

        Args:
            response_text: Raw LLM response

        Returns:
            List of dicts with 'number' and 'category' keys
        """
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
            raise LLMCategorizationError(f"Invalid JSON response: {e}")


# CLI for standalone execution
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Categorize interventions and conditions using LLM")
    parser.add_argument("--interventions-only", action="store_true",
                       help="Only categorize interventions")
    parser.add_argument("--conditions-only", action="store_true",
                       help="Only categorize conditions")
    parser.add_argument("--intervention-limit", type=int,
                       help="Limit number of interventions to process")
    parser.add_argument("--condition-limit", type=int,
                       help="Limit number of conditions to process")
    parser.add_argument("--batch-size", type=int, default=20,
                       help="Batch size for LLM calls (default: 20)")
    parser.add_argument("--max-retries", type=int, default=3,
                       help="Maximum retry attempts for failed LLM calls (default: 3)")
    parser.add_argument("--retry-delay", type=float, default=2.0,
                       help="Initial retry delay in seconds, uses exponential backoff (default: 2.0)")

    args = parser.parse_args()

    categorizer = RotationLLMCategorizer(
        batch_size=args.batch_size,
        max_retries=args.max_retries,
        retry_delay=args.retry_delay
    )

    if args.interventions_only:
        stats = categorizer.categorize_interventions(args.intervention_limit)
        print(f"\nIntervention Categorization Results:")
        print(f"  Total: {stats['total']}")
        print(f"  Success: {stats['success']}")
        print(f"  Failed: {stats['failed']}")
    elif args.conditions_only:
        stats = categorizer.categorize_conditions(args.condition_limit)
        print(f"\nCondition Categorization Results:")
        print(f"  Total: {stats['total']}")
        print(f"  Success: {stats['success']}")
        print(f"  Failed: {stats['failed']}")
    else:
        stats = categorizer.categorize_all(args.intervention_limit, args.condition_limit)
        print(f"\nIntervention Categorization Results:")
        print(f"  Total: {stats['interventions']['total']}")
        print(f"  Success: {stats['interventions']['success']}")
        print(f"  Failed: {stats['interventions']['failed']}")
        print(f"\nCondition Categorization Results:")
        print(f"  Total: {stats['conditions']['total']}")
        print(f"  Success: {stats['conditions']['success']}")
        print(f"  Failed: {stats['conditions']['failed']}")
