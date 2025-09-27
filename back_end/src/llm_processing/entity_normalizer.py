"""
Entity Normalization System

This module provides a simple database-backed entity normalization system
that works alongside existing code without interference.

Classes:
    EntityNormalizer: Core class for managing entity normalization operations
"""

import sqlite3
import json
import hashlib
from typing import Optional, List, Union, Dict, Any
from datetime import datetime

# Import existing LLM setup
try:
    import sys
    import os
    # Add parent directories to path to import from data module
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from data.api_clients import get_llm_client
    from data.utils import parse_json_safely
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    print("Warning: LLM functionality not available. Install required dependencies.")


class EntityNormalizer:
    """
    A minimal entity normalizer that manages canonical entities and their mappings.

    This class provides database operations for entity normalization without
    interfering with existing code. It operates on the canonical_entities
    and entity_mappings tables.
    """

    def __init__(self, db_connection: sqlite3.Connection, llm_model: str = "gemma2:9b"):
        """
        Initialize the EntityNormalizer with a database connection.

        Args:
            db_connection: SQLite database connection object
            llm_model: LLM model name for semantic matching
        """
        self.db = db_connection
        self.db.row_factory = sqlite3.Row  # Enable dict-like access to rows
        self.llm_model = llm_model

        # Initialize LLM client if available
        if LLM_AVAILABLE:
            self.llm_client = get_llm_client(llm_model)
        else:
            self.llm_client = None

    def find_canonical_id(self, term: str, entity_type: str) -> Optional[int]:
        """
        Find the canonical ID for a term if it's already mapped.

        Args:
            term: The raw text term to look up
            entity_type: Either 'intervention' or 'condition'

        Returns:
            The canonical_id if term is mapped, None otherwise
        """
        cursor = self.db.cursor()
        cursor.execute("""
            SELECT canonical_id
            FROM entity_mappings
            WHERE raw_text = ? AND entity_type = ?
        """, (term, entity_type))

        result = cursor.fetchone()
        return result['canonical_id'] if result else None

    def create_canonical_entity(self, canonical_name: str, entity_type: str,
                              scientific_name: Optional[str] = None) -> int:
        """
        Create a new canonical entity.

        Args:
            canonical_name: The canonical/normalized name
            entity_type: Either 'intervention' or 'condition'
            scientific_name: Optional scientific name for additional metadata

        Returns:
            The ID of the newly created canonical entity

        Raises:
            sqlite3.IntegrityError: If canonical_name already exists
        """
        cursor = self.db.cursor()

        # Prepare metadata JSON if scientific_name provided
        metadata = {}
        if scientific_name:
            metadata['scientific_name'] = scientific_name

        metadata_json = json.dumps(metadata) if metadata else None

        cursor.execute("""
            INSERT INTO canonical_entities (canonical_name, entity_type, metadata)
            VALUES (?, ?, ?)
        """, (canonical_name, entity_type, metadata_json))

        self.db.commit()
        return cursor.lastrowid

    def add_term_mapping(self, original_term: str, canonical_id: int,
                        confidence: float, method: str,
                        paper_ids: Optional[List[str]] = None) -> int:
        """
        Map a term to a canonical entity.

        Args:
            original_term: The raw text term to map
            canonical_id: ID of the canonical entity to map to
            confidence: Confidence score (0.0 to 1.0)
            method: Mapping method (e.g., 'exact_match', 'fuzzy_match', 'manual')
            paper_ids: Optional list of paper IDs where this term was found

        Returns:
            The ID of the newly created mapping

        Raises:
            sqlite3.IntegrityError: If this exact mapping already exists
        """
        cursor = self.db.cursor()

        # First, get the entity_type from the canonical entity
        cursor.execute("""
            SELECT entity_type FROM canonical_entities WHERE id = ?
        """, (canonical_id,))

        result = cursor.fetchone()
        if not result:
            raise ValueError(f"Canonical entity with ID {canonical_id} not found")

        entity_type = result['entity_type']

        # Create the mapping
        cursor.execute("""
            INSERT INTO entity_mappings
            (canonical_id, raw_text, entity_type, confidence_score, mapping_method)
            VALUES (?, ?, ?, ?, ?)
        """, (canonical_id, original_term, entity_type, confidence, method))

        self.db.commit()
        mapping_id = cursor.lastrowid

        # TODO: In future, could store paper_ids in a separate table for tracking
        # where terms were found, but keeping simple for now

        return mapping_id

    def get_canonical_name(self, term: str, entity_type: str) -> str:
        """
        Get the canonical name for a term, or return the term itself if not mapped.

        Args:
            term: The raw text term to normalize
            entity_type: Either 'intervention' or 'condition'

        Returns:
            The canonical name if term is mapped, otherwise the original term
        """
        cursor = self.db.cursor()
        cursor.execute("""
            SELECT ce.canonical_name
            FROM entity_mappings em
            JOIN canonical_entities ce ON em.canonical_id = ce.id
            WHERE em.raw_text = ? AND em.entity_type = ?
        """, (term, entity_type))

        result = cursor.fetchone()
        return result['canonical_name'] if result else term

    def get_all_mappings_for_canonical(self, canonical_id: int) -> List[dict]:
        """
        Get all term mappings for a canonical entity.

        Args:
            canonical_id: The canonical entity ID

        Returns:
            List of mapping dictionaries with raw_text, confidence_score, etc.
        """
        cursor = self.db.cursor()
        cursor.execute("""
            SELECT raw_text, confidence_score, mapping_method, created_at
            FROM entity_mappings
            WHERE canonical_id = ?
            ORDER BY confidence_score DESC
        """, (canonical_id,))

        return [dict(row) for row in cursor.fetchall()]

    def search_canonical_entities(self, search_term: str, entity_type: Optional[str] = None) -> List[dict]:
        """
        Search for canonical entities by name.

        Args:
            search_term: Term to search for (supports partial matches)
            entity_type: Optional filter by entity type

        Returns:
            List of canonical entity dictionaries
        """
        cursor = self.db.cursor()

        if entity_type:
            cursor.execute("""
                SELECT id, canonical_name, entity_type, metadata, created_at
                FROM canonical_entities
                WHERE canonical_name LIKE ? AND entity_type = ?
                ORDER BY canonical_name
            """, (f"%{search_term}%", entity_type))
        else:
            cursor.execute("""
                SELECT id, canonical_name, entity_type, metadata, created_at
                FROM canonical_entities
                WHERE canonical_name LIKE ?
                ORDER BY canonical_name
            """, (f"%{search_term}%",))

        return [dict(row) for row in cursor.fetchall()]

    def get_mapping_stats(self) -> dict:
        """
        Get statistics about the current mapping state.

        Returns:
            Dictionary with counts of entities, mappings, etc.
        """
        cursor = self.db.cursor()

        # Count canonical entities by type
        cursor.execute("""
            SELECT entity_type, COUNT(*) as count
            FROM canonical_entities
            GROUP BY entity_type
        """)
        entity_counts = {row['entity_type']: row['count'] for row in cursor.fetchall()}

        # Count mappings by type
        cursor.execute("""
            SELECT entity_type, COUNT(*) as count
            FROM entity_mappings
            GROUP BY entity_type
        """)
        mapping_counts = {row['entity_type']: row['count'] for row in cursor.fetchall()}

        # Count unique terms vs mappings ratio
        cursor.execute("""
            SELECT entity_type,
                   COUNT(DISTINCT raw_text) as unique_terms,
                   COUNT(DISTINCT canonical_id) as canonical_entities
            FROM entity_mappings
            GROUP BY entity_type
        """)
        ratio_stats = {}
        for row in cursor.fetchall():
            entity_type = row['entity_type']
            ratio_stats[entity_type] = {
                'unique_terms': row['unique_terms'],
                'canonical_entities': row['canonical_entities'],
                'compression_ratio': row['unique_terms'] / row['canonical_entities'] if row['canonical_entities'] > 0 else 0
            }

        return {
            'canonical_entities': entity_counts,
            'mappings': mapping_counts,
            'ratios': ratio_stats
        }

    # === SMART MATCHING METHODS ===

    def normalize_term(self, term: str) -> str:
        """
        Normalize a term by lowercasing, stripping whitespace, and removing punctuation.

        Args:
            term: The raw text term to normalize

        Returns:
            The normalized term
        """
        import re

        # Convert to lowercase and strip whitespace
        normalized = term.lower().strip()

        # Remove common punctuation but keep meaningful characters like hyphens in compound words
        # Remove parentheses and their contents (like abbreviations)
        normalized = re.sub(r'\([^)]*\)', '', normalized)

        # Remove excess whitespace and common punctuation
        normalized = re.sub(r'[.,;:!?"\']', '', normalized)

        # Normalize multiple spaces to single spaces
        normalized = re.sub(r'\s+', ' ', normalized)

        # Final trim
        return normalized.strip()

    def find_by_exact_match(self, term: str, entity_type: str) -> Optional[dict]:
        """
        Find a canonical entity by exact normalized match.

        Args:
            term: The term to search for
            entity_type: Either 'intervention' or 'condition'

        Returns:
            Dictionary with canonical entity info if found, None otherwise
        """
        normalized_term = self.normalize_term(term)

        cursor = self.db.cursor()

        # Check if the normalized term matches any normalized canonical name
        cursor.execute("""
            SELECT ce.id, ce.canonical_name, ce.entity_type
            FROM canonical_entities ce
            WHERE LOWER(TRIM(ce.canonical_name)) = ? AND ce.entity_type = ?
        """, (normalized_term, entity_type))

        result = cursor.fetchone()
        if result:
            return dict(result)

        # Also check if the normalized term matches any normalized mapping
        cursor.execute("""
            SELECT ce.id, ce.canonical_name, ce.entity_type, em.confidence_score
            FROM entity_mappings em
            JOIN canonical_entities ce ON em.canonical_id = ce.id
            WHERE LOWER(TRIM(em.raw_text)) = ? AND em.entity_type = ?
        """, (normalized_term, entity_type))

        result = cursor.fetchone()
        return dict(result) if result else None

    def find_by_pattern(self, term: str, entity_type: str) -> List[dict]:
        """
        Find canonical entities using SAFE pattern matching for medical terms.

        Only matches TRUE variations of the same medical concept:
        - Case variations (IBS/ibs/Ibs)
        - Safe pluralization (probiotic/probiotics)
        - Punctuation/spacing (multi-strain/multistrain)
        - No similarity matching - too dangerous for medical terms!

        Args:
            term: The term to search for
            entity_type: Either 'intervention' or 'condition'

        Returns:
            List of matching canonical entities with confidence scores
        """
        import re

        normalized_term = self.normalize_term(term)
        matches = []

        cursor = self.db.cursor()

        # Get all canonical entities and mappings for this entity type
        cursor.execute("""
            SELECT DISTINCT ce.id, ce.canonical_name, ce.entity_type,
                   em.raw_text, em.confidence_score
            FROM canonical_entities ce
            LEFT JOIN entity_mappings em ON ce.id = em.canonical_id
            WHERE ce.entity_type = ?
        """, (entity_type,))

        entities = cursor.fetchall()

        for entity in entities:
            canonical_name = self.normalize_term(entity['canonical_name'])
            raw_text = self.normalize_term(entity['raw_text'] or entity['canonical_name'])

            # PATTERN 1: Safe Plural/Singular matching with medical term validation
            # Only apply to terms that are clearly the same concept
            if self._is_safe_pluralization_candidate(normalized_term, canonical_name):

                # Simple 's' addition/removal (most common and safe)
                if normalized_term == canonical_name + 's':
                    matches.append({
                        'id': entity['id'],
                        'canonical_name': entity['canonical_name'],
                        'entity_type': entity['entity_type'],
                        'description': entity['description'] if 'description' in entity.keys() else '',
                        'confidence_score': 0.95,
                        'match_method': 'safe_plural_addition'
                    })
                elif normalized_term + 's' == canonical_name:
                    matches.append({
                        'id': entity['id'],
                        'canonical_name': entity['canonical_name'],
                        'entity_type': entity['entity_type'],
                        'description': entity['description'] if 'description' in entity.keys() else '',
                        'confidence_score': 0.95,
                        'match_method': 'safe_plural_removal'
                    })

            # PATTERN 2: Safe prefix/suffix removal (very conservative)
            # Only remove unambiguous modifiers that don't change meaning
            safe_prefixes = ['the ']  # Reduced to only definite article
            safe_suffixes = []        # Removed all - too risky for medical terms

            # Only remove "the" prefix if it's unambiguous
            if normalized_term.startswith('the ') and len(normalized_term) > 4:
                term_without_the = normalized_term[4:]  # Remove "the "
                if term_without_the == canonical_name:
                    matches.append({
                        'id': entity['id'],
                        'canonical_name': entity['canonical_name'],
                        'entity_type': entity['entity_type'],
                        'description': entity['description'] if 'description' in entity.keys() else '',
                        'confidence_score': 0.9,
                        'match_method': 'definite_article_removal'
                    })

            # PATTERN 3: Punctuation/spacing normalization (very safe)
            # Handle spacing and hyphen differences
            term_spacing_normalized = re.sub(r'[-_\s]+', '', normalized_term)
            canonical_spacing_normalized = re.sub(r'[-_\s]+', '', canonical_name)

            if (term_spacing_normalized == canonical_spacing_normalized and
                term_spacing_normalized != normalized_term):  # Only if there was a change
                matches.append({
                    'id': entity['id'],
                    'canonical_name': entity['canonical_name'],
                    'entity_type': entity['entity_type'],
                    'description': entity['description'] if 'description' in entity.keys() else '',
                    'confidence_score': 0.9,
                    'match_method': 'spacing_punctuation_normalization'
                })

        # Remove duplicates based on canonical ID, keeping highest confidence
        best_matches = {}
        for match in matches:
            entity_id = match['id']
            if entity_id not in best_matches or match['confidence_score'] > best_matches[entity_id]['confidence_score']:
                best_matches[entity_id] = match

        # Sort by confidence score descending
        return sorted(best_matches.values(), key=lambda x: x['confidence_score'], reverse=True)

    def _is_safe_pluralization_candidate(self, term1: str, term2: str) -> bool:
        """
        Check if two terms are safe candidates for pluralization matching.

        Prevents dangerous matches like 'prebiotics'/'probiotics' by checking
        that the terms are very similar except for pluralization.

        Args:
            term1: First term
            term2: Second term

        Returns:
            True if safe to apply pluralization rules
        """
        # Must be at least 4 characters to avoid false positives
        if len(term1) < 4 or len(term2) < 4:
            return False

        # Check if one is exactly the other + 's'
        if term1 == term2 + 's' or term2 == term1 + 's':
            # Additional safety: ensure the root is at least 3 characters
            shorter = min(term1, term2, key=len)
            if len(shorter) >= 3:
                return True

        return False

    def calculate_string_similarity(self, term1: str, term2: str) -> float:
        """
        DEPRECATED: String similarity matching is too dangerous for medical terms!

        This method is kept for backward compatibility but should not be used
        for medical term matching. Use explicit mappings or LLM verification instead.

        Examples of dangerous false positives:
        - probiotics vs prebiotics (different substances)
        - hyperglycemia vs hypoglycemia (opposite conditions)
        - hypertension vs hypotension (opposite conditions)

        Args:
            term1: First term
            term2: Second term

        Returns:
            Similarity score between 0.0 and 1.0
        """
        # Normalize both terms
        norm1 = self.normalize_term(term1)
        norm2 = self.normalize_term(term2)

        if norm1 == norm2:
            return 1.0

        # For medical terms, we should be extremely conservative
        # Only return high similarity for very minor differences

        # Check for safe transformations only
        if self._is_safe_pluralization_candidate(norm1, norm2):
            return 0.98  # High but not perfect to indicate pattern match is better

        # Check for spacing/punctuation differences only
        import re
        norm1_no_spaces = re.sub(r'[-_\s]+', '', norm1)
        norm2_no_spaces = re.sub(r'[-_\s]+', '', norm2)

        if norm1_no_spaces == norm2_no_spaces and norm1 != norm2:
            return 0.97  # Spacing differences only

        # For all other cases, return low similarity to prevent dangerous matches
        return 0.0  # Conservative: no similarity matching for medical terms

    def find_by_similarity(self, term: str, entity_type: str, threshold: float = 0.95) -> List[dict]:
        """
        HEAVILY RESTRICTED similarity matching for medical terms.

        WARNING: This method is now extremely conservative to prevent dangerous
        false positives in medical data. It only matches very safe variations.

        For uncertain matches, use LLM verification instead!

        Args:
            term: The term to search for
            entity_type: Either 'intervention' or 'condition'
            threshold: Minimum similarity score (default 0.95 - very high!)

        Returns:
            List of matching canonical entities (will be very few)
        """
        matches = []

        # Force high threshold for medical safety
        safe_threshold = max(threshold, 0.95)

        cursor = self.db.cursor()

        # Get all canonical entities and their mappings for this entity type
        cursor.execute("""
            SELECT DISTINCT ce.id, ce.canonical_name, ce.entity_type,
                   em.raw_text, em.confidence_score
            FROM canonical_entities ce
            LEFT JOIN entity_mappings em ON ce.id = em.canonical_id
            WHERE ce.entity_type = ?
        """, (entity_type,))

        entities = cursor.fetchall()

        for entity in entities:
            # Check similarity with canonical name
            canonical_similarity = self.calculate_string_similarity(term, entity['canonical_name'])
            if canonical_similarity >= safe_threshold:

                # Additional safety check: flag for review if not identical
                needs_review = canonical_similarity < 1.0 and canonical_similarity > 0.9

                matches.append({
                    'id': entity['id'],
                    'canonical_name': entity['canonical_name'],
                    'entity_type': entity['entity_type'],
                    'description': entity['description'] if 'description' in entity.keys() else '',
                    'similarity_score': canonical_similarity,
                    'match_method': 'restricted_similarity',
                    'matched_text': entity['canonical_name'],
                    'needs_llm_verification': needs_review,
                    'safety_note': 'High similarity but not identical - medical terms require LLM verification'
                })

            # Check similarity with mapped terms
            if entity['raw_text']:
                mapping_similarity = self.calculate_string_similarity(term, entity['raw_text'])
                if mapping_similarity >= safe_threshold:

                    needs_review = mapping_similarity < 1.0 and mapping_similarity > 0.9

                    matches.append({
                        'id': entity['id'],
                        'canonical_name': entity['canonical_name'],
                        'entity_type': entity['entity_type'],
                        'description': entity['description'] if 'description' in entity.keys() else '',
                        'similarity_score': mapping_similarity,
                        'match_method': 'restricted_similarity',
                        'matched_text': entity['raw_text'],
                        'needs_llm_verification': needs_review,
                        'safety_note': 'High similarity but not identical - medical terms require LLM verification'
                    })

        # Remove duplicates based on canonical ID, keeping highest similarity
        best_matches = {}
        for match in matches:
            entity_id = match['id']
            if entity_id not in best_matches or match['similarity_score'] > best_matches[entity_id]['similarity_score']:
                best_matches[entity_id] = match

        # Sort by similarity score descending
        return sorted(best_matches.values(), key=lambda x: x['similarity_score'], reverse=True)

    def find_safe_matches_only(self, term: str, entity_type: str) -> List[dict]:
        """
        Find matches using only the safest methods for medical terms.

        This is the recommended method for medical term matching.
        Uses only:
        1. Existing mappings (100% safe)
        2. Exact normalized matches (case/punctuation only)
        3. Safe pattern matching (plurals, spacing, "the" removal)

        NO similarity matching to prevent dangerous false positives!

        Args:
            term: The term to search for
            entity_type: Either 'intervention' or 'condition'

        Returns:
            List of safe canonical entity matches
        """
        all_matches = []

        # 1. Check existing mappings first (safest)
        existing_id = self.find_canonical_id(term, entity_type)
        if existing_id:
            canonical_name = self.get_canonical_name(term, entity_type)
            all_matches.append({
                'id': existing_id,
                'canonical_name': canonical_name,
                'entity_type': entity_type,
                'confidence': 1.0,  # Use 'confidence' consistently
                'match_method': 'existing_mapping',
                'safety_level': 'safe'
            })

        # 2. Try exact normalized match
        exact_match = self.find_by_exact_match(term, entity_type)
        if exact_match and not any(m['id'] == exact_match['id'] for m in all_matches):
            exact_match['safety_level'] = 'safe'
            exact_match['match_method'] = 'exact_normalized'
            exact_match['confidence'] = 0.95
            all_matches.append(exact_match)

        # 3. Try safe pattern matching
        pattern_matches = self.find_by_pattern(term, entity_type)
        for match in pattern_matches:
            if not any(m['id'] == match['id'] for m in all_matches):
                match['safety_level'] = 'safe'
                # Ensure pattern matches have the right field names
                if 'confidence_score' in match and 'confidence' not in match:
                    match['confidence'] = match['confidence_score']
                all_matches.append(match)

        # Sort by confidence score descending
        return sorted(all_matches, key=lambda x: x.get('confidence', x.get('confidence_score', 0)), reverse=True)

    # === LLM-ENHANCED MATCHING METHODS ===

    def _get_cache_key(self, term: str, entity_type: str, candidate_canonicals: Optional[List[str]] = None) -> str:
        """Generate cache key for LLM decisions"""
        candidates_str = json.dumps(sorted(candidate_canonicals or []), sort_keys=True)
        key_content = f"{term}|{entity_type}|{candidates_str}|{self.llm_model}"
        return hashlib.md5(key_content.encode()).hexdigest()

    def _get_llm_cache(self, term: str, entity_type: str, candidate_canonicals: Optional[List[str]] = None) -> Optional[dict]:
        """Check if we have a cached LLM decision for this term"""
        cursor = self.db.cursor()

        candidates_json = json.dumps(sorted(candidate_canonicals or []), sort_keys=True) if candidate_canonicals else None

        cursor.execute("""
            SELECT match_result, confidence_score, reasoning, llm_response, created_at
            FROM llm_normalization_cache
            WHERE input_term = ? AND entity_type = ? AND candidate_canonicals = ? AND model_name = ?
            ORDER BY created_at DESC
            LIMIT 1
        """, (term, entity_type, candidates_json, self.llm_model))

        result = cursor.fetchone()
        if result:
            return {
                'match_result': result['match_result'],
                'confidence_score': result['confidence_score'],
                'reasoning': result['reasoning'],
                'llm_response': result['llm_response'],
                'cached_at': result['created_at']
            }
        return None

    def _save_llm_cache(self, term: str, entity_type: str, candidate_canonicals: Optional[List[str]],
                       llm_response: str, match_result: Optional[str], confidence: float, reasoning: str):
        """Save LLM decision to cache"""
        cursor = self.db.cursor()

        candidates_json = json.dumps(sorted(candidate_canonicals or []), sort_keys=True) if candidate_canonicals else None

        try:
            cursor.execute("""
                INSERT OR REPLACE INTO llm_normalization_cache
                (input_term, entity_type, candidate_canonicals, llm_response, match_result,
                 confidence_score, reasoning, model_name)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (term, entity_type, candidates_json, llm_response, match_result, confidence, reasoning, self.llm_model))

            self.db.commit()
        except Exception as e:
            print(f"Warning: Failed to cache LLM decision: {e}")

    def _build_llm_prompt(self, term: str, candidate_canonicals: List[str], entity_type: str) -> str:
        """Build medical-aware LLM prompt for entity matching"""

        candidates_list = "\n".join([f"- {canonical}" for canonical in candidate_canonicals])

        prompt = f"""You are a medical terminology expert. Given the {entity_type} term '{term}', determine if it represents the same medical concept as any of these canonical terms:

{candidates_list}

IMPORTANT MEDICAL CONSIDERATIONS:
- Be very conservative - only match if you're confident they represent the SAME medical concept
- Different substances are different (probiotics != prebiotics)
- Opposite conditions are different (hypertension != hypotension)
- Similar-sounding but different medical terms should NOT match
- Consider synonyms, common names, and abbreviations that refer to the same concept

Respond with valid JSON only:
{{
    "match": "exact_canonical_name_from_list_above" or null,
    "confidence": 0.0-1.0,
    "reasoning": "brief medical explanation"
}}"""

        return prompt

    def find_by_llm(self, term: str, entity_type: str, candidate_canonicals: Optional[List[str]] = None) -> Optional[dict]:
        """
        Find canonical match using LLM semantic understanding.

        This method uses an LLM to identify medical synonyms and related terms
        that pattern matching might miss, while being conservative to avoid
        dangerous false positives.

        Args:
            term: The term to find matches for
            entity_type: Either 'intervention' or 'condition'
            candidate_canonicals: Specific canonicals to test against (optional)

        Returns:
            Dictionary with match info or None if no confident match
        """
        if not LLM_AVAILABLE or not self.llm_client:
            return None

        # Get candidate canonicals if not provided
        if candidate_canonicals is None:
            # Get all canonical entities for this type
            cursor = self.db.cursor()
            cursor.execute("""
                SELECT canonical_name FROM canonical_entities WHERE entity_type = ?
                ORDER BY canonical_name
            """, (entity_type,))
            candidate_canonicals = [row['canonical_name'] for row in cursor.fetchall()]

        if not candidate_canonicals:
            return None

        # Check cache first
        cached = self._get_llm_cache(term, entity_type, candidate_canonicals)
        if cached:
            if cached['match_result']:
                # Get canonical ID
                canonical_id = None
                cursor = self.db.cursor()
                cursor.execute("""
                    SELECT id FROM canonical_entities
                    WHERE canonical_name = ? AND entity_type = ?
                """, (cached['match_result'], entity_type))
                result = cursor.fetchone()
                if result:
                    canonical_id = result['id']

                return {
                    'id': canonical_id,
                    'canonical_name': cached['match_result'],
                    'entity_type': entity_type,
                    'confidence': cached['confidence_score'],
                    'match_method': 'llm_semantic',
                    'reasoning': cached['reasoning'],
                    'cached': True
                }
            return None

        # Query LLM
        try:
            prompt = self._build_llm_prompt(term, candidate_canonicals, entity_type)
            response = self.llm_client.generate(prompt, temperature=0.1)  # Low temp for consistency

            # Parse response
            llm_content = response['content'].strip()

            # Try to parse JSON response
            if LLM_AVAILABLE and 'parse_json_safely' in globals():
                parsed_list = parse_json_safely(llm_content)
                # parse_json_safely returns a list, we want the first dict
                parsed = parsed_list[0] if parsed_list and isinstance(parsed_list, list) else None
            else:
                try:
                    parsed = json.loads(llm_content)
                except json.JSONDecodeError:
                    # Try to extract JSON from response
                    import re
                    json_match = re.search(r'\{.*\}', llm_content, re.DOTALL)
                    if json_match:
                        parsed = json.loads(json_match.group())
                    else:
                        parsed = None

            if not parsed or not isinstance(parsed, dict):
                print(f"Warning: Invalid LLM response format for term '{term}'")
                return None

            match_name = parsed.get('match')
            confidence = float(parsed.get('confidence', 0.0))
            reasoning = parsed.get('reasoning', 'No reasoning provided')

            # Cache the result
            self._save_llm_cache(term, entity_type, candidate_canonicals,
                                llm_content, match_name, confidence, reasoning)

            # If we have a match, return it
            if match_name and confidence > 0.3:  # Minimum confidence threshold
                # Get canonical ID
                canonical_id = None
                cursor = self.db.cursor()
                cursor.execute("""
                    SELECT id FROM canonical_entities
                    WHERE canonical_name = ? AND entity_type = ?
                """, (match_name, entity_type))
                result = cursor.fetchone()
                if result:
                    canonical_id = result['id']

                return {
                    'id': canonical_id,
                    'canonical_name': match_name,
                    'entity_type': entity_type,
                    'confidence': confidence,
                    'match_method': 'llm_semantic',
                    'reasoning': reasoning,
                    'cached': False
                }

        except Exception as e:
            print(f"Error in LLM matching for term '{term}': {e}")
            return None

        return None

    def find_or_create_mapping(self, term: str, entity_type: str, confidence_threshold: float = 0.7) -> Dict[str, Any]:
        """
        Find or create a mapping for a term using all available methods

        This is the main method for the extraction pipeline.
        Uses fast methods first, falls back to LLM, creates new canonical if needed.

        Args:
            term: The term to normalize
            entity_type: 'intervention' or 'condition'
            confidence_threshold: Minimum confidence for LLM matching

        Returns:
            Dictionary with canonical_id, canonical_name, method, confidence, is_new
        """
        if not term or not term.strip():
            return {
                'canonical_id': None,
                'canonical_name': term,
                'method': 'empty_term',
                'confidence': 0.0,
                'is_new': False,
                'reasoning': 'Empty or whitespace term'
            }

        term = term.strip()

        # Step 1: Try fast safe methods first
        safe_matches = self.find_safe_matches_only(term, entity_type)

        if safe_matches:
            best_match = safe_matches[0]
            return {
                'canonical_id': best_match['id'],
                'canonical_name': best_match['canonical_name'],
                'method': best_match['match_method'],
                'confidence': best_match['confidence'],
                'is_new': False,
                'reasoning': 'Found via safe matching methods'
            }

        # Step 2: Try LLM semantic matching if available
        if self.llm_client:
            llm_match = self.find_by_llm(term, entity_type)

            if llm_match and llm_match['confidence'] >= confidence_threshold:
                return {
                    'canonical_id': llm_match['id'],
                    'canonical_name': llm_match['canonical_name'],
                    'method': 'llm_semantic',
                    'confidence': llm_match['confidence'],
                    'is_new': False,
                    'reasoning': llm_match.get('reasoning', 'LLM semantic match')
                }

        # Step 3: Create new canonical entity if no good match found
        print(f"Creating new canonical entity for: {term} ({entity_type})")

        try:
            canonical_id = self.create_canonical_entity(term, entity_type)

            # Add the term as its own canonical mapping
            self.add_term_mapping(term, canonical_id, 1.0, "exact_canonical")

            return {
                'canonical_id': canonical_id,
                'canonical_name': term,
                'method': 'new_canonical',
                'confidence': 1.0,
                'is_new': True,
                'reasoning': 'No existing match found, created new canonical entity'
            }

        except Exception as e:
            print(f"Error creating canonical entity for '{term}': {e}")
            return {
                'canonical_id': None,
                'canonical_name': term,
                'method': 'error',
                'confidence': 0.0,
                'is_new': False,
                'reasoning': f'Error creating canonical entity: {e}'
            }

    def batch_find_synonyms(self, terms_list: List[str], entity_type: str = None) -> dict:
        """
        Efficiently find synonyms for multiple terms using LLM batch processing.

        Groups terms by entity type and processes them together for efficiency.

        Args:
            terms_list: List of terms to find synonyms for
            entity_type: Optional entity type filter

        Returns:
            Dictionary mapping terms to their LLM match results
        """
        if not LLM_AVAILABLE or not self.llm_client:
            return {}

        results = {}

        # Group terms by entity type if not specified
        if entity_type:
            term_groups = {entity_type: terms_list}
        else:
            # Try to infer entity types or default to checking both
            term_groups = {
                'intervention': terms_list,
                'condition': terms_list
            }

        for et, terms in term_groups.items():
            # Get all canonical entities for this type once
            cursor = self.db.cursor()
            cursor.execute("""
                SELECT canonical_name FROM canonical_entities WHERE entity_type = ?
                ORDER BY canonical_name
            """, (et,))
            candidate_canonicals = [row['canonical_name'] for row in cursor.fetchall()]

            if not candidate_canonicals:
                continue

            # Process each term (could be optimized further with batch LLM calls)
            for term in terms:
                if term not in results:
                    match_result = self.find_by_llm(term, et, candidate_canonicals)
                    if match_result:
                        results[term] = match_result

        return results

    def find_comprehensive_matches(self, term: str, entity_type: str, use_llm: bool = True) -> List[dict]:
        """
        Find matches using all available methods: safe patterns + LLM.

        This is the recommended method for comprehensive entity matching that
        combines the safety of pattern matching with the intelligence of LLM.

        Args:
            term: The term to find matches for
            entity_type: Either 'intervention' or 'condition'
            use_llm: Whether to use LLM for semantic matching

        Returns:
            List of all matches found, sorted by confidence
        """
        all_matches = []

        # 1. First try safe matching (fast and reliable)
        safe_matches = self.find_safe_matches_only(term, entity_type)
        all_matches.extend(safe_matches)

        # 2. If no safe matches and LLM is available, try semantic matching
        if not all_matches and use_llm and LLM_AVAILABLE:
            llm_match = self.find_by_llm(term, entity_type)
            if llm_match:
                all_matches.append(llm_match)

        # Remove duplicates based on canonical ID
        seen_ids = set()
        unique_matches = []
        for match in all_matches:
            if match['id'] not in seen_ids:
                seen_ids.add(match['id'])
                unique_matches.append(match)

        # Sort by confidence (safe matches first, then LLM matches)
        def sort_key(match):
            # Prioritize safe matches
            if match.get('safety_level') == 'safe':
                return (1, match.get('confidence', 0))
            else:
                return (0, match.get('confidence', 0))

        return sorted(unique_matches, key=sort_key, reverse=True)