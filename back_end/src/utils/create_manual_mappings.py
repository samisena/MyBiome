#!/usr/bin/env python3
"""
Manual creation of canonical entities and mappings for the top 10 most obvious cases
"""

import sqlite3
import sys
import os

# Add the src directory to the path so we can import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from ..llm_processing.batch_entity_processor import BatchEntityProcessor as EntityNormalizer


def create_manual_mappings():
    """Create canonical entities and mappings for the most obvious normalization cases"""

    # Connect to the database
    db_path = "data/processed/intervention_research.db"
    if not os.path.exists(db_path):
        print(f"[FAIL] Database not found at {db_path}")
        return False

    try:
        conn = sqlite3.connect(db_path)
        normalizer = EntityNormalizer(conn)
        print("[INFO] Connected to database and initialized EntityNormalizer")

        # 1. PROBIOTICS (intervention)
        print("\n=== Creating Probiotics Canonical Entity ===")
        try:
            probiotics_id = normalizer.create_canonical_entity(
                canonical_name="probiotics",
                entity_type="intervention"
            )
            print(f"[PASS] Created probiotics canonical entity with ID: {probiotics_id}")

            # Add variants
            probiotic_variants = [
                ("probiotics", 1.0, "exact_match"),
                ("Probiotics", 1.0, "case_variant"),
                ("probiotic", 0.95, "singular_form"),
                ("Probiotic", 0.95, "singular_case_variant"),
                ("probiotic supplements", 0.9, "descriptive_variant")
            ]

            for variant, confidence, method in probiotic_variants:
                mapping_id = normalizer.add_term_mapping(variant, probiotics_id, confidence, method)
                print(f"[PASS] Mapped '{variant}' to probiotics (mapping ID: {mapping_id})")

        except Exception as e:
            print(f"[WARN] Probiotics may already exist: {e}")

        # 2. IRRITABLE BOWEL SYNDROME (condition)
        print("\n=== Creating IBS Canonical Entity ===")
        try:
            ibs_id = normalizer.create_canonical_entity(
                canonical_name="irritable bowel syndrome",
                entity_type="condition"
            )
            print(f"[PASS] Created IBS canonical entity with ID: {ibs_id}")

            # Add variants
            ibs_variants = [
                ("irritable bowel syndrome (IBS)", 1.0, "exact_match"),
                ("Irritable Bowel Syndrome (IBS)", 1.0, "case_variant"),
                ("irritable bowel syndrome", 1.0, "without_abbreviation"),
                ("Irritable bowel syndrome", 1.0, "case_variant_no_abbrev"),
                ("IBS", 1.0, "abbreviation"),
                ("ibs", 1.0, "lowercase_abbreviation")
            ]

            for variant, confidence, method in ibs_variants:
                mapping_id = normalizer.add_term_mapping(variant, ibs_id, confidence, method)
                print(f"[PASS] Mapped '{variant}' to irritable bowel syndrome (mapping ID: {mapping_id})")

        except Exception as e:
            print(f"[WARN] IBS may already exist: {e}")

        # 3. LOW FODMAP DIET (intervention)
        print("\n=== Creating FODMAP Diet Canonical Entity ===")
        try:
            fodmap_id = normalizer.create_canonical_entity(
                canonical_name="low FODMAP diet",
                entity_type="intervention"
            )
            print(f"[PASS] Created FODMAP diet canonical entity with ID: {fodmap_id}")

            # Add variants
            fodmap_variants = [
                ("low FODMAP diet", 1.0, "exact_match"),
                ("FODMAP diet", 0.95, "missing_low"),
                ("low-FODMAP diet", 1.0, "hyphen_variant"),
                ("Low FODMAP diet", 1.0, "case_variant"),
                ("LOW FODMAP DIET", 0.9, "uppercase_variant")
            ]

            for variant, confidence, method in fodmap_variants:
                mapping_id = normalizer.add_term_mapping(variant, fodmap_id, confidence, method)
                print(f"[PASS] Mapped '{variant}' to low FODMAP diet (mapping ID: {mapping_id})")

        except Exception as e:
            print(f"[WARN] FODMAP diet may already exist: {e}")

        # 4. PLACEBO (intervention)
        print("\n=== Creating Placebo Canonical Entity ===")
        try:
            placebo_id = normalizer.create_canonical_entity(
                canonical_name="placebo",
                entity_type="intervention"
            )
            print(f"[PASS] Created placebo canonical entity with ID: {placebo_id}")

            # Add variants
            placebo_variants = [
                ("placebo", 1.0, "exact_match"),
                ("Placebo", 1.0, "case_variant"),
                ("PLACEBO", 0.9, "uppercase_variant"),
                ("placebo control", 0.95, "descriptive_variant"),
                ("placebo treatment", 0.9, "treatment_variant")
            ]

            for variant, confidence, method in placebo_variants:
                mapping_id = normalizer.add_term_mapping(variant, placebo_id, confidence, method)
                print(f"[PASS] Mapped '{variant}' to placebo (mapping ID: {mapping_id})")

        except Exception as e:
            print(f"[WARN] Placebo may already exist: {e}")

        # 5. FECAL MICROBIOTA TRANSPLANTATION (intervention)
        print("\n=== Creating FMT Canonical Entity ===")
        try:
            fmt_id = normalizer.create_canonical_entity(
                canonical_name="fecal microbiota transplantation",
                entity_type="intervention"
            )
            print(f"[PASS] Created FMT canonical entity with ID: {fmt_id}")

            # Add variants
            fmt_variants = [
                ("fecal microbiota transplantation (FMT)", 1.0, "exact_match"),
                ("Faecal microbiota transplantation (FMT)", 1.0, "british_spelling"),
                ("fecal microbiota transplantation", 1.0, "without_abbreviation"),
                ("FMT", 1.0, "abbreviation"),
                ("fmt", 0.9, "lowercase_abbreviation")
            ]

            for variant, confidence, method in fmt_variants:
                mapping_id = normalizer.add_term_mapping(variant, fmt_id, confidence, method)
                print(f"[PASS] Mapped '{variant}' to fecal microbiota transplantation (mapping ID: {mapping_id})")

        except Exception as e:
            print(f"[WARN] FMT may already exist: {e}")

        # 6. SMALL INTESTINAL BACTERIAL OVERGROWTH (condition)
        print("\n=== Creating SIBO Canonical Entity ===")
        try:
            sibo_id = normalizer.create_canonical_entity(
                canonical_name="small intestinal bacterial overgrowth",
                entity_type="condition"
            )
            print(f"[PASS] Created SIBO canonical entity with ID: {sibo_id}")

            # Add variants
            sibo_variants = [
                ("small intestinal bacterial overgrowth (SIBO)", 1.0, "exact_match"),
                ("small intestinal bacterial overgrowth", 1.0, "without_abbreviation"),
                ("SIBO", 1.0, "abbreviation"),
                ("sibo", 1.0, "lowercase_abbreviation"),
                ("Small intestinal bacterial overgrowth", 1.0, "case_variant")
            ]

            for variant, confidence, method in sibo_variants:
                mapping_id = normalizer.add_term_mapping(variant, sibo_id, confidence, method)
                print(f"[PASS] Mapped '{variant}' to small intestinal bacterial overgrowth (mapping ID: {mapping_id})")

        except Exception as e:
            print(f"[WARN] SIBO may already exist: {e}")

        # 7. TYPE 2 DIABETES MELLITUS (condition)
        print("\n=== Creating Type 2 Diabetes Canonical Entity ===")
        try:
            t2dm_id = normalizer.create_canonical_entity(
                canonical_name="type 2 diabetes mellitus",
                entity_type="condition"
            )
            print(f"[PASS] Created T2DM canonical entity with ID: {t2dm_id}")

            # Add variants
            t2dm_variants = [
                ("type 2 diabetes mellitus (T2DM)", 1.0, "exact_match"),
                ("type 2 diabetes mellitus", 1.0, "without_abbreviation"),
                ("type 2 diabetes", 0.98, "shortened_form"),
                ("Type 2 diabetes", 0.98, "case_variant_short"),
                ("T2DM", 1.0, "abbreviation")
            ]

            for variant, confidence, method in t2dm_variants:
                mapping_id = normalizer.add_term_mapping(variant, t2dm_id, confidence, method)
                print(f"[PASS] Mapped '{variant}' to type 2 diabetes mellitus (mapping ID: {mapping_id})")

        except Exception as e:
            print(f"[WARN] T2DM may already exist: {e}")

        # 8. CORONARY HEART DISEASE (condition)
        print("\n=== Creating Coronary Heart Disease Canonical Entity ===")
        try:
            chd_id = normalizer.create_canonical_entity(
                canonical_name="coronary heart disease",
                entity_type="condition"
            )
            print(f"[PASS] Created CHD canonical entity with ID: {chd_id}")

            # Add variants
            chd_variants = [
                ("coronary heart disease", 1.0, "exact_match"),
                ("coronary heart disease (CHD)", 1.0, "with_abbreviation"),
                ("Coronary heart disease", 1.0, "case_variant"),
                ("CHD", 1.0, "abbreviation"),
                ("coronary artery disease", 0.95, "related_term")  # Note: these may be clinically different
            ]

            for variant, confidence, method in chd_variants:
                mapping_id = normalizer.add_term_mapping(variant, chd_id, confidence, method)
                print(f"[PASS] Mapped '{variant}' to coronary heart disease (mapping ID: {mapping_id})")

        except Exception as e:
            print(f"[WARN] CHD may already exist: {e}")

        # 9. MIGRAINE (condition)
        print("\n=== Creating Migraine Canonical Entity ===")
        try:
            migraine_id = normalizer.create_canonical_entity(
                canonical_name="migraine",
                entity_type="condition"
            )
            print(f"[PASS] Created migraine canonical entity with ID: {migraine_id}")

            # Add variants
            migraine_variants = [
                ("migraine", 1.0, "exact_match"),
                ("Migraine", 1.0, "case_variant"),
                ("migraine headache", 1.0, "redundant_descriptor"),
                ("Migraine headache", 1.0, "case_variant_redundant"),
                ("migraines", 0.98, "plural_form")
            ]

            for variant, confidence, method in migraine_variants:
                mapping_id = normalizer.add_term_mapping(variant, migraine_id, confidence, method)
                print(f"[PASS] Mapped '{variant}' to migraine (mapping ID: {mapping_id})")

        except Exception as e:
            print(f"[WARN] Migraine may already exist: {e}")

        # 10. ARRHYTHMOGENIC RIGHT VENTRICULAR CARDIOMYOPATHY (condition)
        print("\n=== Creating ARVC Canonical Entity ===")
        try:
            arvc_id = normalizer.create_canonical_entity(
                canonical_name="arrhythmogenic right ventricular cardiomyopathy",
                entity_type="condition"
            )
            print(f"[PASS] Created ARVC canonical entity with ID: {arvc_id}")

            # Add variants
            arvc_variants = [
                ("arrhythmogenic right ventricular cardiomyopathy (ARVC)", 1.0, "exact_match"),
                ("Arrhythmogenic right ventricular cardiomyopathy (ARVC)", 1.0, "case_variant"),
                ("arrhythmogenic right ventricular cardiomyopathy", 1.0, "without_abbreviation"),
                ("ARVC", 1.0, "abbreviation"),
                ("arvc", 1.0, "lowercase_abbreviation")
            ]

            for variant, confidence, method in arvc_variants:
                mapping_id = normalizer.add_term_mapping(variant, arvc_id, confidence, method)
                print(f"[PASS] Mapped '{variant}' to arrhythmogenic right ventricular cardiomyopathy (mapping ID: {mapping_id})")

        except Exception as e:
            print(f"[WARN] ARVC may already exist: {e}")

        # Get final statistics
        stats = normalizer.get_mapping_stats()
        print(f"\n=== Final Statistics ===")
        print(f"Canonical entities: {stats.get('canonical_entities', {})}")
        print(f"Total mappings: {stats.get('mappings', {})}")
        print(f"Compression ratios: {stats.get('ratios', {})}")

        conn.close()
        print(f"\n[SUCCESS] Manual canonical entities and mappings created successfully!")
        return True

    except Exception as e:
        print(f"[FAIL] Error creating manual mappings: {e}")
        return False


if __name__ == "__main__":
    success = create_manual_mappings()
    sys.exit(0 if success else 1)