#!/usr/bin/env python3
"""
Temperature Experiment for Phase 2 LLM Processing

Tests different temperature settings (0, 0.15, 0.3) on a single paper to compare:
- Runtime performance
- Output quality and completeness
- JSON structure richness

Mimics Phase 2 pipeline exactly but with controlled temperature variations.
"""

import sys
import time
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from back_end.src.data.config import config, setup_logging
from back_end.src.phase_1_data_collection.database_manager import database_manager
from back_end.src.data.api_clients import get_llm_client
from back_end.src.phase_2_llm_processing.phase_2_prompt_service import phase_2_prompt_service
from back_end.src.data.utils import parse_json_safely

logger = setup_logging(__name__, 'temperature_experiment.log')


@dataclass
class ExtractionResult:
    """Results from a single extraction run."""
    temperature: float
    extraction_time: float
    total_interventions: int
    raw_response: str
    parsed_data: List[Dict]
    error: Optional[str] = None

    # Metrics
    field_completeness: float = 0.0  # % of non-null fields
    avg_mechanism_length: float = 0.0
    study_fields_present: int = 0  # Count of study-level fields
    json_size: int = 0  # Character count

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)


class TemperatureExperiment:
    """
    Standalone temperature experiment for Phase 2 extraction.

    Tests multiple temperature values on a single paper to compare:
    - Runtime performance
    - Extraction quality
    - Output consistency
    """

    def __init__(self, temperatures: Optional[List[float]] = None):
        """Initialize the experiment.

        Args:
            temperatures: List of temperature values to test (default: [0, 0.15, 0.3])
        """
        self.model_name = 'qwen3:14b'
        self.temperatures = temperatures if temperatures is not None else [0, 0.15, 0.3]
        self.max_context = 32768
        self.recommended_max_output = 16384

        self.results: List[ExtractionResult] = []

        logger.info(f"Temperature experiment initialized with temperatures: {self.temperatures}")

    def select_paper(self) -> Optional[Dict[str, Any]]:
        """
        Select a single paper from the database for testing.

        Priority:
        1. Processed paper with high intervention count (rich data)
        2. Unprocessed paper with good abstract

        Returns:
            Paper dictionary with pmid, title, abstract
        """
        try:
            with database_manager.get_connection() as conn:
                cursor = conn.cursor()

                # Try to get a processed paper with interventions (use LEFT JOIN to handle papers without interventions)
                cursor.execute("""
                    SELECT p.pmid, p.title, p.abstract, COUNT(i.id) as intervention_count
                    FROM papers p
                    LEFT JOIN interventions i ON p.pmid = i.paper_id
                    WHERE p.abstract IS NOT NULL
                      AND p.abstract != ''
                      AND LENGTH(p.abstract) > 200
                    GROUP BY p.pmid, p.title, p.abstract
                    ORDER BY intervention_count DESC
                    LIMIT 1
                """)

                row = cursor.fetchone()

                if row:
                    intervention_count = row[3]
                    if intervention_count > 0:
                        logger.info(f"Selected processed paper: {row[0]} (has {intervention_count} interventions)")
                    else:
                        logger.info(f"Selected paper: {row[0]} (no interventions yet)")

                    return {
                        'pmid': row[0],
                        'title': row[1],
                        'abstract': row[2]
                    }

                logger.error("No suitable papers found in database")
                return None

        except Exception as e:
            logger.error(f"Error selecting paper: {e}")
            return None

    def calculate_dynamic_max_tokens(self, prompt: str) -> int:
        """
        Calculate dynamic max_tokens (same logic as Phase 2).

        Args:
            prompt: The input prompt

        Returns:
            Optimal max_tokens for this request
        """
        # Rough estimation: 1 token ≈ 4 characters
        estimated_input_tokens = len(prompt) // 4

        # Calculate available context space
        buffer_tokens = 500
        available_for_output = self.max_context - estimated_input_tokens - buffer_tokens

        # Use the minimum of available space and recommended max output
        dynamic_max_tokens = min(available_for_output, self.recommended_max_output)

        # Ensure we have at least a reasonable minimum
        dynamic_max_tokens = max(dynamic_max_tokens, 2048)

        return dynamic_max_tokens

    def extract_with_temperature(self, paper: Dict, temperature: float) -> ExtractionResult:
        """
        Extract interventions from paper with specific temperature.

        Args:
            paper: Paper dictionary with pmid, title, abstract
            temperature: Temperature setting to test

        Returns:
            ExtractionResult with metrics
        """
        pmid = paper['pmid']
        start_time = time.time()

        logger.info(f"Extracting with temperature={temperature}")
        print(f"  [Temperature {temperature}] Starting extraction...")

        try:
            # Create prompt (same as Phase 2)
            print(f"  [Temperature {temperature}] Creating prompt...")
            prompt = prompt_service.create_extraction_prompt(paper)

            # Calculate dynamic max_tokens (same as Phase 2)
            dynamic_max_tokens = self.calculate_dynamic_max_tokens(prompt)
            print(f"  [Temperature {temperature}] Max tokens: {dynamic_max_tokens}")

            # Get system message (same as Phase 2)
            system_message = prompt_service.create_system_message()

            # Create LLM client
            print(f"  [Temperature {temperature}] Calling LLM (this may take 60-120 seconds)...")
            client = get_llm_client(self.model_name)

            # Call LLM with specific temperature
            response = client.generate(
                prompt=prompt,
                temperature=temperature,
                max_tokens=dynamic_max_tokens,
                system_message=system_message
            )

            # Extract response
            response_text = response.get('content', '')
            extraction_time = time.time() - start_time
            print(f"  [Temperature {temperature}] LLM call completed in {extraction_time:.1f}s")

            # Parse JSON response (same as Phase 2)
            print(f"  [Temperature {temperature}] Parsing JSON response...")
            hierarchical_data = parse_json_safely(response_text, f"{pmid}_temp{temperature}")

            # Calculate metrics
            print(f"  [Temperature {temperature}] Calculating metrics...")
            metrics = self._calculate_metrics(hierarchical_data, response_text)

            result = ExtractionResult(
                temperature=temperature,
                extraction_time=extraction_time,
                total_interventions=metrics['total_interventions'],
                raw_response=response_text,
                parsed_data=hierarchical_data or [],
                field_completeness=metrics['field_completeness'],
                avg_mechanism_length=metrics['avg_mechanism_length'],
                study_fields_present=metrics['study_fields_present'],
                json_size=len(response_text)
            )

            print(f"  [Temperature {temperature}] COMPLETE: {result.total_interventions} interventions, "
                  f"{result.field_completeness:.1f}% completeness, {extraction_time:.1f}s\n")
            logger.info(f"Extraction complete: {result.total_interventions} interventions in {extraction_time:.1f}s")

            return result

        except Exception as e:
            extraction_time = time.time() - start_time
            logger.error(f"Error extracting with temperature={temperature}: {e}")

            return ExtractionResult(
                temperature=temperature,
                extraction_time=extraction_time,
                total_interventions=0,
                raw_response="",
                parsed_data=[],
                error=str(e)
            )

    def _calculate_metrics(self, hierarchical_data: List[Dict], raw_response: str) -> Dict[str, Any]:
        """
        Calculate quality metrics from extraction results.

        Args:
            hierarchical_data: Parsed hierarchical extraction data
            raw_response: Raw JSON response text

        Returns:
            Dictionary with metrics
        """
        if not hierarchical_data:
            return {
                'total_interventions': 0,
                'field_completeness': 0.0,
                'avg_mechanism_length': 0.0,
                'study_fields_present': 0
            }

        # Count total interventions
        total_interventions = sum(len(entry.get('interventions', [])) for entry in hierarchical_data)

        # Calculate field completeness
        all_fields = []
        mechanism_lengths = []
        study_fields_present = 0

        for entry in hierarchical_data:
            # Study-level fields
            study_fields = [
                'health_condition', 'study_focus', 'measured_metrics', 'findings',
                'study_location', 'publisher', 'sample_size', 'study_duration',
                'study_type', 'population_details'
            ]

            # Count non-null study fields
            study_fields_present += sum(1 for field in study_fields if entry.get(field) not in [None, '', [], {}])

            # Intervention-level fields
            for intervention in entry.get('interventions', []):
                intervention_fields = [
                    'intervention_name', 'dosage', 'duration', 'frequency', 'intensity',
                    'administration_route', 'mechanism', 'correlation_type',
                    'correlation_strength', 'delivery_method', 'adverse_effects',
                    'extraction_confidence'
                ]

                # Count non-null intervention fields
                non_null_count = sum(1 for field in intervention_fields if intervention.get(field) not in [None, '', [], {}])
                all_fields.append(non_null_count / len(intervention_fields))

                # Track mechanism length
                mechanism = intervention.get('mechanism', '')
                if mechanism and mechanism.strip():
                    mechanism_lengths.append(len(mechanism.split()))

        # Calculate averages
        field_completeness = (sum(all_fields) / len(all_fields) * 100) if all_fields else 0.0
        avg_mechanism_length = (sum(mechanism_lengths) / len(mechanism_lengths)) if mechanism_lengths else 0.0

        return {
            'total_interventions': total_interventions,
            'field_completeness': field_completeness,
            'avg_mechanism_length': avg_mechanism_length,
            'study_fields_present': study_fields_present
        }

    def run_experiment(self, paper: Dict) -> List[ExtractionResult]:
        """
        Run experiment with all temperature values.

        Args:
            paper: Paper to test on

        Returns:
            List of extraction results
        """
        logger.info(f"Starting temperature experiment on paper {paper['pmid']}")
        logger.info(f"Testing temperatures: {self.temperatures}")

        results = []

        for i, temp in enumerate(self.temperatures, 1):
            print(f"\n{'='*80}")
            print(f"RUN {i}/{len(self.temperatures)}: Temperature = {temp}")
            print(f"{'='*80}\n")
            logger.info(f"\n{'='*60}")
            logger.info(f"Testing temperature: {temp}")
            logger.info(f"{'='*60}")

            result = self.extract_with_temperature(paper, temp)
            results.append(result)

            # Small delay between runs
            if temp != self.temperatures[-1]:
                print(f"Cooling down for 5 seconds before next run...\n")
                logger.info("Cooling down for 5 seconds...")
                time.sleep(5)

        self.results = results
        return results

    def generate_comparison_report(self, paper: Dict) -> str:
        """
        Generate detailed comparison report.

        Args:
            paper: Paper that was tested

        Returns:
            Markdown-formatted report
        """
        report = []
        report.append("# Temperature Experiment Report")
        report.append(f"\n**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"**Model**: {self.model_name}")
        report.append(f"**Paper**: {paper['pmid']}")
        report.append(f"\n## Paper Details")
        report.append(f"**Title**: {paper['title']}")
        report.append(f"**Abstract Length**: {len(paper['abstract'])} characters")

        # Summary table
        report.append(f"\n## Performance Summary")
        report.append("\n| Temperature | Time (s) | Interventions | Field Completeness (%) | Avg Mechanism Length | Study Fields | JSON Size |")
        report.append("|-------------|----------|---------------|------------------------|----------------------|--------------|-----------|")

        for result in self.results:
            report.append(
                f"| {result.temperature:.2f} | {result.extraction_time:.1f} | "
                f"{result.total_interventions} | {result.field_completeness:.1f} | "
                f"{result.avg_mechanism_length:.1f} | {result.study_fields_present} | "
                f"{result.json_size} |"
            )

        # Detailed comparison
        report.append(f"\n## Detailed Analysis")

        for result in self.results:
            report.append(f"\n### Temperature: {result.temperature}")

            if result.error:
                report.append(f"**Error**: {result.error}")
                continue

            report.append(f"- **Extraction Time**: {result.extraction_time:.2f} seconds")
            report.append(f"- **Total Interventions**: {result.total_interventions}")
            report.append(f"- **Field Completeness**: {result.field_completeness:.1f}%")
            report.append(f"- **Average Mechanism Length**: {result.avg_mechanism_length:.1f} words")
            report.append(f"- **Study Fields Present**: {result.study_fields_present}")
            report.append(f"- **JSON Response Size**: {result.json_size} characters")

            # Show sample interventions
            if result.parsed_data:
                report.append(f"\n**Sample Extraction** (first intervention):")
                first_entry = result.parsed_data[0]
                if first_entry.get('interventions'):
                    first_intervention = first_entry['interventions'][0]
                    report.append(f"```json")
                    report.append(json.dumps(first_intervention, indent=2))
                    report.append(f"```")

        # Recommendations
        report.append(f"\n## Recommendations")

        # Find best temperature based on different criteria
        best_speed = min(self.results, key=lambda r: r.extraction_time if not r.error else float('inf'))
        best_quality = max(self.results, key=lambda r: r.field_completeness)
        most_interventions = max(self.results, key=lambda r: r.total_interventions)

        report.append(f"\n- **Fastest extraction**: Temperature {best_speed.temperature} ({best_speed.extraction_time:.1f}s)")
        report.append(f"- **Highest quality** (field completeness): Temperature {best_quality.temperature} ({best_quality.field_completeness:.1f}%)")
        report.append(f"- **Most interventions extracted**: Temperature {most_interventions.temperature} ({most_interventions.total_interventions} interventions)")

        # Overall recommendation
        report.append(f"\n**Overall Assessment**:")
        if best_speed.temperature == best_quality.temperature:
            report.append(f"- Temperature **{best_speed.temperature}** provides the best balance of speed and quality")
        else:
            report.append(f"- Temperature **{best_quality.temperature}** recommended for quality (despite {best_speed.extraction_time - best_quality.extraction_time:.1f}s slower)")
            report.append(f"- Temperature **{best_speed.temperature}** recommended for speed (if quality difference is minimal)")

        return "\n".join(report)

    def save_results(self, paper: Dict, output_dir: Optional[Path] = None) -> Dict[str, Path]:
        """
        Save experiment results to files.

        Args:
            paper: Paper that was tested
            output_dir: Output directory (default: data/experimentation)

        Returns:
            Dictionary mapping output type to file path
        """
        if output_dir is None:
            output_dir = config.data_root / "experimentation"

        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        base_name = f"temperature_experiment_{paper['pmid']}_{timestamp}"

        saved_files = {}

        # Save JSON results
        json_path = output_dir / f"{base_name}_results.json"
        json_data = {
            'paper': paper,
            'model': self.model_name,
            'temperatures_tested': self.temperatures,
            'timestamp': datetime.now().isoformat(),
            'results': [result.to_dict() for result in self.results]
        }

        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)

        saved_files['json'] = json_path
        logger.info(f"Saved JSON results to {json_path}")

        # Save markdown report
        report_path = output_dir / f"{base_name}_report.md"
        report = self.generate_comparison_report(paper)

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)

        saved_files['report'] = report_path
        logger.info(f"Saved report to {report_path}")

        # Save raw outputs for each temperature
        for result in self.results:
            raw_path = output_dir / f"{base_name}_temp{result.temperature}_raw.json"
            with open(raw_path, 'w', encoding='utf-8') as f:
                f.write(result.raw_response)
            saved_files[f'raw_temp_{result.temperature}'] = raw_path

        logger.info(f"Saved {len(saved_files)} files to {output_dir}")

        return saved_files

    def print_summary(self):
        """Print summary to console."""
        print("\n" + "="*80)
        print("TEMPERATURE EXPERIMENT SUMMARY")
        print("="*80)

        print(f"\nModel: {self.model_name}")
        print(f"Temperatures tested: {self.temperatures}")

        print(f"\n{'Temperature':<15} {'Time (s)':<12} {'Interventions':<15} {'Quality (%)':<15} {'JSON Size':<12}")
        print("-"*80)

        for result in self.results:
            if result.error:
                print(f"{result.temperature:<15.2f} ERROR: {result.error}")
            else:
                print(
                    f"{result.temperature:<15.2f} {result.extraction_time:<12.1f} "
                    f"{result.total_interventions:<15} {result.field_completeness:<15.1f} "
                    f"{result.json_size:<12}"
                )

        # Recommendations
        print("\n" + "="*80)
        print("RECOMMENDATIONS")
        print("="*80)

        best_speed = min(self.results, key=lambda r: r.extraction_time if not r.error else float('inf'))
        best_quality = max(self.results, key=lambda r: r.field_completeness)

        print(f"\nFastest: Temperature {best_speed.temperature} ({best_speed.extraction_time:.1f}s)")
        print(f"Highest Quality: Temperature {best_quality.temperature} ({best_quality.field_completeness:.1f}% completeness)")

        if best_speed.temperature == best_quality.temperature:
            print(f"\nRECOMMENDED: Temperature {best_speed.temperature} (best speed + quality)")
        else:
            time_diff = abs(best_speed.extraction_time - best_quality.extraction_time)
            quality_diff = abs(best_speed.field_completeness - best_quality.field_completeness)

            if quality_diff < 5.0:
                print(f"\nRECOMMENDED: Temperature {best_speed.temperature} (minimal quality difference, much faster)")
            else:
                print(f"\nRECOMMENDED: Temperature {best_quality.temperature} (quality worth {time_diff:.1f}s extra)")


def main():
    """Main entry point for temperature experiment."""
    import argparse

    parser = argparse.ArgumentParser(description="Temperature experiment for Phase 2 LLM processing")
    parser.add_argument('--temperatures', type=float, nargs='+', default=[0, 0.15, 0.3],
                        help='Temperature values to test (default: 0 0.15 0.3)')
    args = parser.parse_args()

    print("="*80)
    print("PHASE 2 TEMPERATURE EXPERIMENT")
    print("="*80)
    print(f"\nTesting temperature settings: {', '.join(map(str, args.temperatures))}")
    print("Model: qwen3:14b")
    print("\n")

    # Initialize experiment
    experiment = TemperatureExperiment(temperatures=args.temperatures)

    # Select paper
    print("Selecting paper for testing...")
    paper = experiment.select_paper()

    if not paper:
        print("ERROR: Could not select a paper from database")
        return 1

    print(f"\nSelected paper: {paper['pmid']}")
    print(f"Title: {paper['title']}")
    print(f"Abstract length: {len(paper['abstract'])} characters")

    # Run experiment
    print("\n" + "="*80)
    print("STARTING EXTRACTIONS")
    print("="*80 + "\n")

    results = experiment.run_experiment(paper)

    # Print summary
    experiment.print_summary()

    # Save results
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80 + "\n")

    saved_files = experiment.save_results(paper)

    print("\nFiles saved:")
    for file_type, file_path in saved_files.items():
        print(f"  - {file_type}: {file_path}")

    print("\n" + "="*80)
    print("EXPERIMENT COMPLETE")
    print("="*80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
