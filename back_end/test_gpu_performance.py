#!/usr/bin/env python3
"""
GPU Performance Test for LLM Processing

Tests processing speed, GPU utilization, and memory usage
for 2-3 papers with timing and resource monitoring.
"""

import sys
import time
import subprocess
import psutil
from pathlib import Path
from typing import Dict, List, Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.data.config import setup_logging
from src.paper_collection.database_manager import database_manager
from src.llm.pipeline import InterventionResearchPipeline

logger = setup_logging(__name__, 'gpu_performance_test.log')


class GPUPerformanceMonitor:
    """Monitor GPU and system performance during LLM processing."""

    def __init__(self):
        self.pipeline = InterventionResearchPipeline()
        self.measurements = []

    def get_gpu_stats(self) -> Dict[str, Any]:
        """Get current GPU statistics."""
        try:
            result = subprocess.run([
                'nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu',
                '--format=csv,noheader,nounits'
            ], capture_output=True, text=True, timeout=5)

            if result.returncode == 0:
                values = result.stdout.strip().split(', ')
                return {
                    'gpu_utilization': int(values[0]),
                    'gpu_memory_used': int(values[1]),
                    'gpu_memory_total': int(values[2]),
                    'gpu_temperature': int(values[3]),
                    'gpu_memory_percent': (int(values[1]) / int(values[2])) * 100
                }
        except Exception as e:
            logger.warning(f"Error getting GPU stats: {e}")

        return {
            'gpu_utilization': 0,
            'gpu_memory_used': 0,
            'gpu_memory_total': 0,
            'gpu_temperature': 0,
            'gpu_memory_percent': 0
        }

    def get_system_stats(self) -> Dict[str, Any]:
        """Get current system statistics."""
        memory = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent(interval=0.1)

        return {
            'cpu_percent': cpu_percent,
            'ram_used_gb': memory.used / (1024**3),
            'ram_total_gb': memory.total / (1024**3),
            'ram_percent': memory.percent
        }

    def get_ollama_models(self) -> List[Dict]:
        """Get current Ollama model status."""
        try:
            result = subprocess.run(['ollama', 'ps'], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')[1:]  # Skip header
                models = []
                for line in lines:
                    if line.strip():
                        parts = line.split()
                        if len(parts) >= 4:
                            models.append({
                                'name': parts[0],
                                'size': parts[2],
                                'processor': parts[3] if len(parts) > 3 else 'Unknown'
                            })
                return models
        except Exception as e:
            logger.warning(f"Error getting Ollama models: {e}")

        return []

    def take_measurement(self, stage: str) -> Dict[str, Any]:
        """Take a complete system measurement."""
        measurement = {
            'timestamp': time.time(),
            'stage': stage,
            'gpu_stats': self.get_gpu_stats(),
            'system_stats': self.get_system_stats(),
            'ollama_models': self.get_ollama_models()
        }

        self.measurements.append(measurement)
        return measurement

    def get_test_papers(self, count: int = 3) -> List[Dict]:
        """Get papers for testing."""
        try:
            with database_manager.get_connection() as conn:
                cursor = conn.cursor()

                # Get papers with good abstracts that haven't been processed
                cursor.execute("""
                    SELECT pmid, title, abstract, LENGTH(abstract) as abstract_length
                    FROM papers
                    WHERE processing_status = 'pending'
                    AND abstract IS NOT NULL
                    AND LENGTH(abstract) > 200
                    ORDER BY LENGTH(abstract) DESC
                    LIMIT ?
                """, (count,))

                papers = []
                for row in cursor.fetchall():
                    papers.append({
                        'pmid': row[0],
                        'title': row[1],
                        'abstract': row[2],
                        'abstract_length': row[3]
                    })

                return papers
        except Exception as e:
            logger.error(f"Error getting test papers: {e}")
            return []

    def run_performance_test(self, paper_count: int = 3) -> Dict[str, Any]:
        """Run complete performance test."""
        print(f"\n{'='*60}")
        print("GPU PERFORMANCE TEST")
        print(f"{'='*60}")

        # Get test papers
        test_papers = self.get_test_papers(paper_count)
        if not test_papers:
            print("No suitable test papers found")
            return {'error': 'No test papers available'}

        print(f"Testing with {len(test_papers)} papers:")
        for i, paper in enumerate(test_papers, 1):
            print(f"   {i}. {paper['pmid']}: {paper['title'][:60]}... ({paper['abstract_length']} chars)")

        # Take baseline measurement
        print(f"\nTaking baseline measurements...")
        baseline = self.take_measurement('baseline')
        print(f"   GPU: {baseline['gpu_stats']['gpu_utilization']}% util, "
              f"{baseline['gpu_stats']['gpu_memory_percent']:.1f}% memory")
        print(f"   CPU: {baseline['system_stats']['cpu_percent']:.1f}%, "
              f"RAM: {baseline['system_stats']['ram_percent']:.1f}%")

        # Show loaded models
        models = baseline['ollama_models']
        if models:
            model_strs = [f"{m['name']} ({m['processor']})" for m in models]
            print(f"   Loaded models: {', '.join(model_strs)}")
        else:
            print(f"   No models currently loaded")

        # Start processing
        print(f"\nStarting LLM processing...")
        start_time = time.time()

        # Take measurement at start of processing
        self.take_measurement('processing_start')

        try:
            # Process the papers
            results = self.pipeline.analyze_interventions(
                limit_papers=len(test_papers),
                batch_size=1  # Process one at a time for detailed monitoring
            )

            processing_time = time.time() - start_time

            # Take final measurement
            final = self.take_measurement('processing_complete')

            # Calculate performance metrics
            performance_metrics = {
                'total_processing_time': processing_time,
                'papers_processed': results.get('papers_processed', 0),
                'interventions_extracted': results.get('interventions_extracted', 0),
                'avg_time_per_paper': processing_time / max(results.get('papers_processed', 1), 1),
                'success_rate': results.get('success_rate', 0),
                'baseline_measurement': baseline,
                'final_measurement': final,
                'all_measurements': self.measurements
            }

            return performance_metrics

        except Exception as e:
            logger.error(f"Error during processing: {e}")
            return {'error': str(e), 'measurements': self.measurements}

    def print_performance_report(self, metrics: Dict[str, Any]):
        """Print detailed performance report."""
        if 'error' in metrics:
            print(f"\nTest failed: {metrics['error']}")
            return

        print(f"\n{'='*60}")
        print("PERFORMANCE TEST RESULTS")
        print(f"{'='*60}")

        # Processing results
        print(f"\nPROCESSING RESULTS:")
        print(f"   Total Time: {metrics['total_processing_time']:.1f}s")
        print(f"   Papers Processed: {metrics['papers_processed']}")
        print(f"   Interventions Extracted: {metrics['interventions_extracted']}")
        print(f"   Average Time/Paper: {metrics['avg_time_per_paper']:.1f}s")
        print(f"   Success Rate: {metrics['success_rate']:.1f}%")

        # Resource comparison
        baseline = metrics['baseline_measurement']
        final = metrics['final_measurement']

        print(f"\nRESOURCE UTILIZATION:")
        print(f"   GPU Utilization: {baseline['gpu_stats']['gpu_utilization']}% -> {final['gpu_stats']['gpu_utilization']}%")
        print(f"   GPU Memory: {baseline['gpu_stats']['gpu_memory_percent']:.1f}% -> {final['gpu_stats']['gpu_memory_percent']:.1f}%")
        print(f"   CPU Usage: {baseline['system_stats']['cpu_percent']:.1f}% -> {final['system_stats']['cpu_percent']:.1f}%")
        print(f"   RAM Usage: {baseline['system_stats']['ram_percent']:.1f}% -> {final['system_stats']['ram_percent']:.1f}%")

        # Performance assessment
        time_per_paper = metrics['avg_time_per_paper']
        if time_per_paper < 10:
            performance = "Excellent"
        elif time_per_paper < 20:
            performance = "Good"
        elif time_per_paper < 30:
            performance = "Acceptable"
        else:
            performance = "Slow"

        print(f"\nPERFORMANCE ASSESSMENT: {performance}")
        print(f"   {time_per_paper:.1f}s per paper")

        if metrics['papers_processed'] > 0:
            estimated_time_68_papers = (68 * time_per_paper) / 60
            print(f"   Estimated time for 68 pending papers: {estimated_time_68_papers:.1f} minutes")

        print(f"\n{'='*60}")


def main():
    """Run the GPU performance test."""
    monitor = GPUPerformanceMonitor()

    try:
        # Run test with 3 papers
        metrics = monitor.run_performance_test(paper_count=3)

        # Print detailed report
        monitor.print_performance_report(metrics)

        return 0 if 'error' not in metrics else 1

    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        return 130
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        logger.error(f"Unexpected error in performance test: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())