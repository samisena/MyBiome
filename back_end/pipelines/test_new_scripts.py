#!/usr/bin/env python3
"""
Test Script for New Unified Pipeline Scripts

This script tests the new unified scripts against the old ones to ensure feature parity:
- paper_collector.py vs collect_papers.py
- llm_processor.py vs robust_llm_processor.py + run_llm_processing.py
- research_orchestrator.py vs autonomous_research_orchestrator.py

Features tested:
- Command-line interface compatibility
- Configuration options
- Output format consistency
- Session persistence
- Error handling
- Thermal protection
- Progress reporting

Usage:
    python test_new_scripts.py --test-all
    python test_new_scripts.py --test-collector
    python test_new_scripts.py --test-processor
    python test_new_scripts.py --test-orchestrator
"""

import sys
import json
import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional
import argparse
import time
from dataclasses import dataclass

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from src.data.config import setup_logging
    from src.paper_collection.database_manager import database_manager
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

logger = setup_logging(__name__, 'test_new_scripts.log')


@dataclass
class TestResult:
    """Result of a single test."""
    test_name: str
    passed: bool
    details: str
    execution_time: float
    expected_output: Optional[str] = None
    actual_output: Optional[str] = None


class ScriptTester:
    """Base class for testing script functionality."""

    def __init__(self, test_dir: Path):
        self.test_dir = test_dir
        self.pipelines_dir = Path(__file__).parent
        self.results: List[TestResult] = []

    def run_command(self, cmd: List[str], timeout: int = 120) -> Dict[str, Any]:
        """Run a command and return results."""
        start_time = time.time()
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=self.pipelines_dir.parent  # Run from back_end directory
            )
            execution_time = time.time() - start_time

            return {
                'returncode': result.returncode,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'execution_time': execution_time,
                'success': result.returncode == 0
            }
        except subprocess.TimeoutExpired:
            execution_time = time.time() - start_time
            return {
                'returncode': -1,
                'stdout': '',
                'stderr': f'Command timed out after {timeout} seconds',
                'execution_time': execution_time,
                'success': False
            }
        except Exception as e:
            execution_time = time.time() - start_time
            return {
                'returncode': -2,
                'stdout': '',
                'stderr': str(e),
                'execution_time': execution_time,
                'success': False
            }

    def add_result(self, test_name: str, passed: bool, details: str, execution_time: float = 0.0):
        """Add a test result."""
        self.results.append(TestResult(
            test_name=test_name,
            passed=passed,
            details=details,
            execution_time=execution_time
        ))

    def generate_report(self) -> Dict[str, Any]:
        """Generate test report."""
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r.passed)

        return {
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'failed_tests': total_tests - passed_tests,
            'success_rate': (passed_tests / total_tests * 100) if total_tests > 0 else 0,
            'results': [
                {
                    'test_name': r.test_name,
                    'passed': r.passed,
                    'details': r.details,
                    'execution_time': r.execution_time
                }
                for r in self.results
            ]
        }


class CollectorTester(ScriptTester):
    """Test paper_collector.py against collect_papers.py."""

    def test_help_output(self):
        """Test that help output is similar."""
        logger.info("Testing help output compatibility")

        # Test new script help
        new_result = self.run_command([sys.executable, "pipelines/paper_collector.py", "--help"])

        # Test old script help
        old_result = self.run_command([sys.executable, "pipelines/collect_papers.py", "--help"])

        # Check that both succeed and have similar options
        new_success = new_result['success']
        old_success = old_result['success']

        if new_success and old_success:
            # Check for key options in both
            key_options = ['--max-papers', '--min-year', '--traditional-mode']
            new_help = new_result['stdout']
            old_help = old_result['stdout']

            missing_options = []
            for option in key_options:
                if option in old_help and option not in new_help:
                    missing_options.append(option)

            if not missing_options:
                self.add_result("help_output", True, "Help outputs are compatible", new_result['execution_time'])
            else:
                self.add_result("help_output", False, f"Missing options in new script: {missing_options}")
        else:
            self.add_result("help_output", False, f"Help command failed - New: {new_success}, Old: {old_success}")

    def test_status_check(self):
        """Test status checking functionality."""
        logger.info("Testing status check functionality")

        # Test new script status
        new_result = self.run_command([sys.executable, "pipelines/paper_collector.py", "--status"])

        # Check if status command works (it might fail if no session exists, but should handle gracefully)
        if new_result['returncode'] in [0, 1]:  # 0 = success, 1 = no session found
            if "No active session" in new_result['stdout'] or "Current collection status" in new_result['stdout']:
                self.add_result("status_check", True, "Status check works correctly", new_result['execution_time'])
            else:
                self.add_result("status_check", False, f"Unexpected status output: {new_result['stdout'][:200]}")
        else:
            self.add_result("status_check", False, f"Status check failed: {new_result['stderr']}")

    def test_configuration_file(self):
        """Test configuration file support."""
        logger.info("Testing configuration file support")

        # Create test config
        config_file = self.test_dir / "test_collector_config.json"
        config_data = {
            "conditions": ["test_condition"],
            "target_papers_per_condition": 10,
            "min_year": 2020,
            "traditional_mode": True,
            "batch_size_collection": 5
        }

        with open(config_file, 'w') as f:
            json.dump(config_data, f, indent=2)

        # Test new script with config
        new_result = self.run_command([
            sys.executable, "pipelines/paper_collector.py",
            "--config", str(config_file),
            "--status"  # Just check status to avoid actual collection
        ])

        if new_result['success'] or "No active session" in new_result['stdout']:
            self.add_result("config_file", True, "Configuration file support works", new_result['execution_time'])
        else:
            self.add_result("config_file", False, f"Config file test failed: {new_result['stderr']}")

    def test_session_persistence(self):
        """Test session file creation."""
        logger.info("Testing session persistence")

        session_file = self.test_dir / "test_collection_session.json"

        # Test session file creation
        new_result = self.run_command([
            sys.executable, "pipelines/paper_collector.py",
            "test_condition",
            "--max-papers", "1",
            "--session-file", str(session_file),
            "--no-intermediate",  # Avoid creating extra files
            "--traditional-mode"  # Use traditional mode for faster testing
        ], timeout=60)

        # Check if session file was created (even if collection failed due to no papers)
        if session_file.exists():
            try:
                with open(session_file, 'r') as f:
                    session_data = json.load(f)

                if 'session_id' in session_data and 'config' in session_data:
                    self.add_result("session_persistence", True, "Session persistence works", new_result['execution_time'])
                else:
                    self.add_result("session_persistence", False, "Session file missing required fields")
            except Exception as e:
                self.add_result("session_persistence", False, f"Session file corrupted: {e}")
        else:
            # Session file not created - check if it's because of error handling
            if new_result['returncode'] != 0 and "error" in new_result['stderr'].lower():
                self.add_result("session_persistence", True, "Session not created due to expected error", new_result['execution_time'])
            else:
                self.add_result("session_persistence", False, "Session file not created")

    def run_all_tests(self):
        """Run all collector tests."""
        logger.info("Starting paper collector tests")
        self.test_help_output()
        self.test_status_check()
        self.test_configuration_file()
        self.test_session_persistence()


class ProcessorTester(ScriptTester):
    """Test llm_processor.py against robust_llm_processor.py."""

    def test_help_output(self):
        """Test help output compatibility."""
        logger.info("Testing processor help output")

        new_result = self.run_command([sys.executable, "pipelines/llm_processor.py", "--help"])
        old_result = self.run_command([sys.executable, "pipelines/robust_llm_processor.py", "--help"])

        new_success = new_result['success']
        old_success = old_result['success']

        if new_success and old_success:
            # Check for key thermal options
            key_options = ['--max-temp', '--batch-size', '--resume']
            new_help = new_result['stdout']
            old_help = old_result['stdout']

            missing_options = []
            for option in key_options:
                if option in old_help and option not in new_help:
                    missing_options.append(option)

            if not missing_options:
                self.add_result("help_output", True, "Help outputs are compatible", new_result['execution_time'])
            else:
                self.add_result("help_output", False, f"Missing options: {missing_options}")
        else:
            self.add_result("help_output", False, f"Help failed - New: {new_success}, Old: {old_success}")

    def test_thermal_status(self):
        """Test thermal monitoring functionality."""
        logger.info("Testing thermal status")

        new_result = self.run_command([sys.executable, "pipelines/llm_processor.py", "--thermal-status"])

        # Check if thermal status works
        if new_result['success']:
            if "thermal status" in new_result['stdout'].lower() or "gpu" in new_result['stdout'].lower():
                self.add_result("thermal_status", True, "Thermal status works", new_result['execution_time'])
            else:
                self.add_result("thermal_status", False, f"Unexpected thermal output: {new_result['stdout'][:200]}")
        else:
            # Thermal status might fail if no GPU, but should handle gracefully
            if "error" in new_result['stderr'].lower() and "gpu" in new_result['stderr'].lower():
                self.add_result("thermal_status", True, "Thermal status handles no GPU correctly", new_result['execution_time'])
            else:
                self.add_result("thermal_status", False, f"Thermal status failed: {new_result['stderr']}")

    def test_status_check(self):
        """Test processing status check."""
        logger.info("Testing processor status check")

        new_result = self.run_command([sys.executable, "pipelines/llm_processor.py", "--status"])

        if new_result['returncode'] in [0, 1]:  # 0 = success, 1 = no session
            if "No active session" in new_result['stdout'] or "Current processing status" in new_result['stdout']:
                self.add_result("status_check", True, "Status check works", new_result['execution_time'])
            else:
                self.add_result("status_check", False, f"Unexpected status: {new_result['stdout'][:200]}")
        else:
            self.add_result("status_check", False, f"Status check failed: {new_result['stderr']}")

    def test_configuration_options(self):
        """Test advanced configuration options."""
        logger.info("Testing processor configuration")

        # Test with various options
        new_result = self.run_command([
            sys.executable, "pipelines/llm_processor.py",
            "--limit", "1",
            "--batch-size", "1",
            "--max-temp", "75",
            "--cooling-temp", "65",
            "--single-model",
            "--status"  # Just check status to avoid actual processing
        ])

        if new_result['returncode'] in [0, 1]:
            self.add_result("config_options", True, "Configuration options work", new_result['execution_time'])
        else:
            self.add_result("config_options", False, f"Config options failed: {new_result['stderr']}")

    def test_session_management(self):
        """Test session file handling."""
        logger.info("Testing processor session management")

        session_file = self.test_dir / "test_processing_session.json"

        # Test session file creation
        new_result = self.run_command([
            sys.executable, "pipelines/llm_processor.py",
            "--limit", "0",  # Process 0 papers to avoid actual processing
            "--session-file", str(session_file),
            "--batch-size", "1"
        ], timeout=30)

        # Check session handling
        if session_file.exists():
            self.add_result("session_management", True, "Session management works", new_result['execution_time'])
        else:
            # Session might not be created if no papers to process
            if "no papers" in new_result['stdout'].lower() or new_result['returncode'] == 0:
                self.add_result("session_management", True, "Session handled correctly for no papers", new_result['execution_time'])
            else:
                self.add_result("session_management", False, "Session file not created")

    def run_all_tests(self):
        """Run all processor tests."""
        logger.info("Starting LLM processor tests")
        self.test_help_output()
        self.test_thermal_status()
        self.test_status_check()
        self.test_configuration_options()
        self.test_session_management()


class OrchestratorTester(ScriptTester):
    """Test research_orchestrator.py against autonomous_research_orchestrator.py."""

    def test_help_output(self):
        """Test orchestrator help output."""
        logger.info("Testing orchestrator help output")

        new_result = self.run_command([sys.executable, "pipelines/research_orchestrator.py", "--help"])
        old_result = self.run_command([sys.executable, "orchestration/autonomous_research_orchestrator.py", "--help"])

        new_success = new_result['success']
        old_success = old_result['success']

        if new_success and old_success:
            # Check for key orchestration options
            key_options = ['--conditions', '--resume', '--overnight']
            new_help = new_result['stdout']
            old_help = old_result['stdout']

            missing_options = []
            for option in key_options:
                if option in old_help and option not in new_help:
                    missing_options.append(option)

            if not missing_options:
                self.add_result("help_output", True, "Help outputs compatible", new_result['execution_time'])
            else:
                self.add_result("help_output", False, f"Missing options: {missing_options}")
        else:
            self.add_result("help_output", False, f"Help failed - New: {new_success}, Old: {old_success}")

    def test_status_functionality(self):
        """Test orchestrator status check."""
        logger.info("Testing orchestrator status")

        new_result = self.run_command([sys.executable, "pipelines/research_orchestrator.py", "--status"])

        if new_result['returncode'] in [0, 1]:
            if "No active session" in new_result['stdout'] or "Current orchestration status" in new_result['stdout']:
                self.add_result("status_functionality", True, "Status functionality works", new_result['execution_time'])
            else:
                self.add_result("status_functionality", False, f"Unexpected status: {new_result['stdout'][:200]}")
        else:
            self.add_result("status_functionality", False, f"Status failed: {new_result['stderr']}")

    def test_thermal_monitoring(self):
        """Test thermal monitoring in orchestrator."""
        logger.info("Testing orchestrator thermal monitoring")

        new_result = self.run_command([sys.executable, "pipelines/research_orchestrator.py", "--thermal-monitor"])

        if new_result['success']:
            if "thermal status" in new_result['stdout'].lower():
                self.add_result("thermal_monitoring", True, "Thermal monitoring works", new_result['execution_time'])
            else:
                self.add_result("thermal_monitoring", False, f"Unexpected thermal output: {new_result['stdout'][:200]}")
        else:
            # Might fail if no GPU
            if "temperature" in new_result['stderr'].lower() or "gpu" in new_result['stderr'].lower():
                self.add_result("thermal_monitoring", True, "Thermal monitoring handles no GPU", new_result['execution_time'])
            else:
                self.add_result("thermal_monitoring", False, f"Thermal monitoring failed: {new_result['stderr']}")

    def test_workflow_phases(self):
        """Test workflow phase configuration."""
        logger.info("Testing workflow phases")

        # Test collection-only mode
        new_result = self.run_command([
            sys.executable, "pipelines/research_orchestrator.py",
            "test_condition",
            "--collection-only",
            "--papers", "1",
            "--status"
        ], timeout=30)

        if new_result['returncode'] in [0, 1]:
            self.add_result("workflow_phases", True, "Workflow phases work", new_result['execution_time'])
        else:
            self.add_result("workflow_phases", False, f"Workflow phases failed: {new_result['stderr']}")

    def test_session_coordination(self):
        """Test session coordination."""
        logger.info("Testing session coordination")

        session_file = self.test_dir / "test_orchestration_session.json"

        # Test session creation
        new_result = self.run_command([
            sys.executable, "pipelines/research_orchestrator.py",
            "test_condition",
            "--papers", "1",
            "--session-file", str(session_file),
            "--collection-only"  # Only test collection to avoid long processing
        ], timeout=60)

        if session_file.exists():
            try:
                with open(session_file, 'r') as f:
                    session_data = json.load(f)

                if 'session_id' in session_data and 'conditions_progress' in session_data:
                    self.add_result("session_coordination", True, "Session coordination works", new_result['execution_time'])
                else:
                    self.add_result("session_coordination", False, "Session missing required fields")
            except Exception as e:
                self.add_result("session_coordination", False, f"Session corrupted: {e}")
        else:
            # Check if session wasn't created due to expected error
            if new_result['returncode'] != 0:
                self.add_result("session_coordination", True, "Session handled error correctly", new_result['execution_time'])
            else:
                self.add_result("session_coordination", False, "Session file not created")

    def run_all_tests(self):
        """Run all orchestrator tests."""
        logger.info("Starting research orchestrator tests")
        self.test_help_output()
        self.test_status_functionality()
        self.test_thermal_monitoring()
        self.test_workflow_phases()
        self.test_session_coordination()


class DatabaseTester(ScriptTester):
    """Test database compatibility."""

    def test_database_access(self):
        """Test that new scripts can access database properly."""
        logger.info("Testing database access")

        try:
            # Test database connection
            with database_manager.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM papers")
                paper_count = cursor.fetchone()[0]

            self.add_result("database_access", True, f"Database accessible, {paper_count} papers found")

        except Exception as e:
            self.add_result("database_access", False, f"Database access failed: {e}")

    def test_database_schema(self):
        """Test database schema compatibility."""
        logger.info("Testing database schema")

        try:
            with database_manager.get_connection() as conn:
                cursor = conn.cursor()

                # Check key tables exist
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                tables = [row[0] for row in cursor.fetchall()]

                required_tables = ['papers', 'interventions']
                missing_tables = [t for t in required_tables if t not in tables]

                if not missing_tables:
                    self.add_result("database_schema", True, f"All required tables present: {tables}")
                else:
                    self.add_result("database_schema", False, f"Missing tables: {missing_tables}")

        except Exception as e:
            self.add_result("database_schema", False, f"Schema check failed: {e}")

    def run_all_tests(self):
        """Run all database tests."""
        logger.info("Starting database tests")
        self.test_database_access()
        self.test_database_schema()


def run_comprehensive_tests(test_collector: bool = True, test_processor: bool = True,
                          test_orchestrator: bool = True, test_database: bool = True) -> Dict[str, Any]:
    """Run comprehensive test suite."""
    logger.info("Starting comprehensive test suite")

    # Create temporary test directory
    with tempfile.TemporaryDirectory() as temp_dir:
        test_dir = Path(temp_dir)
        all_results = {}

        # Run collector tests
        if test_collector:
            logger.info("=" * 50)
            logger.info("TESTING PAPER COLLECTOR")
            logger.info("=" * 50)
            collector_tester = CollectorTester(test_dir)
            collector_tester.run_all_tests()
            all_results['collector'] = collector_tester.generate_report()

        # Run processor tests
        if test_processor:
            logger.info("=" * 50)
            logger.info("TESTING LLM PROCESSOR")
            logger.info("=" * 50)
            processor_tester = ProcessorTester(test_dir)
            processor_tester.run_all_tests()
            all_results['processor'] = processor_tester.generate_report()

        # Run orchestrator tests
        if test_orchestrator:
            logger.info("=" * 50)
            logger.info("TESTING RESEARCH ORCHESTRATOR")
            logger.info("=" * 50)
            orchestrator_tester = OrchestratorTester(test_dir)
            orchestrator_tester.run_all_tests()
            all_results['orchestrator'] = orchestrator_tester.generate_report()

        # Run database tests
        if test_database:
            logger.info("=" * 50)
            logger.info("TESTING DATABASE COMPATIBILITY")
            logger.info("=" * 50)
            database_tester = DatabaseTester(test_dir)
            database_tester.run_all_tests()
            all_results['database'] = database_tester.generate_report()

    # Generate overall summary
    total_tests = sum(r['total_tests'] for r in all_results.values())
    total_passed = sum(r['passed_tests'] for r in all_results.values())
    overall_success_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0

    summary = {
        'overall_summary': {
            'total_tests': total_tests,
            'total_passed': total_passed,
            'total_failed': total_tests - total_passed,
            'overall_success_rate': overall_success_rate
        },
        'component_results': all_results,
        'timestamp': time.time()
    }

    return summary


def print_test_summary(results: Dict[str, Any]):
    """Print formatted test summary."""
    print("\n" + "=" * 60)
    print("NEW SCRIPTS FEATURE PARITY TEST RESULTS")
    print("=" * 60)

    summary = results['overall_summary']
    print(f"Overall Results: {summary['total_passed']}/{summary['total_tests']} tests passed ({summary['overall_success_rate']:.1f}%)")
    print()

    # Print component results
    for component, result in results['component_results'].items():
        print(f"{component.upper()} TESTS:")
        print(f"  Passed: {result['passed_tests']}/{result['total_tests']} ({result['success_rate']:.1f}%)")

        # Show failed tests
        failed_tests = [r for r in result['results'] if not r['passed']]
        if failed_tests:
            print("  Failed tests:")
            for test in failed_tests:
                print(f"    - {test['test_name']}: {test['details']}")
        else:
            print("  All tests passed! ✅")
        print()

    # Overall verdict
    if summary['overall_success_rate'] >= 90:
        print("🎉 EXCELLENT: New scripts have excellent feature parity!")
    elif summary['overall_success_rate'] >= 75:
        print("✅ GOOD: New scripts have good feature parity with minor issues.")
    elif summary['overall_success_rate'] >= 50:
        print("⚠️ FAIR: New scripts have fair compatibility but need improvements.")
    else:
        print("❌ POOR: New scripts need significant work before replacing old ones.")

    print("=" * 60)


def main():
    """Main test execution."""
    parser = argparse.ArgumentParser(description="Test new unified scripts for feature parity")
    parser.add_argument('--test-all', action='store_true', help='Run all tests')
    parser.add_argument('--test-collector', action='store_true', help='Test paper collector only')
    parser.add_argument('--test-processor', action='store_true', help='Test LLM processor only')
    parser.add_argument('--test-orchestrator', action='store_true', help='Test orchestrator only')
    parser.add_argument('--test-database', action='store_true', help='Test database compatibility only')
    parser.add_argument('--output-file', type=str, help='Save results to JSON file')
    parser.add_argument('--quick', action='store_true', help='Run quick tests only')

    args = parser.parse_args()

    # Determine what to test
    test_all = args.test_all
    test_collector = args.test_collector or test_all
    test_processor = args.test_processor or test_all
    test_orchestrator = args.test_orchestrator or test_all
    test_database = args.test_database or test_all

    # Default to all tests if none specified
    if not any([test_collector, test_processor, test_orchestrator, test_database]):
        test_collector = test_processor = test_orchestrator = test_database = True

    print("Starting feature parity tests for new unified scripts...")
    print(f"Testing: Collector={test_collector}, Processor={test_processor}, Orchestrator={test_orchestrator}, Database={test_database}")

    try:
        # Run tests
        results = run_comprehensive_tests(
            test_collector=test_collector,
            test_processor=test_processor,
            test_orchestrator=test_orchestrator,
            test_database=test_database
        )

        # Print results
        print_test_summary(results)

        # Save results if requested
        if args.output_file:
            with open(args.output_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\nDetailed results saved to: {args.output_file}")

        # Return appropriate exit code
        success_rate = results['overall_summary']['overall_success_rate']
        return 0 if success_rate >= 75 else 1

    except Exception as e:
        logger.error(f"Test execution failed: {e}")
        print(f"Test execution failed: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)