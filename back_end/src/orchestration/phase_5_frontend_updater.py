#!/usr/bin/env python3
"""
Phase 5 Frontend Data Export Orchestrator

Coordinates all frontend data exports as final step in the automated pipeline.
Runs after Phase 4b (Bayesian Scoring) to update frontend JSON files.

Pipeline Flow:
    Phase 4b (Bayesian Scoring) →
    Phase 5 (Frontend Export) →
    COMPLETED

Exports:
- interventions.json (table view data)
- network_graph.json (network visualization)
- mechanism_clusters.json (mechanism data)

Features:
- Session tracking (like other phases)
- Atomic writes with backups
- Post-export validation
- Statistics collection
- Error recovery

Usage:
    # Run complete Phase 5 export
    python -m back_end.src.orchestration.phase_5_frontend_updater

    # Check status
    python -m back_end.src.orchestration.phase_5_frontend_updater --status
"""

import sys
import argparse
import json
import time
import traceback
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

try:
    from ..data.config import config, setup_logging
    from ..phase_1_data_collection.database_manager import database_manager
    from ..phase_5_frontend_export.phase_5_table_view_exporter import TableViewExporter
    from ..phase_5_frontend_export.phase_5_network_viz_exporter import NetworkVizExporter
except ImportError:
    # Fallback for standalone execution
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent.parent))
    from back_end.src.data.config import config, setup_logging
    from back_end.src.phase_1_data_collection.database_manager import database_manager
    from back_end.src.phase_5_frontend_export.phase_5_table_view_exporter import TableViewExporter
    from back_end.src.phase_5_frontend_export.phase_5_network_viz_exporter import NetworkVizExporter

logger = setup_logging(__name__, 'phase_5_frontend_updater.log')


@dataclass
class Phase5Results:
    """Results from Phase 5 execution."""
    success: bool

    # Export completion flags
    table_view_completed: bool = False
    network_viz_completed: bool = False

    # Export statistics
    files_exported: int = 0
    table_view_size_mb: float = 0.0
    network_viz_size_mb: float = 0.0
    total_interventions: int = 0
    total_nodes: int = 0
    total_edges: int = 0

    # Validation
    validation_passed: bool = False
    validation_warnings: List[str] = None

    # Timing
    total_duration_seconds: float = 0.0
    error: Optional[str] = None


class Phase5FrontendExportOrchestrator:
    """
    Phase 5 orchestrator - coordinates all frontend data exports.

    Final step in the automated pipeline (Phase 1 → 2 → 3 → 4 → 5).
    """

    def __init__(self, db_path: Optional[str] = None, config_path: Optional[str] = None):
        """
        Initialize Phase 5 orchestrator.

        Args:
            db_path: Path to intervention_research.db (defaults to config.db_path)
            config_path: Path to phase_5_config.yaml (optional)
        """
        self.db_path = Path(db_path) if db_path else Path(config.db_path)
        self.config_path = config_path

        # Initialize exporters (lazy loaded)
        self.table_view_exporter = None
        self.network_viz_exporter = None

        logger.info("Phase 5 Frontend Export Orchestrator initialized")
        logger.info(f"Database: {self.db_path}")

    def run(self, skip_table_view: bool = False, skip_network_viz: bool = False) -> Phase5Results:
        """
        Run complete Phase 5 pipeline.

        Args:
            skip_table_view: Skip table view export
            skip_network_viz: Skip network visualization export

        Returns:
            Phase5Results with execution statistics
        """
        logger.info("="*60)
        logger.info("PHASE 5: FRONTEND DATA EXPORT")
        logger.info("="*60)
        logger.info("Export 1: Table View Data (interventions.json)")
        logger.info("Export 2: Network Visualization (network_graph.json)")
        logger.info("="*60)

        start_time = time.time()
        results = Phase5Results(success=False, validation_warnings=[])
        session_id = self._create_export_session()

        try:
            # Export 1: Table View Data
            if not skip_table_view:
                table_view_result = self._run_table_view_export()
                results.table_view_completed = table_view_result['success']

                if table_view_result['success']:
                    results.table_view_size_mb = table_view_result['statistics']['file_size_mb']
                    results.total_interventions = table_view_result['statistics'].get('records_processed', 0)
                    results.files_exported += 1

                    # Collect validation warnings
                    if table_view_result.get('validation', {}).get('warnings'):
                        results.validation_warnings.extend(table_view_result['validation']['warnings'])
                else:
                    results.error = f"Table view export failed: {table_view_result.get('error', 'Unknown error')}"
                    self._update_export_session(session_id, 'failed', results)
                    return results

            # Export 2: Network Visualization
            if not skip_network_viz:
                network_viz_result = self._run_network_viz_export()
                results.network_viz_completed = network_viz_result['success']

                if network_viz_result['success']:
                    results.network_viz_size_mb = network_viz_result['statistics']['file_size_mb']

                    # Extract node and edge counts from validation stats
                    validation_stats = network_viz_result.get('validation', {}).get('statistics', {})
                    results.total_nodes = validation_stats.get('node_count', 0)
                    results.total_edges = validation_stats.get('link_count', 0)
                    results.files_exported += 1

                    # Collect validation warnings
                    if network_viz_result.get('validation', {}).get('warnings'):
                        results.validation_warnings.extend(network_viz_result['validation']['warnings'])
                else:
                    results.error = f"Network viz export failed: {network_viz_result.get('error', 'Unknown error')}"
                    self._update_export_session(session_id, 'failed', results)
                    return results

            # Success!
            results.success = True
            results.validation_passed = len(results.validation_warnings) == 0
            results.total_duration_seconds = time.time() - start_time

            logger.info("\n" + "="*60)
            logger.info("PHASE 5 COMPLETED SUCCESSFULLY")
            logger.info("="*60)
            logger.info(f"Total duration: {results.total_duration_seconds:.1f}s")
            logger.info(f"Files exported: {results.files_exported}")
            logger.info(f"Table view: {results.table_view_size_mb:.2f} MB")
            logger.info(f"Network viz: {results.network_viz_size_mb:.2f} MB")
            logger.info(f"Total interventions: {results.total_interventions}")
            logger.info(f"Knowledge graph nodes: {results.total_nodes}")
            logger.info(f"Knowledge graph edges: {results.total_edges}")

            if results.validation_warnings:
                logger.warning(f"Validation warnings: {len(results.validation_warnings)}")
                for warning in results.validation_warnings[:5]:  # Show first 5
                    logger.warning(f"  - {warning}")

            # Update session
            self._update_export_session(session_id, 'completed', results)

            return results

        except Exception as e:
            logger.error(f"Phase 5 pipeline failed: {e}")
            logger.error(traceback.format_exc())
            results.error = str(e)
            results.total_duration_seconds = time.time() - start_time
            self._update_export_session(session_id, 'failed', results)
            return results

    def _run_table_view_export(self) -> Dict[str, Any]:
        """
        Run table view data export.

        Exports interventions.json for frontend DataTables display.

        Returns:
            Dictionary with export results
        """
        logger.info("\n" + "="*60)
        logger.info("EXPORT 1: TABLE VIEW DATA")
        logger.info("="*60)

        try:
            # Initialize exporter
            self.table_view_exporter = TableViewExporter(
                db_path=str(self.db_path),
                config_path=self.config_path
            )

            # Run export
            result = self.table_view_exporter.run()

            if result['success']:
                logger.info(f"[SUCCESS] Table view export completed")
                logger.info(f"  Output: {result['output_path']}")
                logger.info(f"  Records: {result['statistics']['records_processed']}")
                logger.info(f"  Size: {result['statistics']['file_size_mb']:.2f} MB")
            else:
                logger.error(f"[FAILED] Table view export failed: {result.get('error')}")

            return result

        except Exception as e:
            logger.error(f"Table view export failed: {e}")
            logger.error(traceback.format_exc())
            return {
                'success': False,
                'error': str(e)
            }

    def _run_network_viz_export(self) -> Dict[str, Any]:
        """
        Run network visualization data export.

        Exports network_graph.json for D3.js visualization.

        Returns:
            Dictionary with export results
        """
        logger.info("\n" + "="*60)
        logger.info("EXPORT 2: NETWORK VISUALIZATION DATA")
        logger.info("="*60)

        try:
            # Initialize exporter
            self.network_viz_exporter = NetworkVizExporter(
                db_path=str(self.db_path),
                config_path=self.config_path
            )

            # Run export
            result = self.network_viz_exporter.run()

            if result['success']:
                logger.info(f"[SUCCESS] Network viz export completed")
                logger.info(f"  Output: {result['output_path']}")
                logger.info(f"  Records: {result['statistics']['records_processed']}")
                logger.info(f"  Size: {result['statistics']['file_size_mb']:.2f} MB")
            else:
                logger.error(f"[FAILED] Network viz export failed: {result.get('error')}")

            return result

        except Exception as e:
            logger.error(f"Network viz export failed: {e}")
            logger.error(traceback.format_exc())
            return {
                'success': False,
                'error': str(e)
            }

    def _create_export_session(self) -> str:
        """Create export session record in database."""
        session_id = f"export_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        try:
            with database_manager.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO frontend_export_sessions
                    (session_id, status, start_time)
                    VALUES (?, ?, ?)
                """, (session_id, 'running', datetime.now().isoformat()))
                conn.commit()

            logger.debug(f"Created export session: {session_id}")
        except Exception as e:
            logger.warning(f"Failed to create export session: {e}")

        return session_id

    def _update_export_session(self, session_id: str, status: str, results: Phase5Results):
        """Update export session with results."""
        try:
            with database_manager.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE frontend_export_sessions
                    SET status = ?,
                        end_time = ?,
                        files_exported = ?,
                        table_view_size_kb = ?,
                        network_viz_size_kb = ?,
                        validation_passed = ?,
                        error_message = ?
                    WHERE session_id = ?
                """, (
                    status,
                    datetime.now().isoformat(),
                    results.files_exported,
                    int(results.table_view_size_mb * 1024),
                    int(results.network_viz_size_mb * 1024),
                    results.validation_passed,
                    results.error,
                    session_id
                ))
                conn.commit()

            logger.debug(f"Updated export session: {session_id} (status: {status})")
        except Exception as e:
            logger.warning(f"Failed to update export session: {e}")

    def get_status(self) -> Dict[str, Any]:
        """
        Get Phase 5 export status.

        Returns:
            Dictionary with export statistics and last session info
        """
        try:
            with database_manager.get_connection() as conn:
                cursor = conn.cursor()

                # Get last export session
                cursor.execute("""
                    SELECT session_id, status, start_time, end_time,
                           files_exported, validation_passed
                    FROM frontend_export_sessions
                    ORDER BY start_time DESC
                    LIMIT 1
                """)
                last_session = cursor.fetchone()

                if last_session:
                    last_session_info = {
                        'session_id': last_session['session_id'],
                        'status': last_session['status'],
                        'start_time': last_session['start_time'],
                        'end_time': last_session['end_time'],
                        'files_exported': last_session['files_exported'],
                        'validation_passed': last_session['validation_passed']
                    }
                else:
                    last_session_info = None

                # Check if export files exist
                project_root = Path(__file__).parent.parent.parent.parent
                table_view_path = project_root / "frontend" / "data" / "interventions.json"
                network_viz_path = project_root / "frontend" / "data" / "network_graph.json"

                return {
                    'phase_5_completed': last_session_info is not None and last_session_info['status'] == 'completed',
                    'last_session': last_session_info,
                    'export_files': {
                        'table_view_exists': table_view_path.exists(),
                        'network_viz_exists': network_viz_path.exists()
                    }
                }

        except Exception as e:
            logger.error(f"Error getting Phase 5 status: {e}")
            return {
                'error': str(e),
                'phase_5_completed': False
            }


def main():
    """Command line interface for Phase 5 Frontend Export."""
    parser = argparse.ArgumentParser(
        description="Phase 5 Frontend Data Export Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run complete Phase 5 export
  python phase_5_frontend_updater.py

  # Skip table view export
  python phase_5_frontend_updater.py --skip-table-view

  # Skip network viz export
  python phase_5_frontend_updater.py --skip-network-viz

  # Check status
  python phase_5_frontend_updater.py --status
        """
    )

    parser.add_argument('--db-path', type=str, help='Path to intervention_research.db')
    parser.add_argument('--config', type=str, help='Path to phase_5_config.yaml')
    parser.add_argument('--skip-table-view', action='store_true', help='Skip table view export')
    parser.add_argument('--skip-network-viz', action='store_true', help='Skip network viz export')
    parser.add_argument('--status', action='store_true', help='Show Phase 5 status')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')

    args = parser.parse_args()

    try:
        # Initialize orchestrator
        orchestrator = Phase5FrontendExportOrchestrator(
            db_path=args.db_path,
            config_path=args.config
        )

        # Handle status check
        if args.status:
            status = orchestrator.get_status()
            print(json.dumps(status, indent=2))
            return

        # Run Phase 5
        results = orchestrator.run(
            skip_table_view=args.skip_table_view,
            skip_network_viz=args.skip_network_viz
        )

        # Print summary
        if results.success:
            print("\n[SUCCESS] Phase 5 completed successfully")
            print(f"Total duration: {results.total_duration_seconds:.1f}s")
            print(f"\nExports:")
            if results.table_view_completed:
                print(f"  Table view: {results.table_view_size_mb:.2f} MB ({results.total_interventions} interventions)")
            if results.network_viz_completed:
                print(f"  Network viz: {results.network_viz_size_mb:.2f} MB ({results.total_nodes} nodes, {results.total_edges} edges)")

            if results.validation_warnings:
                print(f"\nValidation warnings: {len(results.validation_warnings)}")
        else:
            print(f"\n[FAILED] Phase 5 failed: {results.error}")
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Phase 5 interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Phase 5 failed: {e}")
        if args.verbose:
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
