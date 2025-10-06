"""
Data Exporter
Exports intervention names from the database for semantic normalization experiments.
"""

import sqlite3
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any
import yaml


class InterventionDataExporter:
    """Export intervention names from database with metadata."""

    def __init__(self, config_path: str = None):
        """Initialize exporter with configuration."""
        if config_path is None:
            config_path = Path(__file__).parent.parent / "config" / "config.yaml"

        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.db_path = Path(self.config['database']['path'])
        self.export_limit = self.config['database']['export_limit']
        self.min_frequency = self.config['export']['min_frequency']
        self.output_dir = Path(self.config['export']['output_dir'])

        # Setup logging
        logging.basicConfig(
            level=self.config['logging']['level'],
            format=self.config['logging']['format']
        )
        self.logger = logging.getLogger(__name__)

    def export_interventions(self) -> Dict[str, Any]:
        """
        Export unique intervention names with metadata from database.

        Returns:
            Dict containing exported data and metadata
        """
        self.logger.info(f"Connecting to database: {self.db_path}")

        if not self.db_path.exists():
            raise FileNotFoundError(f"Database not found: {self.db_path}")

        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Query unique intervention names with frequency and health conditions
        query = """
        SELECT
            intervention_name,
            health_condition,
            COUNT(*) as frequency,
            intervention_category,
            GROUP_CONCAT(DISTINCT correlation_type) as correlation_types,
            AVG(sample_size) as avg_sample_size
        FROM interventions
        WHERE intervention_name IS NOT NULL
        AND intervention_name != ''
        GROUP BY intervention_name, health_condition
        HAVING COUNT(*) >= ?
        ORDER BY frequency DESC
        LIMIT ?
        """

        self.logger.info(f"Querying interventions (min_frequency={self.min_frequency}, limit={self.export_limit})")
        cursor.execute(query, (self.min_frequency, self.export_limit))
        rows = cursor.fetchall()

        # Convert to structured format
        interventions = []
        unique_names = set()

        for row in rows:
            intervention_data = {
                "intervention_name": row["intervention_name"],
                "health_condition": row["health_condition"],
                "frequency": row["frequency"],
                "category": row["intervention_category"],
                "correlation_types": row["correlation_types"],
                "avg_sample_size": int(row["avg_sample_size"]) if row["avg_sample_size"] else None
            }
            interventions.append(intervention_data)
            unique_names.add(row["intervention_name"])

        conn.close()

        # Create export package
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        export_data = {
            "metadata": {
                "export_timestamp": timestamp,
                "total_records": len(interventions),
                "unique_intervention_names": len(unique_names),
                "min_frequency": self.min_frequency,
                "export_limit": self.export_limit,
                "source_database": str(self.db_path)
            },
            "interventions": interventions,
            "unique_names": sorted(list(unique_names))
        }

        # Save to file
        self.output_dir.mkdir(parents=True, exist_ok=True)
        output_file = self.output_dir / f"interventions_export_{timestamp}.json"

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)

        self.logger.info(f"Exported {len(interventions)} intervention records to {output_file}")
        self.logger.info(f"Unique intervention names: {len(unique_names)}")

        return export_data

    def get_latest_export(self) -> Dict[str, Any]:
        """Load the most recent export file."""
        export_files = list(self.output_dir.glob("interventions_export_*.json"))

        if not export_files:
            raise FileNotFoundError(f"No export files found in {self.output_dir}")

        latest_file = max(export_files, key=lambda p: p.stat().st_mtime)

        with open(latest_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.logger.info(f"Loaded export from {latest_file}")
        return data


def main():
    """CLI entry point for data export."""
    exporter = InterventionDataExporter()
    export_data = exporter.export_interventions()

    print("\n" + "="*60)
    print("INTERVENTION DATA EXPORT COMPLETE")
    print("="*60)
    print(f"Total records exported: {export_data['metadata']['total_records']}")
    print(f"Unique intervention names: {export_data['metadata']['unique_intervention_names']}")
    print(f"Export timestamp: {export_data['metadata']['export_timestamp']}")
    print("="*60 + "\n")

    # Show sample
    print("Sample interventions:")
    for i, intervention in enumerate(export_data['interventions'][:10], 1):
        print(f"{i}. {intervention['intervention_name']} (freq: {intervention['frequency']}, condition: {intervention['health_condition']})")

    if len(export_data['interventions']) > 10:
        print(f"... and {len(export_data['interventions']) - 10} more")


if __name__ == "__main__":
    main()
