"""
Phase 5: Frontend Data Export

Automated export of processed data to frontend JSON files.
Final step in the automated pipeline (Phase 1 → 2 → 3 → 4 → 5).

Modules:
- phase_5_table_view_exporter: Export interventions data for DataTables
- phase_5_network_viz_exporter: Export knowledge graph for D3.js visualization
- phase_5_export_operations: Shared utilities (atomic writes, validation, backups)
- phase_5_base_exporter: Base exporter class
"""

__version__ = "1.0.0"
