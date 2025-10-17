"""
Data Access Object (DAO) layer for database operations.

This package implements the DAO pattern, separating database operations by domain:
- PapersDAO: Paper CRUD operations
- InterventionsDAO: Intervention CRUD operations
- AnalyticsDAO: Data mining and analytics queries
- SchemaDAO: Table creation and migrations
"""

from .base_dao import BaseDAO
from .papers_dao import PapersDAO
from .interventions_dao import InterventionsDAO
from .analytics_dao import AnalyticsDAO
from .schema_dao import SchemaDAO

__all__ = [
    'BaseDAO',
    'PapersDAO',
    'InterventionsDAO',
    'AnalyticsDAO',
    'SchemaDAO'
]
