"""
Base DAO (Data Access Object) with shared database connection handling.

All DAOs inherit from this class to get consistent connection management.
"""

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from back_end.src.data.config import setup_logging

logger = setup_logging(__name__, 'database.log')


class BaseDAO:
    """
    Base Data Access Object with thread-safe connection handling.

    All DAOs inherit from this class to ensure consistent database access patterns.
    """

    def __init__(self, db_path: Path):
        """
        Initialize DAO with database path.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path) if not isinstance(db_path, Path) else db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def get_connection(self):
        """
        Get a thread-safe database connection.

        This creates a fresh connection for each context, ensuring thread safety
        without the dangerous check_same_thread=False hack.

        Usage:
            with dao.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(...)
                # conn.commit() is called automatically on success
                # conn.rollback() is called automatically on error
        """
        # Create a fresh connection for this thread/context
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row  # Enable column access by name

        # Enable WAL mode for safe concurrent reads
        conn.execute('PRAGMA journal_mode = WAL')
        conn.execute('PRAGMA foreign_keys = ON')

        try:
            yield conn
            # Automatically commit on successful completion
            conn.commit()
        except Exception as e:
            # Automatically rollback on any error
            conn.rollback()
            logger.error(f"Database operation failed, rolled back: {e}")
            raise
        finally:
            # Always close the connection
            conn.close()

    def execute_query(self, query: str, params: tuple = ()) -> list:
        """
        Execute a SELECT query and return all rows.

        Args:
            query: SQL SELECT query
            params: Query parameters

        Returns:
            List of row dictionaries
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]

    def execute_single(self, query: str, params: tuple = ()) -> dict | None:
        """
        Execute a SELECT query and return a single row.

        Args:
            query: SQL SELECT query
            params: Query parameters

        Returns:
            Row dictionary or None if no results
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            row = cursor.fetchone()
            return dict(row) if row else None

    def execute_insert(self, query: str, params: tuple = ()) -> int:
        """
        Execute an INSERT query and return the last inserted row ID.

        Args:
            query: SQL INSERT query
            params: Query parameters

        Returns:
            Last inserted row ID
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            return cursor.lastrowid

    def execute_update(self, query: str, params: tuple = ()) -> int:
        """
        Execute an UPDATE or DELETE query and return rows affected.

        Args:
            query: SQL UPDATE/DELETE query
            params: Query parameters

        Returns:
            Number of rows affected
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            return cursor.rowcount

    def execute_batch(self, query: str, params_list: list) -> int:
        """
        Execute a batch INSERT/UPDATE using executemany.

        Args:
            query: SQL query with placeholders
            params_list: List of parameter tuples

        Returns:
            Number of rows affected
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.executemany(query, params_list)
            return cursor.rowcount

    def table_exists(self, table_name: str) -> bool:
        """
        Check if a table exists in the database.

        Args:
            table_name: Name of table to check

        Returns:
            True if table exists
        """
        row = self.execute_single(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (table_name,)
        )
        return row is not None

    def column_exists(self, table_name: str, column_name: str) -> bool:
        """
        Check if a column exists in a table.

        Args:
            table_name: Table to check
            column_name: Column to check

        Returns:
            True if column exists
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(f"PRAGMA table_info({table_name})")
            columns = [row[1] for row in cursor.fetchall()]
            return column_name in columns

    def add_column_if_missing(self, table_name: str, column_name: str, column_type: str) -> bool:
        """
        Add a column to a table if it doesn't exist.

        Args:
            table_name: Table to modify
            column_name: Column to add
            column_type: SQL column definition (e.g., 'TEXT', 'REAL')

        Returns:
            True if column was added, False if already exists
        """
        if self.column_exists(table_name, column_name):
            return False

        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(f'ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type}')
                logger.info(f"Added column {column_name} to {table_name}")
                return True
        except sqlite3.OperationalError as e:
            if "duplicate column name" in str(e):
                return False
            logger.error(f"Failed to add column {column_name} to {table_name}: {e}")
            raise
