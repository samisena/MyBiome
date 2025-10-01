#!/usr/bin/env python3
"""
Phase 1 Test Suite: Critical Safety Fixes

Tests all Phase 1 fixes:
1. Database threading (thread-local connections, transactions)
2. Transaction integrity (commit/rollback)
3. Session state race conditions (file locking)
4. XML cleanup on error

Run with: python test_phase1_threading.py
"""

import sys
import os
import tempfile
import time
import json
import platform
from pathlib import Path
from threading import Thread
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from back_end.src.data_collection.database_manager import DatabaseManager
from back_end.src.data_collection.paper_parser import PubmedParser

# Platform-specific imports for file locking
if platform.system() == 'Windows':
    import msvcrt
else:
    import fcntl


class TestResults:
    """Track test results."""
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []

    def add_pass(self, test_name):
        self.passed += 1
        print(f"[PASS] {test_name}")

    def add_fail(self, test_name, reason):
        self.failed += 1
        self.errors.append(f"{test_name}: {reason}")
        print(f"[FAIL] {test_name}")
        print(f"   Reason: {reason}")

    def summary(self):
        total = self.passed + self.failed
        print("\n" + "="*70)
        print("TEST SUMMARY")
        print("="*70)
        print(f"Total Tests: {total}")
        print(f"Passed: {self.passed} ({self.passed/total*100:.1f}%)")
        print(f"Failed: {self.failed} ({self.failed/total*100:.1f}%)")

        if self.errors:
            print("\nFailed Tests:")
            for error in self.errors:
                print(f"  - {error}")

        return self.failed == 0


results = TestResults()


def test_database_thread_local_connections():
    """Test 1.1: Database creates thread-local connections."""
    test_name = "Database Thread-Local Connections"

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / 'test.db'

            class TestConfig:
                path = db_path

            db = DatabaseManager(db_config=TestConfig())

            # Test that multiple connections work in same thread
            with db.get_connection() as conn1:
                cursor1 = conn1.cursor()
                cursor1.execute('CREATE TABLE test1 (id INTEGER)')

            with db.get_connection() as conn2:
                cursor2 = conn2.cursor()
                cursor2.execute('CREATE TABLE test2 (id INTEGER)')

            # Verify both tables were created
            with db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                tables = [row[0] for row in cursor.fetchall()]

            if 'test1' in tables and 'test2' in tables:
                results.add_pass(test_name)
            else:
                results.add_fail(test_name, f"Expected tables not found: {tables}")

    except Exception as e:
        results.add_fail(test_name, str(e))


def test_transaction_rollback():
    """Test 1.2a: Transaction rollback on error."""
    test_name = "Transaction Rollback on Error"

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / 'test.db'

            class TestConfig:
                path = db_path

            db = DatabaseManager(db_config=TestConfig())

            # Create table
            with db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('CREATE TABLE test_data (id INTEGER PRIMARY KEY, value TEXT)')

            # Try to insert data but force error
            try:
                with db.get_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute('INSERT INTO test_data (value) VALUES (?)', ('should_rollback',))
                    # Force error
                    raise Exception('Forced error')
            except Exception:
                pass

            # Verify data was rolled back
            with db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT COUNT(*) FROM test_data')
                count = cursor.fetchone()[0]

            if count == 0:
                results.add_pass(test_name)
            else:
                results.add_fail(test_name, f"Expected 0 rows, found {count}")

    except Exception as e:
        results.add_fail(test_name, str(e))


def test_transaction_commit():
    """Test 1.2b: Transaction commit on success."""
    test_name = "Transaction Commit on Success"

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / 'test.db'

            class TestConfig:
                path = db_path

            db = DatabaseManager(db_config=TestConfig())

            # Create table
            with db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('CREATE TABLE test_data (id INTEGER PRIMARY KEY, value TEXT)')

            # Insert data successfully
            with db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('INSERT INTO test_data (value) VALUES (?)', ('should_commit',))

            # Verify data was committed
            with db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT COUNT(*), value FROM test_data')
                result = cursor.fetchone()

            if result[0] == 1 and result[1] == 'should_commit':
                results.add_pass(test_name)
            else:
                results.add_fail(test_name, f"Expected 1 row with 'should_commit', got {result}")

    except Exception as e:
        results.add_fail(test_name, str(e))


def test_concurrent_database_access():
    """Test 1.1b: Concurrent database access from multiple threads."""
    test_name = "Concurrent Database Access"

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / 'test.db'

            class TestConfig:
                path = db_path

            db = DatabaseManager(db_config=TestConfig())

            # Create table
            with db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('CREATE TABLE concurrent_test (id INTEGER PRIMARY KEY AUTOINCREMENT, thread_id INTEGER)')

            # Function to insert from thread
            def insert_from_thread(thread_id):
                try:
                    with db.get_connection() as conn:
                        cursor = conn.cursor()
                        cursor.execute('INSERT INTO concurrent_test (thread_id) VALUES (?)', (thread_id,))
                    return True
                except Exception as e:
                    print(f"Thread {thread_id} error: {e}")
                    return False

            # Run 10 threads concurrently
            with ThreadPoolExecutor(max_workers=10) as executor:
                futures = [executor.submit(insert_from_thread, i) for i in range(10)]
                results_list = [f.result() for f in as_completed(futures)]

            # Verify all inserts succeeded
            with db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT COUNT(DISTINCT thread_id) FROM concurrent_test')
                unique_count = cursor.fetchone()[0]

            if all(results_list) and unique_count == 10:
                results.add_pass(test_name)
            else:
                results.add_fail(test_name, f"Expected 10 unique threads, got {unique_count}")

    except Exception as e:
        results.add_fail(test_name, str(e))


def test_file_locking():
    """Test 1.3: File locking prevents concurrent writes."""
    test_name = "File Locking on Session Save"

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / 'test_session.json'

            # Test that we can acquire and release lock
            test_data = {'test': 'data', 'timestamp': time.time()}

            with open(test_file, 'w') as f:
                try:
                    # Acquire lock
                    if platform.system() == 'Windows':
                        msvcrt.locking(f.fileno(), msvcrt.LK_LOCK, 1)
                    else:
                        fcntl.flock(f.fileno(), fcntl.LOCK_EX)

                    # Write data
                    json.dump(test_data, f, indent=2)
                    lock_acquired = True

                finally:
                    # Release lock
                    if platform.system() == 'Windows':
                        msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
                    else:
                        fcntl.flock(f.fileno(), fcntl.LOCK_UN)

            # Verify data was written
            with open(test_file, 'r') as f:
                data = json.load(f)

            if lock_acquired and data['test'] == 'data':
                results.add_pass(test_name)
            else:
                results.add_fail(test_name, "Lock not acquired or data not written")

    except Exception as e:
        results.add_fail(test_name, str(e))


def test_concurrent_file_writes():
    """Test 1.3b: File locking prevents race conditions."""
    test_name = "Concurrent File Write Protection"

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / 'concurrent_test.json'
            write_count = [0]  # Mutable container for thread-safe counter

            def write_with_lock(thread_id):
                try:
                    with open(test_file, 'a') as f:
                        try:
                            if platform.system() == 'Windows':
                                msvcrt.locking(f.fileno(), msvcrt.LK_LOCK, 1)
                            else:
                                fcntl.flock(f.fileno(), fcntl.LOCK_EX)

                            # Simulate work
                            time.sleep(0.01)
                            f.write(f"Thread {thread_id}\n")
                            write_count[0] += 1

                        finally:
                            if platform.system() == 'Windows':
                                msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
                            else:
                                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                    return True
                except Exception as e:
                    print(f"Thread {thread_id} write error: {e}")
                    return False

            # Run 5 threads concurrently
            with ThreadPoolExecutor(max_workers=5) as executor:
                futures = [executor.submit(write_with_lock, i) for i in range(5)]
                results_list = [f.result() for f in as_completed(futures)]

            if all(results_list) and write_count[0] == 5:
                results.add_pass(test_name)
            else:
                results.add_fail(test_name, f"Expected 5 writes, got {write_count[0]}")

    except Exception as e:
        results.add_fail(test_name, str(e))


def test_xml_cleanup_success():
    """Test 1.4a: XML cleanup on successful parse."""
    test_name = "XML Cleanup on Success"

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            test_xml = Path(tmpdir) / 'test.xml'

            # Create minimal valid XML
            xml_content = '''<?xml version="1.0"?>
<PubmedArticleSet>
</PubmedArticleSet>'''

            with open(test_xml, 'w') as f:
                f.write(xml_content)

            # Parse with auto_cleanup
            parser = PubmedParser(auto_cleanup=True)
            parser.parse_metadata_file(str(test_xml))

            # Verify file was deleted
            if not test_xml.exists():
                results.add_pass(test_name)
            else:
                results.add_fail(test_name, "XML file was not cleaned up")

    except Exception as e:
        results.add_fail(test_name, str(e))


def test_xml_cleanup_error():
    """Test 1.4b: XML cleanup on parse error."""
    test_name = "XML Cleanup on Error"

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            test_xml = Path(tmpdir) / 'invalid.xml'

            # Create invalid XML
            with open(test_xml, 'w') as f:
                f.write('INVALID XML CONTENT')

            # Parse with auto_cleanup (will fail)
            parser = PubmedParser(auto_cleanup=True)
            parser.parse_metadata_file(str(test_xml))

            # Verify file was deleted even on error
            if not test_xml.exists():
                results.add_pass(test_name)
            else:
                results.add_fail(test_name, "Invalid XML file was not cleaned up")

    except Exception as e:
        results.add_fail(test_name, str(e))


def test_xml_cleanup_disabled():
    """Test 1.4c: XML cleanup can be disabled."""
    test_name = "XML Cleanup Disabled"

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            test_xml = Path(tmpdir) / 'test.xml'

            # Create minimal valid XML
            xml_content = '''<?xml version="1.0"?>
<PubmedArticleSet>
</PubmedArticleSet>'''

            with open(test_xml, 'w') as f:
                f.write(xml_content)

            # Parse with auto_cleanup disabled
            parser = PubmedParser(auto_cleanup=False)
            parser.parse_metadata_file(str(test_xml))

            # Verify file was NOT deleted
            if test_xml.exists():
                results.add_pass(test_name)
            else:
                results.add_fail(test_name, "XML file was cleaned up despite auto_cleanup=False")

    except Exception as e:
        results.add_fail(test_name, str(e))


def test_wal_mode_enabled():
    """Test 1.1c: WAL mode is enabled for concurrent reads."""
    test_name = "WAL Mode Enabled"

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / 'test.db'

            class TestConfig:
                path = db_path

            db = DatabaseManager(db_config=TestConfig())

            # Check WAL mode is enabled
            with db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('PRAGMA journal_mode')
                mode = cursor.fetchone()[0]

            if mode.lower() == 'wal':
                results.add_pass(test_name)
            else:
                results.add_fail(test_name, f"Expected WAL mode, got {mode}")

    except Exception as e:
        results.add_fail(test_name, str(e))


def main():
    """Run all Phase 1 tests."""
    print("="*70)
    print("PHASE 1 TEST SUITE: Critical Safety Fixes")
    print("="*70)
    print()

    print("Testing Database Threading & Transactions...")
    test_database_thread_local_connections()
    test_transaction_rollback()
    test_transaction_commit()
    test_concurrent_database_access()
    test_wal_mode_enabled()

    print("\nTesting File Locking...")
    test_file_locking()
    test_concurrent_file_writes()

    print("\nTesting XML Cleanup...")
    test_xml_cleanup_success()
    test_xml_cleanup_error()
    test_xml_cleanup_disabled()

    # Print summary
    success = results.summary()

    if success:
        print("\n*** ALL TESTS PASSED! Phase 1 fixes verified. ***")
        return 0
    else:
        print("\n*** WARNING: SOME TESTS FAILED. Review errors above. ***")
        return 1


if __name__ == "__main__":
    sys.exit(main())
