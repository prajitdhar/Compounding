"""Small standard-library-only tests of heartbeat visibility and cleanup."""

import io
from pathlib import Path
import re
import sys
import threading
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from progress_logging import ProgressReporter, memory_usage


class FlushTrackingStream(io.StringIO):
    def __init__(self):
        super().__init__()
        self.flushes = 0
        self.heartbeat = threading.Event()

    def write(self, value):
        result = super().write(value)
        if 'event="HEARTBEAT"' in value:
            self.heartbeat.set()
        return result

    def flush(self):
        self.flushes += 1
        super().flush()


class ProgressLoggingTests(unittest.TestCase):
    def test_heartbeat_appears_during_blocking_stage_without_update_calls(self):
        stream = FlushTrackingStream()
        with ProgressReporter(interval=.01, stream=stream) as progress:
            with progress.stage("fit SVD"):
                # Simulates blocking work with no batches or callbacks.
                self.assertTrue(stream.heartbeat.wait(timeout=1))
                self.assertTrue(progress.running)
        self.assertFalse(progress.running)
        lines = stream.getvalue().splitlines()
        self.assertTrue(any('event="HEARTBEAT"' in line and 'stage="fit SVD"' in line for line in lines))
        self.assertTrue(all("elapsed_s=" in line for line in lines))
        self.assertTrue(all(re.match(r"\d{4}-\d\d-\d\dT.*[+-]\d\d:\d\d ", line) for line in lines))
        self.assertTrue(any("rss_mib=" in line for line in lines))
        self.assertNotIn("percent", stream.getvalue())
        self.assertEqual(stream.flushes, len(lines))
        self.assertFalse(stream.closed)

    def test_exception_is_logged_and_propagated_and_thread_stops(self):
        stream = FlushTrackingStream()
        progress = ProgressReporter(interval=.01, stream=stream)
        with self.assertRaisesRegex(RuntimeError, "original failure"):
            with progress:
                with progress.stage("aggregate contexts"):
                    raise RuntimeError("original failure")
        self.assertFalse(progress.running)
        self.assertIn('event="FAILED"', stream.getvalue())
        self.assertIn('event="RUN_FAILED"', stream.getvalue())
        self.assertIn("RuntimeError: original failure", stream.getvalue())
        self.assertNotIn('event="END"', stream.getvalue())
        recorded = stream.getvalue()
        time.sleep(.03)
        self.assertEqual(stream.getvalue(), recorded)

    def test_nested_stage_counts_and_updates_are_flushed_and_parent_restored(self):
        stream = FlushTrackingStream()
        with ProgressReporter(interval=30, stream=stream) as progress:
            with progress.stage("build", total=5):
                before_update = stream.getvalue()
                progress.update("aggregated three rows", completed=3, rows=3)
                self.assertEqual(stream.getvalue(), before_update)
                with progress.stage("write block"):
                    progress.log("file written", path="output.parquet")
                progress.log("parent resumed")
        output = stream.getvalue()
        self.assertIn('stage="build > write block"', output)
        line = next(line for line in output.splitlines() if 'message="parent resumed"' in line)
        self.assertIn('stage="build"', line)
        self.assertIn("completed=3", line)
        self.assertIn("total=5", line)
        self.assertIn('detail="aggregated three rows"', line)
        self.assertEqual(stream.flushes, len(output.splitlines()))

    def test_duckdb_polling_uses_only_supervisory_api_in_sql_stage(self):
        class Connection:
            def __init__(self):
                self.poll_threads = []

            def query_progress(self):
                self.poll_threads.append(threading.current_thread().name)
                return 12.34

            def execute(self, *args):
                raise AssertionError("Heartbeat must never execute SQL")

        connection = Connection()
        stream = FlushTrackingStream()
        with ProgressReporter(interval=.01, stream=stream, connection=connection) as progress:
            with progress.stage("SQL aggregation", query_progress=True):
                self.assertTrue(stream.heartbeat.wait(timeout=1))
        self.assertIn("query_percent=12.34", stream.getvalue())
        self.assertEqual(set(connection.poll_threads), {"compounding-progress"})

    def test_detaching_connection_waits_for_in_flight_poll_before_caller_closes_it(self):
        class Connection:
            def __init__(self):
                self.poll_entered = threading.Event()
                self.allow_poll_to_finish = threading.Event()
                self.closed = False
                self.polled_after_close = False

            def query_progress(self):
                self.poll_entered.set()
                self.allow_poll_to_finish.wait(timeout=1)
                self.polled_after_close |= self.closed
                return 50

        connection = Connection()
        stream = FlushTrackingStream()
        with ProgressReporter(interval=.01, stream=stream, connection=connection) as progress:
            with progress.stage("SQL", query_progress=True):
                detached = threading.Event()
                setter_entered = threading.Event()

                def detach():
                    setter_entered.set()
                    progress.connection = None
                    detached.set()

                try:
                    self.assertTrue(connection.poll_entered.wait(timeout=1))
                    setter = threading.Thread(target=detach)
                    setter.start()
                    self.assertTrue(setter_entered.wait(timeout=1))
                    self.assertFalse(detached.wait(timeout=.02))
                finally:
                    connection.allow_poll_to_finish.set()
                self.assertTrue(detached.wait(timeout=1))
                setter.join(timeout=1)
                connection.closed = True
        self.assertFalse(connection.polled_after_close)
        self.assertIsNone(progress.connection)

    def test_missing_or_unknown_query_progress_never_fabricates_a_percent(self):
        for value in (-1, float("nan"), float("inf"), 101):
            class Connection:
                def query_progress(self):
                    return value

            stream = FlushTrackingStream()
            with self.subTest(progress=value):
                with ProgressReporter(interval=.01, stream=stream, connection=Connection()) as progress:
                    with progress.stage("SQL aggregation", query_progress=True):
                        self.assertTrue(stream.heartbeat.wait(timeout=1))
                self.assertNotIn("query_percent", stream.getvalue())

    def test_resource_memory_fallback_is_explicitly_labeled_as_peak(self):
        with patch("src.progress.Path.open", side_effect=OSError("no procfs")):
            value = memory_usage()
        self.assertIn("peak_rss_mib", value)
        self.assertGreater(value["peak_rss_mib"], 0)
        self.assertNotIn("rss_mib", value)

    def test_invalid_interval_and_missing_context_fail_promptly(self):
        for interval in [0, -1, float("nan"), float("inf")]:
            with self.assertRaises(ValueError):
                ProgressReporter(interval=interval)
        progress = ProgressReporter()
        with self.assertRaisesRegex(RuntimeError, "active"):
            with progress.stage("not entered"):
                pass


if __name__ == "__main__":
    unittest.main()
