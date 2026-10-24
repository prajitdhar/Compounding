"""Observable, deterministic specifications for the shared progress reporter."""

import io
import math
import time
import unittest

from src.progress import ProgressReporter, memory_usage


class SharedProgressTests(unittest.TestCase):
    def test_stage_lifecycle_and_default_utc(self):
        output = io.StringIO()
        reporter = ProgressReporter(interval=30, stream=output)
        with reporter:
            self.assertTrue(reporter.running)
            with reporter.stage("example", total=2):
                reporter.update("one row", completed=1)
                reporter.log("milestone")
        self.assertFalse(reporter.running)
        lines = output.getvalue().splitlines()
        self.assertTrue(all(line.split(" ", 1)[0].endswith("+00:00") for line in lines))
        for event in ["RUN_START", "START", "INFO", "END", "RUN_END"]:
            self.assertTrue(any(f'event="{event}"' in line for line in lines))
        self.assertNotIn("query_percent", output.getvalue())

    def test_failure_is_logged_and_original_error_propagates(self):
        output = io.StringIO()
        reporter = ProgressReporter(interval=30, stream=output)
        with self.assertRaisesRegex(ValueError, "example failure"):
            with reporter:
                with reporter.stage("failed stage"):
                    raise ValueError("example failure")
        self.assertIn('event="FAILED"', output.getvalue())
        self.assertIn('event="RUN_FAILED"', output.getvalue())
        self.assertFalse(reporter.running)

    def test_background_poll_never_executes_sql_and_omits_unknown_percent(self):
        class Connection:
            polls = 0

            def execute(self, *args):
                raise AssertionError("Heartbeat must never execute SQL")

            def query_progress(self):
                self.polls += 1
                return -1

        connection = Connection()
        output = io.StringIO()
        with ProgressReporter(interval=0.01, stream=output, connection=connection) as reporter:
            with reporter.stage("query", query_progress=True):
                deadline = time.monotonic() + 1
                while connection.polls == 0 and time.monotonic() < deadline:
                    time.sleep(0.01)
                self.assertGreater(connection.polls, 0)
                reporter.set_connection(None)
        self.assertIn('event="HEARTBEAT"', output.getvalue())
        self.assertNotIn("query_percent", output.getvalue())

    def test_invalid_counts_and_inactive_updates_are_rejected(self):
        for interval in [0, -1, math.inf, math.nan]:
            with self.subTest(interval=interval), self.assertRaises(ValueError):
                ProgressReporter(interval=interval)
        reporter = ProgressReporter(stream=io.StringIO())
        with self.assertRaises(RuntimeError):
            reporter.update(completed=1)
        with reporter:
            with self.assertRaises(RuntimeError):
                reporter.update(completed=1)
            with reporter.stage("counts"):
                with self.assertRaises(ValueError):
                    reporter.update(completed=-1)
        self.assertTrue(set(memory_usage()).issubset({"rss_mib", "peak_rss_mib"}))


if __name__ == "__main__":
    unittest.main()
