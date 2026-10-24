"""Flushed stage logs and observable heartbeats for long-running programs.

Optional DuckDB progress polling calls only query_progress(), never SQL from
the supervisory thread. Stages without real counts have no invented percent
or ETA. RSS is current process memory on Linux, with a labeled peak fallback.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime
import json
import math
from pathlib import Path
import sys
import threading
import time
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

__all__ = ["ProgressReporter", "memory_usage"]


def memory_usage() -> dict[str, float]:
    """Return current RSS in MiB, or an explicitly labeled peak fallback."""
    try:
        with Path("/proc/self/status").open() as status:
            for line in status:
                if line.startswith("VmRSS:"):
                    return {"rss_mib": round(float(line.split()[1]) / 1024, 1)}
    except (OSError, ValueError, IndexError):
        pass
    try:
        import resource
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        divisor = 1024 * 1024 if sys.platform == "darwin" else 1024
        return {"peak_rss_mib": round(float(peak) / divisor, 1)}
    except (ImportError, OSError, ValueError):
        return {}


class ProgressReporter:
    """One daemon heartbeat thread, explicit stage boundaries, flushed output.

    Use as a context manager. The interval must be positive seconds. Nested
    stage durations include nested work. Updates store real observations for
    later heartbeats; log() emits immediate milestones. Close the reporter
    before closing a supplied DuckDB connection.
    """

    def __init__(self, interval: float = 30, stream=None, connection=None,
                 timezone: str = "UTC"):
        if not math.isfinite(interval) or interval <= 0:
            raise ValueError("Log interval must be a positive, finite number of seconds")
        try:
            self.timezone = ZoneInfo(timezone)
        except ZoneInfoNotFoundError as error:
            raise ValueError(f"Timezone {timezone!r} is unavailable; install system timezone data") from error
        self.interval = float(interval)
        self.stream = sys.stderr if stream is None else stream
        self._connection_lock = threading.RLock()
        self.connection = connection
        self._stop = threading.Event()
        self._state_lock = threading.RLock()
        self._write_lock = threading.Lock()
        self._stages = []
        self._thread = None
        self._started_at = None
        self._closed = False

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def connection(self):
        with self._connection_lock:
            return self._connection

    @connection.setter
    def connection(self, connection):
        # Assign None before closing/replacing a caller-owned connection.
        # This lock waits for its last supervisory progress poll to finish.
        with self._connection_lock:
            self._connection = connection

    def set_connection(self, connection):
        self.connection = connection

    def __enter__(self):
        with self._state_lock:
            if self._started_at is not None:
                raise RuntimeError("ProgressReporter cannot be entered more than once")
            self._started_at = time.monotonic()
        self._emit("RUN_START")
        self._thread = threading.Thread(target=self._heartbeat, name="compounding-progress", daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exception_type, exception, traceback):
        self.close()
        if exception is None:
            self._emit("RUN_END")
        else:
            self._emit("RUN_FAILED", error=f"{exception_type.__name__}: {exception}")
        return False

    def close(self) -> None:
        """Stop and join the heartbeat; never close the caller's stream."""
        self._stop.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join()
        self._closed = True

    def _check_active(self):
        if self._started_at is None or self._closed:
            raise RuntimeError("Use an active 'with ProgressReporter(...)' context")

    def _snapshot(self):
        with self._state_lock:
            stage = dict(self._stages[-1]) if self._stages else {}
            stage["path"] = " > ".join(item["name"] for item in self._stages) or "idle"
            stage["fields"] = dict(stage.get("fields", {}))
            return stage

    def _query_percent(self):
        with self._connection_lock:
            query_progress = getattr(self._connection, "query_progress", None)
            if not callable(query_progress):
                return None
            try:
                percent = float(query_progress())
            except Exception:
                return None
        return percent if math.isfinite(percent) and 0 <= percent <= 100 else None

    def _emit(self, event: str, *, snapshot=None, **fields):
        now = time.monotonic()
        stage = self._snapshot() if snapshot is None else snapshot
        values = {"event": event, "elapsed_s": round(now - self._started_at, 1), "stage": stage["path"]}
        if "started_at" in stage:
            values["stage_elapsed_s"] = round(now - stage["started_at"], 1)
        values.update(memory_usage())
        if stage.get("total") is not None:
            values["total"] = stage["total"]
        if stage.get("completed") is not None:
            values["completed"] = stage["completed"]
        if stage.get("detail"):
            values["detail"] = stage["detail"]
        values.update(stage["fields"])
        values.update(fields)
        if event == "HEARTBEAT" and stage.get("query_progress") and not self._stop.is_set():
            percent = self._query_percent()
            if percent is not None:
                values["query_percent"] = round(percent, 2)
        timestamp = datetime.now(self.timezone).isoformat(timespec="seconds")
        line = timestamp + " " + " ".join(
            f"{key}={json.dumps(value, ensure_ascii=False, default=str)}" for key, value in values.items())
        with self._write_lock:
            print(line, file=self.stream, flush=True)

    def _heartbeat(self):
        while not self._stop.wait(self.interval):
            try:
                self._emit("HEARTBEAT")
            except (OSError, ValueError):
                self._stop.set()
                break

    @contextmanager
    def stage(self, name: str, total=None, *, query_progress: bool = False):
        self._check_active()
        if not name:
            raise ValueError("Stage name must be non-empty")
        if total is not None and (not math.isfinite(total) or total < 0):
            raise ValueError("Stage total must be non-negative and finite")
        stage = {"name": str(name), "started_at": time.monotonic(), "total": total,
                 "completed": None, "detail": "", "fields": {}, "query_progress": query_progress}
        with self._state_lock:
            self._stages.append(stage)
        try:
            self._emit("START")
            try:
                yield self
            except BaseException as error:
                self._emit("FAILED", error=f"{type(error).__name__}: {error}")
                raise
            else:
                self._emit("END")
        finally:
            with self._state_lock:
                self._stages.remove(stage)

    def log(self, message: str, **fields):
        self._check_active()
        self._emit("INFO", message=str(message), **fields)

    def update(self, detail=None, *, completed=None, total=None, **fields):
        self._check_active()
        for value in (completed, total):
            if value is not None and (not math.isfinite(value) or value < 0):
                raise ValueError("Progress counts must be non-negative and finite")
        with self._state_lock:
            if not self._stages:
                raise RuntimeError("update() requires an active stage")
            stage = self._stages[-1]
            if detail is not None:
                stage["detail"] = str(detail)
            if completed is not None:
                stage["completed"] = completed
            if total is not None:
                stage["total"] = total
            stage["fields"].update(fields)
