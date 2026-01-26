"""Simple runtime metrics collector used by the test harness.

This module provides a minimal process-global collector so different
parts of the code can record timings/values for the current
calculation. It's intentionally small and dependency-free.
"""

import time

runs = []
_current = None


def start_run(run_id=None):
    global _current
    _current = {"id": run_id, "metrics": {}, "start": time.time()}
    runs.append(_current)


def record(key, value):
    """Record a scalar value for the current run."""
    if _current is None:
        start_run(None)
    _current["metrics"][key] = value


def finish_run():
    if _current is not None:
        _current["finished"] = time.time()


def get_runs():
    return runs


def get_last_metrics():
    if not runs:
        return None
    return runs[-1]["metrics"]

