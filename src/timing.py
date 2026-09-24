"""Latency timing helpers.

Provides a simple timer / context manager for measuring stage durations in
milliseconds, and appends one JSON record per turn to the latency log whose
path comes from ``config.yaml``.

CLI: ``python -m src.timing [path/to/latency.jsonl]`` prints mean latency per stage.
"""

from __future__ import annotations

import json
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_LOG_PATH = Path("logs/latency.jsonl")


def now() -> float:
    """Return a monotonic timestamp in seconds, for use with ``ms_since``."""
    return time.perf_counter()


def ms_since(start: float) -> float:
    """Return milliseconds elapsed since ``start`` (a value from ``now()``)."""
    return (time.perf_counter() - start) * 1000.0


@contextmanager
def timed(timings: dict[str, float], key: str) -> Iterator[None]:
    """Time the enclosed block and store its duration in ms as ``timings[key]``.

    The duration is recorded even if the block raises.

        timings = {}
        with timed(timings, "stt_ms"):
            text = transcribe(audio)
    """
    start = now()
    try:
        yield
    finally:
        timings[key] = ms_since(start)


def log_turn(
    timings: dict[str, float],
    path: str | Path = DEFAULT_LOG_PATH,
    **extra: object,
) -> dict[str, object]:
    """Append one turn's timings as a single JSON line to ``path``.

    Adds a UTC ``timestamp``, rounds ms values to 0.1, and merges any ``extra``
    fields (e.g. ``transcript=...``). Creates the parent directory if needed.
    Returns the record that was written.
    """
    record: dict[str, object] = {"timestamp": datetime.now(timezone.utc).isoformat()}
    record.update({k: round(v, 1) for k, v in timings.items()})
    record.update(extra)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return record


def _summarize(path: Path) -> None:
    """Print the number of turns and mean of each ``*_ms`` field in the log."""
    values: dict[str, list[float]] = {}
    turns = 0
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            turns += 1
            for key, value in json.loads(line).items():
                if key.endswith("_ms") and isinstance(value, (int, float)):
                    values.setdefault(key, []).append(float(value))
    print(f"{turns} turns in {path}")
    for key, vals in values.items():
        print(f"  {key:<28} mean {sum(vals) / len(vals):8.1f} ms  (n={len(vals)})")


if __name__ == "__main__":
    log_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_LOG_PATH
    if not log_path.exists():
        sys.exit(f"No latency log at {log_path}")
    _summarize(log_path)
