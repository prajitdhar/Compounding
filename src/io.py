"""Atomic publication of small metadata files and calculated Parquet tables."""

from contextlib import contextmanager
from pathlib import Path
import os
import tempfile

__all__ = ["atomic_path", "write_parquet"]


@contextmanager
def atomic_path(path):
    """Yield a unique sibling temporary path and replace the target on success.

    The target is left untouched if the writer raises. Temporary files stay on
    the target filesystem so publication uses a single atomic replacement.
    This does not provide a lock against two successful concurrent publishers.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{destination.stem}-", suffix=destination.suffix + ".tmp",
        dir=destination.parent,
    )
    os.close(descriptor)
    temporary = Path(name)
    try:
        yield temporary
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def write_parquet(frame, path):
    """Publish a dataframe as index-free, Zstandard-compressed Parquet."""
    with atomic_path(path) as temporary:
        frame.to_parquet(temporary, engine="pyarrow", compression="zstd", index=False)
