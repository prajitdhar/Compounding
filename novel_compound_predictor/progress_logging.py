"""Compatibility imports for the shared, corpus-independent progress reporter.

New code can import :mod:`src.progress` directly. Install the repository
package first, or set PYTHONPATH to the repository root for development.
"""
from src.progress import ProgressReporter, memory_usage

__all__ = ["ProgressReporter", "memory_usage"]
