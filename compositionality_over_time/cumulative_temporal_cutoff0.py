"""Cumulative temporal features for the supported cutoff-zero configuration.

Counting, PPMI, lag selection and output writing use the common temporal
implementation. The configuration and its source identity are recorded in
the generated feature manifests.
"""

import contextlib
import json
import sys
from pathlib import Path

import cumulative_temporal as temporal
from pipeline_source_info import software_record


def _require_configuration(args):
    expected = {"context_size": 30000, "condition": "CompoundAware",
                "weight": "PPMI", "cutoff": 0, "time_step": 10}
    if any(getattr(args, name, None) != value for name, value in expected.items()):
        raise ValueError("Cumulative temporal cutoff-zero features are scoped to "
                         "context-size=30000, UnTagged, condition=CompoundAware, "
                         "weight=PPMI, cutoff=0, time-step=10")


@contextlib.contextmanager
def _cutoff_zero_scope():
    """Apply the extension only during a delegated call; never mutate on import."""
    previous_guard = temporal._require_configuration
    previous_specification = temporal.read_specification

    def extended_specification(args):
        specification = previous_specification(args)
        material = json.loads(specification["input_identity_material"])
        material["cutoff_extension_software"] = software_record(
            [Path(__file__)], ["duckdb", "pyarrow", "pandas", "numpy", "scipy"])
        specification["input_identity_material"] = temporal.cumulative._material(material)
        specification["input_identity"] = temporal.cumulative._sha_text(
            specification["input_identity_material"])
        return specification

    temporal._require_configuration = _require_configuration
    temporal.read_specification = extended_specification
    try:
        yield
    finally:
        temporal._require_configuration = previous_guard
        temporal.read_specification = previous_specification


def read_specification(args):
    with _cutoff_zero_scope():
        return temporal.read_specification(args)


def prepare(args):
    with _cutoff_zero_scope():
        return temporal.prepare(args)


def run_features(args):
    with _cutoff_zero_scope():
        return temporal.run_features(args)


def check(args):
    with _cutoff_zero_scope():
        return temporal.check(args)


def _arguments(argv):
    values = list(sys.argv[1:] if argv is None else argv)
    if values and values[0] in ("plan", "prepare", "features", "check"):
        # The common parser inserts cutoff 500 before these arguments. Our
        # default follows that value, but explicit user options still come last
        # and are checked against the cutoff-zero-only production guard.
        values[1:1] = ["--cutoff", "0"]
    return values


def parse_args(argv=None):
    with _cutoff_zero_scope():
        return temporal.parse_args(_arguments(argv))


def main(argv=None):
    with _cutoff_zero_scope():
        return temporal.main(_arguments(argv))


if __name__ == "__main__":
    main()
