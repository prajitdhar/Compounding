#!/usr/bin/env python3
"""Streaming pipeline for Google Books Ngrams version 3 (20200217).

The public V3 files contain one ngram followed by packed yearly triples::

    ngram<TAB>year,match_count,volume_count<TAB>...

This module uses Google's POS/dependency annotations.

Pipeline stages
---------------
1. ``manifest`` downloads the official shard indexes.
2. ``extract`` processes one manifest task into atomic Parquet fragments.
3. ``reduce`` combines fragments into canonical Parquet tables with DuckDB.
4. ``lemmatize`` maps each aggregated word/POS type to a lemma once.
5. ``vocab`` creates a fixed, lemma-aware context vocabulary.
6. ``ner-predict`` tags unique, cased five-token snippets with spaCy.
7. ``ner-aggregate`` creates occurrence-weighted pair/decade NER estimates.

Only ``manifest`` and ``extract`` tasks use the network; however, ``extract`` can
also read a local .gz file. PyArrow, DuckDB, and spaCy are imported only by
stages that need them.
"""

import argparse
import contextlib
import csv
import gzip
import hashlib
import html.parser
import http.client
import io
import importlib
import json
import os
import shutil
import socket
import sys
import time
import unicodedata
import urllib.error
import urllib.request
from collections import OrderedDict, defaultdict, namedtuple
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from src.source_info import (FEATURE_INPUT_SCHEMA_VERSION, artifact,
                                 external_artifact, feature_table, source_info,
                                 software_record, write_json)
from src.partitioned_parquet import (add_partition_arguments, connection_scope,
                                 partition_options, partitioned_sum)
from src import partitioned_parquet as _partitioned_implementation


CORPUS_VERSION = "20200217"
DEFAULT_CORPUS = "eng"
EXPORT_ROOT = "https://storage.googleapis.com/books/ngrams/books"
VALID_TAGS = {
    "ADJ", "ADP", "ADV", "CONJ", "DET", "NOUN", "NUM", "PRON",
    "PRT", "VERB", "X", ".",
}
ELIGIBLE_PAIR_POS = {("ADJ", "NOUN"), ("NOUN", "NOUN")}
DEFAULT_CONTEXT_POS = {"ADJ", "ADV", "NOUN", "NUM", "VERB"}
BOUNDARIES = {"_START_", "_END_"}

Token = namedtuple("Token", "raw surface pos wildcard boundary")
Measurement = namedtuple("Measurement", "year match_count volume_count")


def pipeline_software(extra_paths=(), packages=()):
    return software_record(
        [__file__, Path(_partitioned_implementation.__file__), *extra_paths],
        packages,
    )


def corpus_record(corpus=DEFAULT_CORPUS):
    return {"name": "Google Books Ngrams", "release": CORPUS_VERSION,
            "subset": corpus}


def time_record(year_min=None, year_max=None, bin_size=None):
    return {
        "column": "time", "source_column": "year", "unit": "calendar_year",
        "precision": "year", "observed_min": year_min, "observed_max": year_max,
        "binning": (f"floor(year / {bin_size}) * {bin_size}"
                    if bin_size is not None else None),
    }


def normalize(text):
    """Stable normalization used for aggregation; cased surfaces are retained."""
    return unicodedata.normalize("NFKC", text).casefold()


def parse_tagged_token(raw):
    """Parse a V3 token such as ``ice_NOUN`` or ``_NOUN_``.

    Wildcard POS tokens are projections with no lexical identity and are not
    accepted as targets or contexts. Sentence boundary markers are retained
    for NER-window alignment but are not lexical contexts.
    """
    if raw in BOUNDARIES:
        return Token(raw, None, "BOUNDARY", False, True)

    if len(raw) > 2 and raw[0] == "_" and raw[-1] == "_":
        possible_tag = raw[1:-1]
        if possible_tag in VALID_TAGS:
            return Token(raw, None, possible_tag, True, False)

    if "_" not in raw:
        return Token(raw, None, None, False, False)

    surface, possible_tag = raw.rsplit("_", 1)
    if surface and possible_tag in VALID_TAGS:
        return Token(raw, surface, possible_tag, False, False)
    return Token(raw, None, None, False, False)


def is_lexical(token):
    return token.surface is not None and token.pos in VALID_TAGS


def is_lexical_word(token):
    """A lexical corpus token, excluding Google's punctuation POS bucket."""
    return is_lexical(token) and token.pos != "."


def is_fully_annotated(tokens):
    return all(is_lexical(token) or token.boundary for token in tokens)


def is_eligible_pair(modifier, head):
    return (
        is_lexical(modifier)
        and is_lexical(head)
        and (modifier.pos, head.pos) in ELIGIBLE_PAIR_POS
    )


def parse_v3_record(line):
    """Return ``(ngram, measurements)`` for one packed V3 record."""
    fields = line.rstrip("\r\n").split("\t")
    if len(fields) < 2 or not fields[0]:
        raise ValueError("record has no ngram or yearly measurements")

    measurements = []
    for field in fields[1:]:
        values = field.split(",")
        if len(values) < 3:
            raise ValueError(f"invalid year,count,volume triple: {field!r}")
        try:
            measurements.append(
                Measurement(int(values[0]), int(values[1]), int(values[2]))
            )
        except ValueError:
            raise ValueError(f"non-integer year/count triple: {field!r}")
    return fields[0], measurements


def aggregate_measurements(measurements, year_min, year_max, bin_size):
    """Aggregate yearly Google match counts into bins without expanding rows."""
    totals = defaultdict(int)
    for measurement in measurements:
        if measurement.year < year_min or measurement.year > year_max:
            continue
        decade = measurement.year - (measurement.year % bin_size)
        totals[decade] += measurement.match_count
    return totals


def normalized_token_columns(prefix, token):
    """Compact join key used by high-volume context-event tables."""
    return {
        prefix + "_norm": normalize(token.surface),
        prefix + "_pos": token.pos,
    }


def surface_pair_event(modifier, head):
    row = {}
    row.update({
        "modifier_surface": modifier.surface,
        "modifier_pos": modifier.pos,
        "head_surface": head.surface,
        "head_pos": head.pos,
    })
    return row


def unigram_events(tokens):
    if len(tokens) != 1 or not is_lexical(tokens[0]):
        return []
    token = tokens[0]
    return [{
        "surface": token.surface,
        "pos": token.pos,
    }]


def bigram_events(tokens):
    """Retain every cased, fully lexical adjacency in one reusable stream."""
    if len(tokens) != 2 or not all(is_lexical_word(token) for token in tokens):
        return []
    return [surface_pair_event(tokens[0], tokens[1])]


def context_events(tokens, context_pos=None):
    """Extract one event at the far edge of an annotated 3/4/5-gram.

    For order n, the exposed context has distance n-2. Only edge pairs are
    used. Thus a context at distance 1 comes from a 3-gram, distance 2 from a
    4-gram, and distance 3 from a 5-gram. This avoids triangular duplication
    from treating every word in every overlapping 5-gram as context.
    """
    if context_pos is None:
        context_pos = DEFAULT_CONTEXT_POS
    order = len(tokens)
    if order not in (3, 4, 5) or not is_fully_annotated(tokens):
        return []

    distance = order - 2
    events = []

    # modifier head ... context
    if all(is_lexical_word(token) for token in tokens[:2]):
        context = tokens[-1]
        if is_lexical(context) and context.pos in context_pos:
            row = {}
            row.update(normalized_token_columns("modifier", tokens[0]))
            row.update(normalized_token_columns("head", tokens[1]))
            row.update(normalized_token_columns("context", context))
            row.update({"side": "R", "distance": distance})
            events.append(row)

    # context ... modifier head
    if all(is_lexical_word(token) for token in tokens[-2:]):
        context = tokens[0]
        if is_lexical(context) and context.pos in context_pos:
            row = {}
            row.update(normalized_token_columns("modifier", tokens[-2]))
            row.update(normalized_token_columns("head", tokens[-1]))
            row.update(normalized_token_columns("context", context))
            row.update({"side": "L", "distance": distance})
            events.append(row)
    return events


def word_context_events(tokens, context_pos=None, target_pos=None):
    """Extract general word contexts at distances 1--3 from 2--4-grams.

    As with pair contexts, only the two edges are used. A target at the left
    edge receives the right-edge word as context and vice versa. This supplies
    context-agnostic constituent vectors without duplicating nearer contexts
    across multiple ngram orders.
    """
    if context_pos is None:
        context_pos = DEFAULT_CONTEXT_POS
    if target_pos is None:
        # The Agnostic representation includes every lexical use of a word.
        # POS can still be retained or pooled later by the shared feature code.
        target_pos = VALID_TAGS - {"."}
    order = len(tokens)
    if order not in (2, 3, 4) or not is_fully_annotated(tokens):
        return []


    distance = order - 1
    events = []

    ## Right side

    target = tokens[0]
    context = tokens[-1]
    if (
        is_lexical(target) and target.pos in target_pos
        and is_lexical(context) and context.pos in context_pos
    ):
        row = normalized_token_columns("word", target)
        row.update(normalized_token_columns("context", context))
        row.update({"side": "R", "distance": distance})
        events.append(row)

    ## Left side


    target = tokens[-1]
    context = tokens[0]
    if (
        is_lexical(target) and target.pos in target_pos
        and is_lexical(context) and context.pos in context_pos
    ):
        row = normalized_token_columns("word", target)
        row.update(normalized_token_columns("context", context))
        row.update({"side": "L", "distance": distance})
        events.append(row)
    return events


def dependency_events(ngram):
    """Extract ADJ->NOUN and NOUN->NOUN dependency arcs.

    Google dependency ngrams use an arrow pointing from head to dependent:
    ``head=>dependent`` or equivalently ``dependent<=head``. Dependency arcs
    are not guaranteed to be linearly adjacent and their labels are absent.
    """
    if ngram.count("=>") == 1:
        head_raw, dependent_raw = ngram.split("=>", 1)
        direction = "head=>dependent"
    elif ngram.count("<=") == 1:
        dependent_raw, head_raw = ngram.split("<=", 1)
        direction = "dependent<=head"
    else:
        return []

    head = parse_tagged_token(head_raw)
    dependent = parse_tagged_token(dependent_raw)
    if not is_eligible_pair(dependent, head):
        return []
    row = surface_pair_event(dependent, head)
    row["direction"] = direction
    return [row]


def ner_snippet_events(tokens, pair_offset=2):
    """Create one canonical, cased NER snippet from a fully tagged 5-gram."""
    if len(tokens) != 5 or not is_fully_annotated(tokens):
        return []
    if pair_offset < 0 or pair_offset + 1 >= len(tokens):
        return []
    modifier = tokens[pair_offset]
    head = tokens[pair_offset + 1]
    if not is_eligible_pair(modifier, head):
        return []

    model_tokens = []
    original_to_model = {}
    for index, token in enumerate(tokens):
        if token.boundary:
            continue
        if not is_lexical(token):
            return []
        original_to_model[index] = len(model_tokens)
        model_tokens.append(token.surface)

    row = surface_pair_event(modifier, head)
    row.update({
        "tokens_json": json.dumps(model_tokens, ensure_ascii=False),
        "modifier_index": original_to_model[pair_offset],
        "head_index": original_to_model[pair_offset + 1],
        "sentence_start": tokens[0].raw == "_START_",
        "sentence_end": tokens[-1].raw == "_END_",
    })
    return [row]


class ExportIndexParser(html.parser.HTMLParser):
    def __init__(self):
        html.parser.HTMLParser.__init__(self)
        self.urls = []

    def handle_starttag(self, tag, attrs):
        if tag != "a":
            return
        href = dict(attrs).get("href")
        if href and href.endswith(".gz"):
            self.urls.append(href.replace("http://", "https://", 1))


def download_manifest(corpus, orders, timeout=120):
    entries = []
    indexes = []
    for order in orders:
        index_url = (
            f"{EXPORT_ROOT}/{CORPUS_VERSION}/{corpus}/{corpus}-{order}-ngrams_exports.html"
        )
        with contextlib.closing(urllib.request.urlopen(index_url, timeout=timeout)) as response:
            body_bytes = response.read()
        body = body_bytes.decode("utf-8")
        indexes.append(external_artifact(
            f"{corpus}-{order}-ngrams_exports.html", index_url,
            bytes=len(body_bytes), sha256=hashlib.sha256(body_bytes).hexdigest(),
        ))
        parser = ExportIndexParser()
        parser.feed(body)
        if not parser.urls:
            raise RuntimeError(f"no shard URLs found at {index_url}")
        for url in parser.urls:
            entries.append({
                "order": int(order),
                "source": url,
                "shard": os.path.basename(url)[:-3],
            })
    result = {
        "schema_version": 1,
        "corpus_version": CORPUS_VERSION,
        "corpus": corpus,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "entries": entries,
    }
    result["source_info"] = source_info(
        "manifest", corpus_record(corpus), indexes,
        {"orders": list(orders), "export_root": EXPORT_ROOT,
         "task_definition": "one official compressed shard per entry"},
        time_record(), pipeline_software(),
        statistics={"tasks": len(entries),
                    "tasks_by_order": {str(order): sum(e["order"] == order for e in entries)
                                       for order in orders}},
        execution={"network_timeout_seconds": timeout}, status="planned",
    )
    return result

class _DigestingReader:
    """Read-through wrapper that identifies the exact compressed source bytes."""

    def __init__(self, raw):
        self.raw = raw
        self.digest = hashlib.sha256()
        self.bytes_read = 0

    def read(self, size=-1):
        data = self.raw.read(size)
        self.digest.update(data)
        self.bytes_read += len(data)
        return data

    def tell(self):
        return self.bytes_read

    def close(self):
        self.raw.close()


class _TextSource:
    def __init__(self, text, compressed):
        self.text = text
        self.compressed = compressed

    def __iter__(self):
        return iter(self.text)

    @property
    def source_sha256(self):
        return self.compressed.digest.hexdigest()

    @property
    def source_bytes(self):
        return self.compressed.bytes_read


@contextlib.contextmanager
def open_text_source(source, timeout=120):
    """Open a gzip shard and hash compressed bytes while streaming it."""
    if source.startswith("http://") or source.startswith("https://"):
        raw_handle = urllib.request.urlopen(source, timeout=timeout)
    else:
        raw_handle = open(source, "rb")
    compressed = _DigestingReader(raw_handle)
    gz_handle = gzip.GzipFile(fileobj=compressed)
    text_handle = io.TextIOWrapper(gz_handle, encoding="utf-8", errors="replace")
    opened = _TextSource(text_handle, compressed)
    try:
        yield opened
    finally:
        try:
            text_handle.close()
        finally:
            raw_handle.close()


def require_pyarrow():
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:
        raise RuntimeError(
            "PyArrow is required for extraction: pip install -e '.[preprocessing]'"
        )
    return pa, pq


def require_duckdb():
    try:
        import duckdb
    except ImportError:
        raise RuntimeError(
            "DuckDB is required for reduction: pip install -e '.[preprocessing]'"
        )
    return duckdb


SCHEMAS = {
    "unigrams": [
        ("surface", "string"), ("pos", "string"),
        ("decade", "int32"), ("count", "int64"),
    ],
    "bigrams": [
        ("modifier_surface", "string"), ("modifier_pos", "string"),
        ("head_surface", "string"), ("head_pos", "string"),
        ("decade", "int32"), ("count", "int64"),
    ],
    "contexts": [
        ("modifier_norm", "string"), ("modifier_pos", "string"),
        ("head_norm", "string"), ("head_pos", "string"),
        ("context_norm", "string"), ("context_pos", "string"),
        ("side", "string"), ("distance", "int8"),
        ("decade", "int32"), ("count", "int64"),
    ],
    "word_contexts": [
        ("word_norm", "string"), ("word_pos", "string"),
        ("context_norm", "string"), ("context_pos", "string"),
        ("side", "string"), ("distance", "int8"),
        ("decade", "int32"), ("count", "int64"),
    ],
    "dependencies": [
        ("modifier_surface", "string"), ("modifier_pos", "string"),
        ("head_surface", "string"), ("head_pos", "string"),
        ("direction", "string"),
        ("decade", "int32"), ("count", "int64"),
    ],
    "ner_snippets": [
        ("modifier_surface", "string"), ("modifier_pos", "string"),
        ("head_surface", "string"), ("head_pos", "string"),
        ("tokens_json", "string"),
        ("modifier_index", "int8"), ("head_index", "int8"),
        ("sentence_start", "bool"), ("sentence_end", "bool"),
        ("decade", "int32"), ("count", "int64"),
    ],
    "ner_predictions": [
        ("tokens_json", "string"),
        ("modifier_index", "int8"), ("head_index", "int8"),
        ("modifier_ner", "string"), ("head_ner", "string"), ("pair_ner", "string"),
        ("same_entity_span", "bool"),
    ],
    "lexicon": [
        ("surface", "string"),
        ("norm", "string"),
        ("pos", "string"),
        ("lemma", "string"),
        ("count", "int64"),
    ],
}


def arrow_schema(name, pa):
    type_map = {
        "string": pa.string(), "int8": pa.int8(), "int32": pa.int32(),
        "int64": pa.int64(), "bool": pa.bool_(), "float64": pa.float64(),
    }
    return pa.schema([(column, type_map[type_name]) for column, type_name in SCHEMAS[name]])


class ParquetSink(object):
    """Buffered atomic Parquet writer."""
    def __init__(self, path, schema_name, batch_size=50000):
        self.pa, self.pq = require_pyarrow()
        self.path = Path(path)
        self.temp_path = self.path.with_name(self.path.name + ".tmp")
        self.schema = arrow_schema(schema_name, self.pa)
        self.batch_size = batch_size
        self.rows = []
        self.writer = None
        self.count = 0

    def append(self, row):
        self.rows.append(row)
        self.count += 1
        if len(self.rows) >= self.batch_size:
            self.flush()

    def flush(self):
        if not self.rows:
            return
        table = self.pa.Table.from_pylist(self.rows, schema=self.schema)
        if self.writer is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.temp_path.unlink(missing_ok=True)
            self.writer = self.pq.ParquetWriter(
                str(self.temp_path), self.schema, compression="zstd",
                use_dictionary=True,
            )
        self.writer.write_table(table)
        self.rows = []

    def close(self):
        self.flush()
        if self.writer is None:
            self.temp_path.unlink(missing_ok=True)
            self.path.unlink(missing_ok=True)
            return False
        else:
            self.writer.close()
        os.replace(str(self.temp_path), str(self.path))
        return True


def load_context_vocabulary(path):
    if not path:
        return None
    norms = set()
    pos_pairs = set()
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle):
            fields = line.rstrip("\r\n").split("\t")
            if not fields or not fields[0]:
                continue
            if line_number == 0 and fields[0].casefold() == "norm":
                continue
            norm = normalize(fields[0])
            norms.add(norm)
            if len(fields) > 1 and fields[1] in VALID_TAGS:
                pos_pairs.add((norm, fields[1]))
    return {"norms": norms, "pos_pairs": pos_pairs}


def vocabulary_contains(vocabulary, norm, pos):
    """Accept an old norm-only vocabulary or a newer POS-specific one."""
    if vocabulary is None:
        return True
    if vocabulary["pos_pairs"]:
        return (norm, pos) in vocabulary["pos_pairs"]
    return norm in vocabulary["norms"]


def file_sha256(path):
    if not path:
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def add_counts(row, decade, counts):
    result = dict(row)
    result.update({
        "decade": decade,
        "count": counts,
    })
    return result


def extraction_settings(args, source, order, shard):
    context_pos = sorted(value for value in args.context_pos.split(",") if value)
    result = {
        "schema_version": 2, "corpus_version": CORPUS_VERSION,
        "source": source, "order": int(order), "shard": shard,
        "year_min": args.year_min, "year_max": args.year_max,
        "bin_size": args.bin_size, "context_pos": context_pos,
        "context_vocabulary_sha256": file_sha256(args.context_vocab),
        "ner_pair_offset": args.ner_pair_offset,
    }
    if int(order) in (2, 3, 4):
        result["word_target_pos"] = "all_lexical"
    if int(order) == 2:
        result["bigram_population"] = "all_fully_tagged_lexical_adjacencies"
        result["bigram_storage"] = "single_all_lexical_cased_surface_stream"
    if int(order) in (3, 4, 5):
        result["bigram_target_pos"] = "all_lexical"
    if int(order) in (2, 3, 4, 5):
        result["representation_version"] = 6
        result["context_surface_storage"] = "normalized_context_keys"
    result["count_measure"] = "match_count"
    return result


def extraction_output_names(order):
    names = {
        0: ["dependencies"],
        1: ["unigrams"],
        2: ["bigrams", "word_contexts"],
        3: ["contexts", "word_contexts"],
        4: ["contexts", "word_contexts"],
        5: ["contexts", "ner_snippets"],
    }
    return names[int(order)]


def validate_completed_task(done_path, expected, required_outputs=()):
    with open(done_path, "r", encoding="utf-8") as handle:
        observed = json.load(handle)
    mismatches = {
        key: (observed.get(key), value)
        for key, value in expected.items() if observed.get(key) != value
    }
    if mismatches:
        details = ", ".join(
            f"{key}: saved={old!r}, requested={new!r}"
            for key, (old, new) in sorted(mismatches.items())
        )
        raise RuntimeError(
            f"Completion marker {done_path} has incompatible settings ({details}); "
            "use --force to replace this task"
        )
    declared = {
        Path(item["path"]).resolve(): item
        for item in observed.get("source_info", {}).get("outputs", [])
        if item.get("path")
    }
    missing = []
    changed_size = []
    for output_path in map(Path, required_outputs):
        row_key = output_path.stem + "_rows"
        expected_rows = observed.get(row_key)
        if expected_rows is not None and int(expected_rows) == 0:
            if output_path.exists():
                changed_size.append(f"unexpected empty output: {output_path}")
            continue
        if not output_path.is_file():
            missing.append(str(output_path))
            continue
        record = declared.get(output_path.resolve())
        if record and record.get("bytes") != output_path.stat().st_size:
            changed_size.append(str(output_path))
    if missing or changed_size:
        details = []
        if missing:
            details.append("missing: " + ", ".join(missing))
        if changed_size:
            details.append("size changed: " + ", ".join(changed_size))
        raise RuntimeError(
            f"Completion marker {done_path} does not match its outputs "
            f"({' ; '.join(details)}); use --force to rebuild this task"
        )
    return observed

def completion_marker(output, order, shard):
    return Path(output) / "completion" / f"order={order}" / f"shard={shard}.json"


def task_data_dir(output, order, shard):
    return Path(output) / "raw" / f"order={order}" / f"shard={shard}"


def extract_source(args, source, order, shard):
    task_dir = task_data_dir(args.output, order, shard)
    done_path = completion_marker(args.output, order, shard)
    expected = extraction_settings(args, source, order, shard)
    required_outputs = [task_dir / f"{name}.parquet"
                        for name in extraction_output_names(order)]
    if done_path.exists() and not args.force:
        try:
            validate_completed_task(done_path, expected, required_outputs)
        except RuntimeError:
            if not args.replace_incompatible:
                raise
            print(f"Replacing incompatible completed task: {task_dir}", file=sys.stderr)
            done_path.unlink(missing_ok=True)
        else:
            print(f"Already complete: {task_dir}")
            return
    if args.force:
        done_path.unlink(missing_ok=True)

    sink_names = {order: extraction_output_names(order)}
    sinks = {
        name: ParquetSink(task_dir / (name + ".parquet"), name, args.batch_size)
        for name in sink_names[order]
    }
    context_pos = set(value for value in args.context_pos.split(",") if value)
    context_vocab = load_context_vocabulary(args.context_vocab)
    stats = defaultdict(int)
    started = time.time()

    opened_source = None
    try:
        with open_text_source(source, args.timeout) as handle:
            opened_source = handle
            for line_number, line in enumerate(handle, 1):
                stats["records_seen"] += 1
                try:
                    ngram, measurements = parse_v3_record(line)
                except ValueError:
                    stats["invalid_records"] += 1
                    continue
                decade_counts = aggregate_measurements(
                    measurements, args.year_min, args.year_max, args.bin_size
                )
                if not decade_counts:
                    continue

                if order == 0:
                    events_by_sink = {"dependencies": dependency_events(ngram)}
                else:
                    raw_tokens = ngram.split(" ")
                    if len(raw_tokens) != order:
                        stats["wrong_order_records"] += 1
                        continue
                    tokens = [parse_tagged_token(raw) for raw in raw_tokens]
                    if order == 1:
                        events_by_sink = {"unigrams": unigram_events(tokens)}
                    elif order == 2:
                        word_contexts = word_context_events(tokens, context_pos)
                        if context_vocab is not None:
                            word_contexts = [
                                event for event in word_contexts
                                if vocabulary_contains(
                                    context_vocab, event["context_norm"], event["context_pos"]
                                )
                            ]
                        events_by_sink = {
                            "bigrams": bigram_events(tokens),
                            "word_contexts": word_contexts,
                        }
                    else:
                        contexts = context_events(tokens, context_pos)
                        if context_vocab is not None:
                            contexts = [
                                event for event in contexts
                                if vocabulary_contains(
                                    context_vocab, event["context_norm"], event["context_pos"]
                                )
                            ]
                        events_by_sink = {"contexts": contexts}
                        if order in (3, 4):
                            word_contexts = word_context_events(tokens, context_pos)
                            if context_vocab is not None:
                                word_contexts = [
                                    event for event in word_contexts
                                    if vocabulary_contains(
                                        context_vocab, event["context_norm"], event["context_pos"]
                                    )
                                ]
                            events_by_sink["word_contexts"] = word_contexts
                        if order == 5:
                            events_by_sink["ner_snippets"] = ner_snippet_events(
                                tokens, args.ner_pair_offset
                            )

                for sink_name, events in events_by_sink.items():
                    stats[sink_name + "_ngram_events"] += len(events)
                    for event in events:
                        for decade, counts in decade_counts.items():
                            sinks[sink_name].append(add_counts(event, decade, counts))
                            stats[sink_name + "_rows"] += 1

                if args.progress_every and line_number % args.progress_every == 0:
                    elapsed = max(time.time() - started, 0.001)

                    print(
                        f"{shard}: {line_number:,} records ({line_number / elapsed:,.0f}/s)",
                        file=sys.stderr,
                    )

    except Exception:
        for sink in sinks.values():
            if sink.writer is not None:
                sink.writer.close()
        raise

    populated_sinks = []
    for name, sink in sinks.items():
        stats.setdefault(name + "_rows", 0)
        stats.setdefault(name + "_ngram_events", 0)
        if sink.close():
            populated_sinks.append(name)
    # Remove obsolete/empty products only after the replacement extraction
    # completed successfully. Empty tasks retain only their compact marker.
    for name in set(REDUCE_GROUPS).union({"pairs", "bigram_annotations"}):
        path = task_dir / f"{name}.parquet"
        if name not in populated_sinks:
            path.unlink(missing_ok=True)
    if task_dir.exists() and not any(task_dir.iterdir()):
        task_dir.rmdir()
    stats.update(expected)
    stats.update({
        "context_vocabulary": args.context_vocab,
        "source_bytes": opened_source.source_bytes,
        "source_sha256": opened_source.source_sha256,
        "elapsed_seconds": round(time.time() - started, 3),
    })
    source_input = external_artifact(
        f"compressed_shard/{shard}.gz", source,
        bytes=opened_source.source_bytes, sha256=opened_source.source_sha256,
    )
    inputs = [source_input]
    if args.manifest:
        inputs.append(artifact(args.manifest, "task_manifest"))
    if args.context_vocab:
        inputs.append(artifact(args.context_vocab, "context_vocabulary"))
    outputs = [artifact(sinks[name].path, sinks[name].path.name)
               for name in populated_sinks]
    scientific = {
        "representation_version": expected.get("representation_version"),
        "normalization": "NFKC plus casefold for context keys; cased surfaces elsewhere",
        "lexical_token": "fully POS-annotated non-wildcard token",
        "target_eligibility": "adjacent ADJ-NOUN or NOUN-NOUN",
        "word_target_pos": ("all lexical POS excluding punctuation" if order in (2, 3, 4)
                            else "not applicable"),
        "bigram_population": ("all fully tagged lexical adjacencies excluding punctuation"
                              if order == 2 else "not applicable"),
        "bigram_target_pos": ("all lexical POS excluding punctuation"
                              if order in (3, 4, 5) else "not applicable"),
        "bigram_storage": ("one cased surface/POS stream for counts and annotations"
                           if order == 2 else "not applicable"),
        "context_geometry": "shortest 3/4/5-gram edge for pair distances 1/2/3; "
                            "shortest 2/3/4-gram edge for word distances 1/2/3",
        "context_pos": sorted(context_pos),
        "context_vocabulary_sha256": expected["context_vocabulary_sha256"],
        "context_surface_storage": ("normalized form/POS only; cased pair surfaces "
                                    "stored once in order-2 bigrams"
                                    if order in (2, 3, 4, 5) else "not applicable"),
        "count_measure": "Google match_count renamed count; volume_count discarded",
        "ner_pair_offset": args.ner_pair_offset,
    }
    numeric_stats = {key: value for key, value in stats.items()
                     if key.endswith("_rows") or key.endswith("_events")
                     or key in {"records_seen", "invalid_records", "wrong_order_records"}}
    stats["source_info"] = source_info(
        "extract_shard", corpus_record(), inputs, scientific,
        time_record(args.year_min, args.year_max, args.bin_size),
        pipeline_software(packages=["pyarrow"]), outputs=outputs,
        statistics=numeric_stats,
        execution={"batch_size": args.batch_size,
                   "network_timeout_seconds": args.timeout,
                   "elapsed_seconds": stats["elapsed_seconds"]},
    )
    write_json(done_path, dict(stats))
    if populated_sinks:
        print(f"Completed {task_dir}")
    else:
        print(f"Completed empty shard; marker only: {done_path}")


RETRYABLE_SOURCE_ERRORS = (
    EOFError,
    TimeoutError,
    ConnectionError,
    socket.timeout,
    urllib.error.URLError,
    http.client.IncompleteRead,
    getattr(gzip, "BadGzipFile", EOFError),
)


def extract_source_with_retries(args, source, order, shard):
    """Retry transient remote truncation without accepting partial Parquet files."""
    attempts = max(1, int(args.retries) + 1)
    for attempt in range(1, attempts + 1):
        try:
            return extract_source(args, source, order, shard)
        except RETRYABLE_SOURCE_ERRORS as error:
            if not source.startswith(("http://", "https://")) or attempt == attempts:
                raise
            delay = max(0.0, float(args.retry_delay)) * attempt
            print(
                f"Transient source error for {shard} on attempt {attempt}/{attempts}: "
                f"{type(error).__name__}: {error}; retrying in {delay:g}s",
                file=sys.stderr,
                flush=True,
            )
            if delay:
                time.sleep(delay)


def sql_quote(value):
    return str(value).replace("'", "''")


REDUCE_GROUPS = {
    "unigrams": ["surface", "pos", "decade"],
    "bigrams": [
        "modifier_surface", "modifier_pos", "head_surface", "head_pos", "decade",
    ],
    "contexts": [
        "modifier_norm", "modifier_pos", "head_norm", "head_pos",
        "context_norm", "context_pos", "decade",
    ],
    "word_contexts": [
        "word_norm", "word_pos", "context_norm", "context_pos", "decade",
    ],
    "dependencies": [
        "modifier_surface", "modifier_pos",
        "head_surface", "head_pos", "direction", "decade",
    ],
    "ner_snippets": [
        "modifier_surface", "modifier_pos",
        "head_surface", "head_pos", "tokens_json",
        "modifier_index", "head_index", "sentence_start", "sentence_end",
        "decade",
    ],
}


def validate_reduction_inputs(output, manifest_paths, allow_incomplete=False):
    """Reject incomplete manifests and incompatible extraction settings."""
    output = Path(output)
    missing = []
    requested = []
    manifest_values = []
    for manifest_path in manifest_paths or []:
        with open(manifest_path, "r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        manifest_values.append((Path(manifest_path), manifest))
        for entry in manifest.get("entries", []):
            stats_path = completion_marker(output, entry["order"], entry["shard"])
            if not stats_path.exists():
                missing.append(str(stats_path))
            else:
                requested.append(stats_path)
    if missing and not allow_incomplete:
        preview = "\n".join(missing[:10])
        raise RuntimeError(
            f"{len(missing)} manifest tasks are incomplete. First missing outputs:\n{preview}\n"
            "Use --allow-incomplete only for an intentional partial reduction."
        )

    selected = requested if manifest_paths else sorted(
        output.glob("completion/order=*/shard=*.json")
    )
    stats_records = []
    for stats_path in selected:
        with open(stats_path, "r", encoding="utf-8") as handle:
            stats_records.append((stats_path, json.load(handle)))
    if not stats_records:
        raise RuntimeError(f"no completed extraction tasks found under {output}")

    global_keys = ("schema_version", "corpus_version", "year_min", "year_max", "bin_size")
    reference_path, reference = stats_records[0]
    for stats_path, stats in stats_records[1:]:
        for key in global_keys:
            if stats.get(key) != reference.get(key):
                raise RuntimeError(
                    f"incompatible {key}: {reference_path} has {reference.get(key),!r}, but {stats_path} has {stats.get(key),!r}"
                )

    context_records = [
        item for item in stats_records
        if int(item[1].get("order", -1)) in (2, 3, 4, 5)
    ]
    if context_records:
        context_path, context_reference = context_records[0]
        context_keys = ("context_pos", "context_vocabulary_sha256")
        for stats_path, stats in context_records[1:]:
            for key in context_keys:
                if stats.get(key) != context_reference.get(key):
                    raise RuntimeError(
                        f"incompatible context setting {key}: {context_path} has {context_reference.get(key),!r}, but {stats_path} has {stats.get(key),!r}"
                    )
    for stats_path, stats in stats_records:
        order = int(stats.get("order", -1))
        if order in (2, 3, 4) and stats.get("word_target_pos") != "all_lexical":
            raise RuntimeError(
                f"{stats_path} lacks all-lexical general word contexts; "
                "rerun context extraction with the current pipeline and --force"
            )
        if order == 2 and stats.get("bigram_population") != "all_fully_tagged_lexical_adjacencies":
            raise RuntimeError(
                f"{stats_path} lacks the all-lexical bigram association population; "
                "rerun order 2 with the current pipeline and --force"
            )
        if order == 2 and stats.get("bigram_storage") != "single_all_lexical_cased_surface_stream":
            raise RuntimeError(
                f"{stats_path} lacks the compact all-lexical cased bigram stream; "
                "rerun order 2 with the current pipeline and --force"
            )
        if order in (3, 4, 5) and stats.get("bigram_target_pos") != "all_lexical":
            raise RuntimeError(
                f"{stats_path} lacks all-lexical bigram contexts; "
                "rerun context extraction with the current pipeline and --force"
            )
        if order in (2, 3, 4, 5) and stats.get("context_surface_storage") != "normalized_context_keys":
            raise RuntimeError(
                f"{stats_path} uses an incompatible context-row layout; "
                "rerun context extraction with the current pipeline and --force"
            )
        if order in (2, 3, 4, 5) and stats.get("representation_version") != 6:
            raise RuntimeError(
                f"{stats_path} is not representation version 6; "
                "rerun context extraction with the current pipeline and --force"
            )
        if stats.get("count_measure") != "match_count":
            raise RuntimeError(
                f"{stats_path} does not use the compact match-count specification; "
                "rerun extraction with the current pipeline and --force"
            )
    ner_records = [item for item in stats_records if int(item[1].get("order", -1)) == 5]
    if ner_records:
        ner_path, ner_reference = ner_records[0]
        for stats_path, stats in ner_records[1:]:
            if stats.get("ner_pair_offset") != ner_reference.get("ner_pair_offset"):
                raise RuntimeError(
                    f"incompatible ner_pair_offset: {ner_path} has "
                    f"{ner_reference.get('ner_pair_offset')!r}, but {stats_path} has "
                    f"{stats.get('ner_pair_offset')!r}"
                )
    if manifest_values:
        corpora = {manifest.get("corpus") for _, manifest in manifest_values}
        releases = {manifest.get("corpus_version") for _, manifest in manifest_values}
        if len(corpora) != 1 or len(releases) != 1:
            raise RuntimeError("reduction manifests describe different corpora or releases")
        expected_entries = {
            (int(entry["order"]), entry["shard"]): entry
            for _, manifest in manifest_values for entry in manifest.get("entries", [])
        }
        for stats_path, stats in stats_records:
            key = (int(stats["order"]), stats["shard"])
            entry = expected_entries[key]
            if stats.get("source") != entry.get("source"):
                raise RuntimeError(
                    f"source mismatch for {stats_path}: {stats.get('source')!r} != {entry.get('source')!r}"
                )
    return stats_records, missing, manifest_values


def completed_reduction_is_current(path, output, stats_records, manifest_values, selected_tables):
    """Quickly validate a completed reduction without rereading large Parquet files."""
    path = Path(path)
    if not path.is_file() or not manifest_values:
        return False, "no completed manifest to reuse"
    try:
        with path.open("r", encoding="utf-8") as handle:
            saved = json.load(handle)
        block = saved["source_info"]
        if block.get("stage") != "reduce" or block.get("status") != "complete":
            return False, "saved reduction is not marked complete"

        selected = sorted(str(manifest_path.resolve())
                          for manifest_path, unused in manifest_values)
        if sorted(block.get("parameters", {}).get("selected_manifests", [])) != selected:
            return False, "selected task manifests changed"
        saved_tables = block.get("parameters", {}).get(
            "selected_tables", sorted(REDUCE_GROUPS)
        )
        if sorted(saved_tables) != sorted(selected_tables):
            return False, "selected reduction tables changed"

        saved_inputs = {item.get("path"): item for item in block.get("inputs", [])}
        for manifest_path, unused in manifest_values:
            resolved = str(manifest_path.resolve())
            record = saved_inputs.get(resolved)
            if not record or record.get("bytes") != manifest_path.stat().st_size:
                return False, f"task manifest changed: {manifest_path}"
            if record.get("sha256") != file_sha256(manifest_path):
                return False, f"task manifest changed: {manifest_path}"

        inventory_record = next(
            (item for item in block.get("inputs", [])
             if item.get("name") == "completion_inventory"), None
        )
        if not inventory_record:
            return False, "completion inventory is not recorded"
        inventory_path = Path(inventory_record["path"])
        if (not inventory_path.is_file()
                or inventory_record.get("bytes") != inventory_path.stat().st_size
                or inventory_record.get("sha256") != file_sha256(inventory_path)):
            return False, "completion inventory changed"
        with inventory_path.open("r", encoding="utf-8") as handle:
            inventory = json.load(handle)
        saved_stats = {
            item["stats"]["name"]: item["stats"]
            for item in inventory.get("tasks", [])
        }
        current_names = {str(stats_path.relative_to(output))
                         for stats_path, unused in stats_records}
        if set(saved_stats) != current_names:
            return False, "completed task set changed"
        for stats_path, unused in stats_records:
            name = str(stats_path.relative_to(output))
            record = saved_stats[name]
            if (record.get("bytes") != stats_path.stat().st_size
                    or record.get("sha256") != file_sha256(stats_path)):
                return False, f"completion marker changed: {name}"

        for record in block.get("outputs", []):
            output_path = Path(record["path"])
            if (not output_path.is_file()
                    or record.get("bytes") != output_path.stat().st_size):
                return False, f"reduced output missing or resized: {output_path}"
    except (KeyError, OSError, ValueError, TypeError, json.JSONDecodeError) as error:
        return False, f"saved reduction cannot be validated: {error}"
    return True, "manifest, task markers, and output sizes match"


def reduce_tables(args):
    require_duckdb()
    output = Path(args.output)
    stats_records, missing, manifest_values = validate_reduction_inputs(
        output, args.manifest, args.allow_incomplete
    )
    print(
        f"Validated {len(stats_records):,} completed tasks; {len(missing):,} requested tasks missing"
    )
    reduced_dir = output / "reduced"
    reduced_dir.mkdir(parents=True, exist_ok=True)
    selected_tables = list(dict.fromkeys(args.table or REDUCE_GROUPS.keys()))
    reduction_manifest_path = reduced_dir / "reduction_manifest.json"
    if not args.force:
        current, reason = completed_reduction_is_current(
            reduction_manifest_path, output, stats_records, manifest_values, selected_tables
        )
        if current:
            print(f"Already complete: reduction ({reason})")
            return
        if reduction_manifest_path.exists():
            print(f"Rebuilding reduction: {reason}")
    reduction_manifest_path.unlink(missing_ok=True)

    inventory = []
    for stats_path, stats in stats_records:
        validate_completed_task(
            stats_path, {},
            [task_data_dir(output, stats["order"], stats["shard"]) / f"{name}.parquet"
             for name in extraction_output_names(stats["order"])],
        )
        declared = {record["path"]: record for record in
                    stats.get("source_info", {}).get("outputs", [])}
        item = {
            "order": stats["order"], "shard": stats["shard"],
            "source": stats["source"],
            "stats": artifact(stats_path, str(stats_path.relative_to(output))),
            "outputs": [],
        }
        for table_name in REDUCE_GROUPS:
            path = task_data_dir(output, stats["order"], stats["shard"]) / f"{table_name}.parquet"
            if path.exists():
                # Extraction already hashes immutable shard products. Reuse
                # their hashes after validating sizes instead of rereading
                # hundreds of GiB just to construct a reduction inventory.
                source_hash = declared.get(str(path.resolve()), {}).get("sha256")
                item["outputs"].append(artifact(
                    path, str(path.relative_to(output)), sha256=source_hash))
        inventory.append(item)
    inventory_path = reduced_dir / "completion_inventory.json"
    write_json(inventory_path, {"schema_version": 1, "tasks": inventory})

    output_records = []
    table_statistics = {}
    temp_directory = Path(args.temp_directory or (output / "duckdb_tmp"))
    temp_directory.mkdir(parents=True, exist_ok=True)
    reduce_threads = args.threads or int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
    options = partition_options(args)
    options["threads"] = int(reduce_threads)
    work_root = Path(getattr(args, "work_directory", None)
                     or (temp_directory / "partitioned"))
    for table_name in selected_tables:
        groups = REDUCE_GROUPS[table_name]
        destination = reduced_dir / (table_name + ".parquet")
        source_records = [record for task in inventory for record in task["outputs"]
                          if Path(record["path"]).stem == table_name]
        sources = [Path(record["path"]) for record in source_records]
        if not sources:
            destination.unlink(missing_ok=True)
            print(f"Skipping {table_name}: no fragments")
            continue
        group_sql = ", ".join(groups)
        table_statistics[table_name] = partitioned_sum(
            sources, destination, groups,
            f"SELECT {group_sql}, count FROM partition_input", [*groups, "count"],
            options=options, work_root=work_root / "reduce",
            spill_root=temp_directory / "spill", identity=source_records,
            force=args.force, time_source_column="decade",
        )
        output_records.append(artifact(destination, f"reduced/{destination.name}"))
        print(f"Wrote {destination}")

    if {"bigrams", "ner_snippets"}.intersection(selected_tables):
        # NER predictions are snippet-text classifications and may be reused,
        # but their count-weighted surface/decade summary must be recomputed.
        (reduced_dir / "pair_ner_summary.parquet").unlink(missing_ok=True)

    reference = stats_records[0][1]
    context_records = [
        (path, stats) for path, stats in stats_records
        if int(stats.get("order", -1)) in (2, 3, 4, 5)
    ]
    context_reference = next(
        (stats for unused_path, stats in context_records),
        reference,
    )
    corpus_name = (manifest_values[0][1].get("corpus")
                   if manifest_values else DEFAULT_CORPUS)
    inputs = [artifact(path, f"task_manifest/{path.name}")
              for path, unused in manifest_values]
    inputs.append(artifact(inventory_path, "completion_inventory"))
    status = "partial" if missing or not manifest_values else "complete"
    reduction_parameters = {
        "allow_incomplete": args.allow_incomplete,
        "selected_manifests": [str(path.resolve()) for path, unused in manifest_values],
        "selected_tables": selected_tables,
        "context_pos": context_reference.get("context_pos"),
        "context_vocabulary": context_reference.get("context_vocabulary"),
        "context_vocabulary_sha256": context_reference.get("context_vocabulary_sha256"),
        "normalization": "NFKC plus casefold for context keys; cased surface retained elsewhere",
        "aggregation": "sum Google match_count (stored as count); pool context side/distance",
    }
    if context_records:
        reduction_parameters.update({
            "word_target_pos": "all_lexical",
            "bigram_target_pos": "all_lexical",
            "bigram_population": "all_fully_tagged_lexical_adjacencies",
            "bigram_storage": "single_all_lexical_cased_surface_stream",
            "context_surface_storage": "normalized_context_keys",
            "representation_version": 6,
            "count_measure": "match_count",
        })
    reduction_manifest = {
        "schema_version": 1,
        "tables": {record["name"].split("/")[-1].removesuffix(".parquet"): record
                   for record in output_records},
        "source_info": source_info(
            "reduce", corpus_record(corpus_name), inputs,
            reduction_parameters,
            time_record(reference.get("year_min"), reference.get("year_max"),
                        reference.get("bin_size")),
            pipeline_software(packages=["duckdb", "pyarrow"]),
            outputs=output_records, statistics={"tasks": len(stats_records),
                                                "missing_tasks": len(missing),
                                                "tables": table_statistics},
            execution={"adapter": "compact time/hash partitions with streamed concatenation",
                       "database": "fresh in-memory connection per aggregation partition",
                       "threads": reduce_threads,
                       "memory_limit": args.memory_limit or "DuckDB automatic limit",
                       "preserve_insertion_order": False,
                       "temp_directory": str(temp_directory.resolve()),
                       "work_directory": str(work_root.resolve()),
                       "partitions": options["partitions"],
                       "partition_options": options,
                       "aggregation_workers": options["aggregation_workers"],
                       "time_planning": "bounded source decade-column samples; no lemma joins",
                       "partition_batch_rows": options["batch_rows"],
                       "max_split_depth": options["max_split_depth"],
                       "max_temp_directory_size": options["max_temp_directory_size"]}, status=status,
        ),
    }
    write_json(reduction_manifest_path, reduction_manifest)
    print(f"Wrote {reduction_manifest_path}")


GOOGLE_TO_UNIVERSAL_POS = {
    "CONJ": "CCONJ",
    "PRT": "PART",
    ".": "PUNCT",
}


def resolve_spacy_model_data(model):
    """Resolve a spaCy package/path without constructing its full pipeline."""
    supplied = Path(model)
    if supplied.exists():
        roots = [supplied]
    else:
        try:
            package = importlib.import_module(model)
        except ImportError as error:
            raise RuntimeError(f"spaCy model package {model!r} is not installed") from error
        roots = [Path(package.__file__).resolve().parent]
    candidates = []
    for root in roots:
        candidates.append(root)
        if root.is_dir():
            candidates.extend(path for path in root.iterdir() if path.is_dir())
    for candidate in candidates:
        if (candidate / "meta.json").is_file() and (candidate / "vocab").is_dir() \
                and (candidate / "lemmatizer" / "lookups" / "lookups.bin").is_file():
            return candidate
    raise RuntimeError(
        f"Could not find model data with rule lemmatizer lookups under {model!r}"
    )


def lemmatize_vocabulary(args):
    """Lemmatize each distinct cased surface/Google-POS type once."""
    try:
        import spacy
        from spacy.lang.en.lemmatizer import EnglishLemmatizer
        from spacy.lookups import Lookups
        from spacy.tokens import Doc
        from spacy.vocab import Vocab
    except ImportError:
        raise RuntimeError(
            "spaCy is required for lemmatization: install the ner extra"
        )
    duckdb = require_duckdb()
    input_path = Path(args.output) / "reduced" / "unigrams.parquet"
    if not input_path.exists():
        raise RuntimeError(f"reduce unigrams first: {input_path} not found")
    destination = Path(args.lexicon or (Path(args.output) / "reduced" / "lexicon.parquet"))
    model_path = resolve_spacy_model_data(args.model)
    model_meta = json.loads((model_path / "meta.json").read_text(encoding="utf-8"))
    lookup_path = model_path / "lemmatizer" / "lookups"
    lookups = Lookups().from_disk(lookup_path)
    required_lookups = {"lemma_rules", "lemma_exc", "lemma_index"}
    missing_lookups = required_lookups.difference(lookups.tables)
    if missing_lookups:
        raise RuntimeError(
            f"Model {args.model} lacks English rule-lemmatizer tables: "
            f"{', '.join(sorted(missing_lookups))}"
        )
    # Google already supplies POS. Loading the transformer's parser/tagger is
    # unnecessary and can activate unrelated plugins; use the model's exact
    # rule tables directly instead.
    connection = duckdb.connect()
    connection.execute("SET threads = 1")
    connection.execute("SET preserve_insertion_order = false")
    lemma_temp = Path(args.output) / "duckdb_tmp"
    lemma_temp.mkdir(parents=True, exist_ok=True)
    connection.execute(f"SET temp_directory = '{sql_quote(lemma_temp.resolve())}'")
    cursor = connection.execute(
        """
        SELECT surface, pos, SUM(count)::BIGINT AS count
        FROM read_parquet(?)
        WHERE surface IS NOT NULL AND pos IS NOT NULL
        GROUP BY surface, pos
        """,
        [str(input_path)],
    )
    sink = ParquetSink(destination, "lexicon", args.batch_size)
    processed = 0
    while True:
        rows = cursor.fetchmany(args.batch_size)
        if not rows:
            break
        # Bound spaCy's string-store growth. Lemmatization is independent for
        # each normalized form/POS type, so a fresh vocabulary per
        # output batch produces identical rows with stable memory use.
        vocab = Vocab()
        lemmatizer = EnglishLemmatizer(vocab, None, mode="rule", overwrite=False)
        lemmatizer.initialize(lookups=lookups)
        for surface, google_pos, count in rows:
            norm = normalize(surface)
            doc = Doc(vocab, words=[norm])
            universal_pos = GOOGLE_TO_UNIVERSAL_POS.get(google_pos, google_pos)
            doc[0].pos = vocab.strings.add(universal_pos)
            lemmatizer(doc)
            lemma = normalize(doc[0].lemma_ or norm)
            sink.append({
                "surface": surface,
                "norm": norm,
                "pos": google_pos,
                "lemma": lemma,
                "count": count,
            })
            processed += 1
        print(f"Lemmatized types: {processed:,}", file=sys.stderr)
    connection.close()
    sink.close()
    model_files = [model_path / "meta.json", lookup_path / "lookups.bin"]
    config_path = model_path / "config.cfg"
    if config_path.exists():
        model_files.append(config_path)
    model_hashes = {str(path.relative_to(model_path)): file_sha256(path)
                    for path in model_files}
    reduction_path = Path(args.output) / "reduced" / "reduction_manifest.json"
    parent = None
    input_hash = None
    if reduction_path.exists():
        with reduction_path.open("r", encoding="utf-8") as handle:
            reduction = json.load(handle)
        parent = reduction.get("source_info")
        for record in (parent or {}).get("outputs", []):
            if Path(record.get("path", "")).resolve() == input_path.resolve():
                input_hash = record.get("sha256")
                break
    output_record = artifact(destination, "lexicon.parquet")
    sidecar = {
        "schema_version": 1, "model": args.model,
        "spacy_version": spacy.__version__, "types": processed,
        "lemmatizer": "English rule lemmatizer loaded directly from model lookups",
        "source_info": source_info(
            "lemmatize", (parent or {}).get("corpus", corpus_record()),
            [artifact(input_path, "unigrams.parquet", sha256=input_hash)],
            {"input_key": ["surface", "pos"],
             "output_keys": [["surface", "pos"], ["norm", "pos"]],
             "normalization": "NFKC plus casefold", "mode": "rule",
             "pos_source": "Google Books V3 tags mapped to Universal POS where needed"},
            (parent or {}).get("time", time_record()),
            pipeline_software(packages=["spacy", "duckdb", "pyarrow"]),
            model={"name": args.model, "metadata": model_meta,
                   "files_sha256": model_hashes},
            outputs=[output_record], statistics={"types": processed},
            execution={"batch_size": args.batch_size,
                       "full_pipeline_loaded": False},
        ),
    }
    write_json(str(destination) + ".json", sidecar)
    print(f"Wrote {processed:,} lexical types to {destination}")


def make_vocabulary(args):
    """Write a fixed lemma/POS context vocabulary with equal decade weights."""
    duckdb = require_duckdb()
    input_path = Path(args.output) / "reduced" / "unigrams.parquet"
    if not input_path.exists():
        raise RuntimeError(f"reduce unigrams first: {input_path} not found")
    allowed = [value for value in args.context_pos.split(",") if value]
    if not allowed:
        raise RuntimeError("--context-pos must contain at least one POS tag")
    placeholders = ",".join(["?"] * len(allowed))
    default_lexicon = Path(args.output) / "reduced" / "lexicon.parquet"
    lexicon_path = Path(args.lexicon) if args.lexicon else default_lexicon
    if not lexicon_path.exists():
        raise RuntimeError(
            f"lemma lexicon not found: {lexicon_path}; run the lemmatize stage first"
        )
    connection = duckdb.connect()
    connection.execute(f"SET threads = {max(1, int(getattr(args, 'threads', 1)))}")
    connection.execute("SET preserve_insertion_order = false")
    temp_directory = Path(getattr(args, "temp_directory", None)
                          or (Path(args.output) / "duckdb_tmp"))
    temp_directory.mkdir(parents=True, exist_ok=True)
    connection.execute(f"SET temp_directory = '{sql_quote(temp_directory.resolve())}'")
    if getattr(args, "memory_limit", None):
        connection.execute(f"SET memory_limit = '{sql_quote(args.memory_limit)}'")
    common_sql = f"""
        WITH lexical AS (
            SELECT l.norm, u.pos, u.decade, l.lemma,
                   SUM(u.count)::BIGINT AS count
            FROM read_parquet(?) u
            JOIN read_parquet(?) l
              ON u.surface=l.surface AND u.pos=l.pos
            WHERE u.pos IN ({placeholders})
            GROUP BY l.norm, u.pos, u.decade, l.lemma
        ),
        lemma_decade AS (
            SELECT lemma, pos, decade,
                   SUM(count)::BIGINT AS count
            FROM lexical
            GROUP BY lemma, pos, decade
        ),
        decade_totals AS (
            SELECT decade, SUM(count)::DOUBLE AS decade_total
            FROM lemma_decade
            GROUP BY decade
        ),
        decade_count AS (
            SELECT COUNT(*)::DOUBLE AS n_decades FROM decade_totals
        ),
        scores AS (
            SELECT d.lemma, d.pos,
                   SUM(d.count / t.decade_total) / dc.n_decades AS score,
                   COUNT(*)::BIGINT AS decades_observed,
                   SUM(d.count)::BIGINT AS lemma_match_count
            FROM lemma_decade d
            JOIN decade_totals t USING (decade)
            CROSS JOIN decade_count dc
            GROUP BY d.lemma, d.pos, dc.n_decades
        ),
        ranked AS (
            SELECT lemma, pos, score, decades_observed, lemma_match_count,
                   ROW_NUMBER() OVER (ORDER BY score DESC, lemma, pos) AS rank
            FROM scores
        ),
        selected AS (
            SELECT * FROM ranked WHERE rank <= ?
        )
    """
    parameters = [str(input_path), str(lexicon_path)] + allowed + [args.top_k]
    rows = connection.execute(
        common_sql + """
        SELECT x.norm, x.pos, x.lemma, s.rank, s.score,
               s.decades_observed,
               SUM(x.count)::BIGINT AS surface_match_count,
               s.lemma_match_count
        FROM lexical x
        JOIN selected s USING (lemma, pos)
        GROUP BY x.norm, x.pos, x.lemma, s.rank, s.score,
                 s.decades_observed, s.lemma_match_count
        ORDER BY s.rank, surface_match_count DESC, x.norm
        """,
        parameters,
    ).fetchall()
    coverage_rows = connection.execute(
        common_sql + """
        SELECT d.decade,
               SUM(d.count)::BIGINT AS eligible_context_tokens,
               SUM(CASE WHEN s.lemma IS NOT NULL THEN d.count ELSE 0 END)::BIGINT
                   AS retained_context_tokens,
               COUNT(*)::BIGINT AS eligible_context_types,
               COUNT(s.lemma)::BIGINT AS retained_context_types
        FROM lemma_decade d
        LEFT JOIN selected s USING (lemma, pos)
        GROUP BY d.decade
        ORDER BY d.decade
        """,
        parameters,
    ).fetchall()
    selected_types = connection.execute(
        common_sql + "SELECT COUNT(*) FROM selected", parameters
    ).fetchone()[0]
    connection.close()
    destination = Path(args.vocab_output)
    destination.parent.mkdir(parents=True, exist_ok=True)

    temp_destination = destination.with_name(destination.name + ".tmp")
    with open(temp_destination, "w", encoding="utf-8") as handle:
        handle.write(
            "norm\tpos\tlemma\trank\tscore\tdecades_observed\t"
            "surface_match_count\tlemma_match_count\n"
        )
        for row in rows:
            handle.write("\t".join(map(str, row)) + "\n")

    os.replace(str(temp_destination), str(destination))

    coverage_path = Path(
        args.coverage_output or str(destination.with_suffix("")) + "_coverage.tsv"
    )
    coverage_path.parent.mkdir(parents=True, exist_ok=True)
    coverage_temp = coverage_path.with_name(coverage_path.name + ".tmp")
    with open(coverage_temp, "w", encoding="utf-8") as handle:
        handle.write(
            "time\teligible_context_tokens\tretained_context_tokens\t"
            "eligible_context_types\tretained_context_types\ttoken_coverage\n"
        )
        for decade, eligible, retained, eligible_types, retained_types in coverage_rows:
            ratio = retained / eligible if eligible else float("nan")
            handle.write(
                f"{decade}\t{eligible}\t{retained}\t{eligible_types}\t"
                f"{retained_types}\t{ratio}\n"
            )
    os.replace(str(coverage_temp), str(coverage_path))

    sidecar = Path(str(destination) + ".json")
    reduction_path = Path(args.output) / "reduced" / "reduction_manifest.json"
    inputs = [artifact(input_path, "unigrams.parquet"),
              artifact(lexicon_path, "lexicon.parquet")]
    if reduction_path.exists():
        inputs.insert(0, artifact(reduction_path, "reduction_manifest"))
    write_json(sidecar, {
        "schema_version": 2,
        "requested_size": args.top_k,
        "actual_size": int(selected_types),
        "selection": "mean within-decade relative frequency; absent decades contribute zero; lexical tie-break",
        "context_pos": allowed,
        "source_info": source_info(
            "context_vocabulary", corpus_record(), inputs,
            {"requested_size": args.top_k, "actual_size": int(selected_types),
             "selection": "equal-decade mean relative frequency",
             "context_pos": allowed, "fixed_across_settings_and_time_bins": True},
            time_record(), pipeline_software(packages=["duckdb"]),
            outputs=[artifact(destination, destination.name),
                     artifact(coverage_path, coverage_path.name)],
            statistics={"selected_lemma_pos_types": int(selected_types),
                        "exported_inflected_forms": len(rows),
                        "decades": len(coverage_rows)},
            execution={"threads": max(1, int(args.threads)),
                       "memory_limit": args.memory_limit or "DuckDB automatic limit",
                       "temp_directory": str(temp_directory.resolve())},
        ),
    })

    print(
        f"Wrote {len(rows):,} inflected forms for {selected_types:,} "
        f"equal-decade-ranked lemma/POS types to {destination}"
    )


def _write_query_table(connection, destination, query, parameters=()):
    destination = Path(destination)
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.unlink(missing_ok=True)
    connection.execute(
        f"COPY ({query}) TO '{sql_quote(temporary)}' "
        "(FORMAT PARQUET, COMPRESSION ZSTD)",
        list(parameters),
    )
    os.replace(str(temporary), str(destination))
    rows, count_sum = connection.execute(
        "SELECT COUNT(*), COALESCE(SUM(count), 0) FROM read_parquet(?)",
        [str(destination)],
    ).fetchone()
    return {"rows": int(rows), "count_sum": int(count_sum)}


def _reject_null_identifiers(connection, path, columns):
    predicate = " OR ".join(f'"{column}" IS NULL' for column in columns)
    missing = connection.execute(
        f"SELECT COUNT(*) FROM read_parquet(?) WHERE {predicate}", [str(path)]
    ).fetchone()[0]
    if missing:
        Path(path).unlink(missing_ok=True)
        raise RuntimeError(
            f"Strict lemmatization failed for {missing:,} rows in {path}; "
            "the lemma lexicon does not cover every normalized form/POS key"
        )


def _link_or_copy(source, destination):
    """Publish an immutable count table at a second canonical path."""
    source, destination = Path(source), Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.unlink(missing_ok=True)
    try:
        os.link(source, temporary)
    except OSError:
        shutil.copyfile(source, temporary)
    os.replace(str(temporary), str(destination))


def _prepare_lemma_tables(connection, lexicon):
    """Build the small lookup relations once, not once per shuffle batch."""
    lexicon_sql = sql_quote(Path(lexicon).resolve())
    conflicts = connection.execute(f"""
        SELECT COUNT(*) FROM (
            SELECT norm, pos FROM read_parquet('{lexicon_sql}')
            GROUP BY norm, pos HAVING COUNT(DISTINCT lemma) > 1
        )
    """).fetchone()[0]
    if conflicts:
        raise RuntimeError(
            f"lemma lexicon has {conflicts:,} normalized form/POS keys with conflicting lemmas"
        )
    connection.execute(f"""
        CREATE TABLE lemma_by_surface AS
        SELECT surface, pos, MIN(norm) AS norm, MIN(lemma) AS lemma
        FROM read_parquet('{lexicon_sql}') GROUP BY surface, pos
    """)
    connection.execute(f"""
        CREATE TABLE lemma_by_norm AS
        SELECT norm, pos, MIN(lemma) AS lemma
        FROM read_parquet('{lexicon_sql}') GROUP BY norm, pos
    """)


class _ExportLemmatizer:
    """Resolve only missing joins with the lexicon's frozen Google-POS rules.

    Higher-order ngrams are not a subset of the published unigram types.
    COALESCE calls this Arrow UDF only for missing lookups, avoiding another
    full-corpus DISTINCT/coverage scan. Nothing is added to unigram counts or
    the vocabulary lexicon. Both the result cache and spaCy string store are
    bounded, even when exporting billions of context rows.
    """

    def __init__(self, lexicon, model=None, cache_size=50000, reset_after=10000):
        self.cache = OrderedDict()
        self.cache_size = cache_size
        self.reset_after = reset_after
        self.computed = 0
        self.cache_hits = 0
        self._since_reset = reset_after
        self.model_record = None
        self.lookups = None
        sidecar = Path(str(lexicon) + ".json")
        saved = json.loads(sidecar.read_text()) if sidecar.is_file() else {}
        recorded = saved.get("source_info", {}).get("model") or {}
        model = model or recorded.get("name") or saved.get("model")
        if model:
            import spacy
            from spacy.lookups import Lookups

            model_path = resolve_spacy_model_data(model)
            rule_path = model_path / "lemmatizer" / "lookups" / "lookups.bin"
            rule_key = "lemmatizer/lookups/lookups.bin"
            rule_hash = file_sha256(rule_path)
            expected_hash = recorded.get("files_sha256", {}).get(rule_key)
            if recorded and not expected_hash:
                raise RuntimeError("Lemma coverage requires the original rule-table SHA-256 "
                                   "in the lexicon source information; rebuild/version that lexicon first")
            if expected_hash and expected_hash != rule_hash:
                raise RuntimeError("Lemma coverage model rules differ from the frozen lexicon; "
                                   "restore the original spaCy model rather than mixing lemmas")
            expected_version = saved.get("spacy_version")
            if expected_version and expected_version != spacy.__version__:
                raise RuntimeError("Lemma coverage spaCy version differs from the frozen lexicon: "
                                   f"{spacy.__version__} != {expected_version}")
            self.lookups = Lookups().from_disk(rule_path.parent)
            missing = {"lemma_rules", "lemma_exc", "lemma_index"} - set(self.lookups.tables)
            if missing:
                raise RuntimeError(f"Lemma coverage model lacks rule tables: {sorted(missing)}")
            self.model_record = {
                "name": str(model),
                "metadata": json.loads((model_path / "meta.json").read_text()),
                "files_sha256": {rule_key: rule_hash},
                "spacy_version": spacy.__version__,
            }
        self.identity = {
            "version": 1,
            "policy": "existing lexicon join, otherwise exact English rules with Google POS",
            "normalization": "NFKC plus casefold",
            "model": self.model_record,
        }

    def lemma(self, surface, google_pos):
        if surface is None or google_pos is None:
            raise RuntimeError("Strict lemmatization failed: null form/POS")
        if self.lookups is None:
            raise RuntimeError(
                "Strict lemmatization failed: missing lemma/POS join for "
                f"{surface!r}/{google_pos}; model not recorded in lexicon sidecar. "
                "Supply --lemma-model with the original lexicon's spaCy rule model."
            )
        if google_pos not in VALID_TAGS:
            raise RuntimeError(f"Strict lemmatization failed: unsupported Google POS {google_pos!r}")
        norm = normalize(surface)
        if not norm:
            raise RuntimeError("Strict lemmatization failed: empty normalized form")
        key = (norm, google_pos)
        if key in self.cache:
            self.cache_hits += 1
            self.cache.move_to_end(key)
            return self.cache[key]
        from spacy.lang.en.lemmatizer import EnglishLemmatizer
        from spacy.tokens import Doc
        from spacy.vocab import Vocab

        if self._since_reset >= self.reset_after:
            self.vocab = Vocab()
            self.lemmatizer = EnglishLemmatizer(self.vocab, None, mode="rule", overwrite=False)
            self.lemmatizer.initialize(lookups=self.lookups)
            self._since_reset = 0
        doc = Doc(self.vocab, words=[norm])
        doc[0].pos = self.vocab.strings.add(GOOGLE_TO_UNIVERSAL_POS.get(google_pos, google_pos))
        self.lemmatizer(doc)
        result = normalize(doc[0].lemma_ or norm)
        self.cache[key] = result
        if len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)
        self._since_reset += 1
        self.computed += 1
        return result

    def register(self, connection):
        import pyarrow as pa

        def resolve(forms, positions):
            return pa.array([self.lemma(form, pos) for form, pos in
                             zip(forms.to_pylist(), positions.to_pylist())], type=pa.string())

        connection.create_function("google_rule_lemma", resolve,
                                   ["VARCHAR", "VARCHAR"], "VARCHAR",
                                   type="arrow", null_handling="special")


def _lemma_lookup_database(lexicon_record, work_root, options, spill_root):
    """Cache compressed lookup tables on scratch and release builder memory."""
    import duckdb
    import fcntl

    identity = {"version": 1, "lexicon": lexicon_record, "duckdb": duckdb.__version__,
                "implementation_sha256": file_sha256(__file__)}
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    directory = Path(work_root) / f"lemma-lookups-{fingerprint[:24]}"
    directory.mkdir(parents=True, exist_ok=True)
    database = directory / "lookups.duckdb"
    marker = directory / "complete.json"
    with (directory / "lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f"Another process is building lemma lookups at {directory}") from None
        if database.exists() and marker.exists():
            saved = json.loads(marker.read_text())
            if (saved.get("identity") == identity
                    and saved.get("bytes") == database.stat().st_size
                    and saved.get("mtime_ns") == database.stat().st_mtime_ns):
                return database
        temporary = database.with_name("lookups.tmp.duckdb")
        temporary.unlink(missing_ok=True)
        Path(str(temporary) + ".wal").unlink(missing_ok=True)
        with connection_scope(options, spill_root, database=temporary) as connection:
            _prepare_lemma_tables(connection, lexicon_record["path"])
        os.replace(temporary, database)
        write_json(marker, {"identity": identity, "bytes": database.stat().st_size,
                            "mtime_ns": database.stat().st_mtime_ns})
        return database


def make_feature_inputs(args):
    """Write strict lemma/POS Parquet inputs using the corpus schema-v3 layout."""
    require_duckdb()
    root = Path(args.output)
    reduced_dir = root / "reduced"
    final_dir = root / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    reduction_path = reduced_dir / "reduction_manifest.json"
    required = {
        "bigrams": reduced_dir / "bigrams.parquet",
        "contexts": reduced_dir / "contexts.parquet",
        "word_contexts": reduced_dir / "word_contexts.parquet",
        "unigrams": reduced_dir / "unigrams.parquet",
        "lexicon": Path(args.lexicon or (reduced_dir / "lexicon.parquet")),
    }
    annotations_only = bool(getattr(args, "annotations_only", False))
    required_labels = ({"bigrams", "lexicon"} if annotations_only
                       else set(required))
    for label, path in required.items():
        if label not in required_labels:
            continue
        if not path.exists():
            raise RuntimeError(f"required {label} input not found: {path}")
    if not reduction_path.exists():
        raise RuntimeError(f"reduction source information not found: {reduction_path}")
    with open(reduction_path, "r", encoding="utf-8") as handle:
        reduction_manifest = json.load(handle)
    if (reduction_manifest.get("source_info", {}).get("status") != "complete"
            and not args.allow_partial):
        raise RuntimeError(
            "reduction is not marked complete; use --allow-partial only for an intentional provisional analysis"
        )

    extracted = root / "extracted"
    destination = Path(args.feature_output or (extracted / "feature_inputs" / "local5"))
    tables_dir = Path(getattr(args, "tables_output", None) or (extracted / "tables"))
    destination.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    # The corpus-neutral manifest describes the actual source and count units.
    manifest_path = destination / "corpus_manifest.json"
    lexicon = required["lexicon"]
    coverage = _ExportLemmatizer(lexicon, getattr(args, "lemma_model", None))
    print("Lemma coverage: " + ("bounded on-demand rules from " + coverage.model_record["name"]
                                if coverage.model_record else "lexicon only; no rule model recorded"),
          flush=True)
    manifest_path.unlink(missing_ok=True)
    paths = {key: str(value) for key, value in required.items()}

    queries = {
        "bigrams": """
            SELECT COALESCE(lm.lemma, google_rule_lemma(c.modifier_norm, c.modifier_pos))
                       || '_' || c.modifier_pos AS modifier,
                   COALESCE(lh.lemma, google_rule_lemma(c.head_norm, c.head_pos))
                       || '_' || c.head_pos AS head,
                   c.decade::BIGINT AS time,
                   COALESCE(lc.lemma, google_rule_lemma(c.context_norm, c.context_pos))
                       || '_' || c.context_pos AS context,
                   c.count::BIGINT AS count
            FROM partition_input c
            LEFT JOIN lemma_by_norm lm
              ON c.modifier_norm=lm.norm AND c.modifier_pos=lm.pos
            LEFT JOIN lemma_by_norm lh
              ON c.head_norm=lh.norm AND c.head_pos=lh.pos
            LEFT JOIN lemma_by_norm lc
              ON c.context_norm=lc.norm AND c.context_pos=lc.pos
        """,
        "bigram_counts": """
            SELECT COALESCE(lm.lemma, google_rule_lemma(b.modifier_surface, b.modifier_pos))
                       || '_' || b.modifier_pos AS modifier,
                   COALESCE(lh.lemma, google_rule_lemma(b.head_surface, b.head_pos))
                       || '_' || b.head_pos AS head,
                   b.decade::BIGINT AS time,
                   b.count::BIGINT AS count
            FROM partition_input b
            LEFT JOIN lemma_by_surface lm
              ON b.modifier_surface=lm.surface AND b.modifier_pos=lm.pos
            LEFT JOIN lemma_by_surface lh
              ON b.head_surface=lh.surface AND b.head_pos=lh.pos
        """,
        "unigrams": """
            SELECT COALESCE(lt.lemma, google_rule_lemma(w.word_norm, w.word_pos))
                       || '_' || w.word_pos AS word,
                   w.decade::BIGINT AS time,
                   COALESCE(lc.lemma, google_rule_lemma(w.context_norm, w.context_pos))
                       || '_' || w.context_pos AS context,
                   w.count::BIGINT AS count
            FROM partition_input w
            LEFT JOIN lemma_by_norm lt
              ON w.word_norm=lt.norm AND w.word_pos=lt.pos
            LEFT JOIN lemma_by_norm lc
              ON w.context_norm=lc.norm AND w.context_pos=lc.pos
        """,
        "unigram_counts": """
            SELECT COALESCE(l.lemma, google_rule_lemma(u.surface, u.pos))
                       || '_' || u.pos AS word,
                   u.decade::BIGINT AS time,
                   u.count::BIGINT AS count
            FROM partition_input u
            LEFT JOIN lemma_by_surface l
              ON u.surface=l.surface AND u.pos=l.pos
            WHERE u.pos <> '.'
        """,
    }
    parameters = {
        "bigrams": [paths["contexts"]],
        "bigram_counts": [paths["bigrams"]],
        "unigrams": [paths["word_contexts"]],
        "unigram_counts": [paths["unigrams"]],
    }
    columns = {
        "unigrams": ["word", "time", "context", "count"],
        "unigram_counts": ["word", "time", "count"],
        "bigrams": ["modifier", "head", "time", "context", "count"],
        "bigram_counts": ["modifier", "head", "time", "count"],
    }
    units = {
        "unigrams": "all_lexical_unigram_context_events",
        "unigram_counts": "all_lexical_token_occurrences",
        "bigrams": "all_lexical_bigram_context_events",
        "bigram_counts": "all_lexical_bigram_occurrences",
    }
    table_records = {}
    statistics = {}
    temp_directory = Path(args.temp_directory or (root / "duckdb_tmp"))
    temp_directory.mkdir(parents=True, exist_ok=True)
    options = partition_options(args)
    work_root = Path(getattr(args, "work_directory", None)
                     or (temp_directory / "partitioned"))
    input_types = {"bigrams": "contexts", "bigram_counts": "bigrams",
                   "unigrams": "word_contexts", "unigram_counts": "unigrams"}
    # The checkpoint identity includes the lookup contents and upstream
    # source information. Surface forms sharing a lemma route to the same final bucket.
    export_identity = {"lexicon": artifact(lexicon),
                       "reduction_manifest": artifact(reduction_path),
                       "lemma_coverage": coverage.identity,
                       "adapter_sha256": file_sha256(__file__)}
    lexicon_sidecar = Path(str(lexicon) + ".json")
    if lexicon_sidecar.is_file():
        export_identity["lexicon_manifest"] = artifact(lexicon_sidecar)
    lookup_database = _lemma_lookup_database(
        export_identity["lexicon"], work_root / "feature-inputs", options,
        temp_directory / "spill",
    )
    identifier_columns = {
        "unigrams": ["word", "context"],
        "unigram_counts": ["word"],
        "bigrams": ["modifier", "head", "context"],
        "bigram_counts": ["modifier", "head"],
    }
    audit_records = {}
    for name, query in queries.items():
        feature_path = destination / f"{name}.parquet"
        if annotations_only:
            if not feature_path.exists():
                raise RuntimeError(
                    f"--annotations-only requires existing feature table: {feature_path}"
                )
            with connection_scope(options, temp_directory / "spill") as connection:
                rows, count_sum = connection.execute(
                    "SELECT COUNT(*), COALESCE(SUM(count), 0) FROM read_parquet(?)",
                    [str(feature_path)],
                ).fetchone()
            statistics[name] = {
                "rows": int(rows), "count_sum": int(count_sum), "reused": True,
            }
        else:
            path = feature_path if name == "bigram_counts" else final_dir / f"{name}.parquet"
            statistics[name] = partitioned_sum(
                parameters[name], path, columns[name][:-1], query,
                [*REDUCE_GROUPS[input_types[name]], "count"],
                options=options, work_root=work_root / "feature-inputs",
                spill_root=temp_directory / "spill", identity=export_identity,
                projection_database=lookup_database,
                projection_setup=coverage.register,
                required_identifiers=identifier_columns[name],
                preserve_count_mass=name != "unigram_counts",
                time_source_column="decade",
            )
            if path != feature_path:
                _link_or_copy(path, feature_path)
        table_records[name] = feature_table(
            feature_path, columns[name], units[name],
            name=f"local5/{feature_path.name}"
        )
        print(f"{'Reused' if annotations_only else 'Wrote'} {feature_path}")

    with connection_scope(options, temp_directory / "spill",
                          database=lookup_database, read_only=True) as connection:
        coverage.register(connection)
        ner_summary = reduced_dir / "pair_ner_summary.parquet"
        if ner_summary.exists():
            ner_select = """
                   n.modifier_ner, n.head_ner, n.pair_ner,
                   n.modifier_ner_confidence, n.head_ner_confidence,
                   n.pair_ner_confidence, n.same_entity_span_fraction,
                   n.ner_coverage, n.classified_ner_count,
                   n.sentence_start_fraction, n.sentence_end_fraction,
                   n.ner_distribution_json,"""
            ner_join = f"""
                LEFT JOIN read_parquet('{sql_quote(ner_summary.resolve())}') n
                  ON b.modifier_surface=n.modifier_surface
                 AND b.modifier_pos=n.modifier_pos
                 AND b.head_surface=n.head_surface
                 AND b.head_pos=n.head_pos
                 AND b.decade=n.time"""
        else:
            ner_select = """
                   CAST(NULL AS VARCHAR) AS modifier_ner,
                   CAST(NULL AS VARCHAR) AS head_ner,
                   CAST(NULL AS VARCHAR) AS pair_ner,
                   CAST(NULL AS DOUBLE) AS modifier_ner_confidence,
                   CAST(NULL AS DOUBLE) AS head_ner_confidence,
                   CAST(NULL AS DOUBLE) AS pair_ner_confidence,
                   CAST(NULL AS DOUBLE) AS same_entity_span_fraction,
                   CAST(NULL AS DOUBLE) AS ner_coverage,
                   CAST(NULL AS BIGINT) AS classified_ner_count,
                   CAST(NULL AS DOUBLE) AS sentence_start_fraction,
                   CAST(NULL AS DOUBLE) AS sentence_end_fraction,
                   CAST(NULL AS VARCHAR) AS ner_distribution_json,"""
            ner_join = ""
        annotation_query = f"""
            SELECT COALESCE(lm.lemma, google_rule_lemma(b.modifier_surface, b.modifier_pos))
                       || '_' || b.modifier_pos AS modifier,
                   COALESCE(lh.lemma, google_rule_lemma(b.head_surface, b.head_pos))
                       || '_' || b.head_pos AS head,
                   b.decade::BIGINT AS time,
                   b.modifier_surface, b.head_surface,
                   b.modifier_pos, b.head_pos,
                   {ner_select}
                   b.count::BIGINT AS count
            FROM read_parquet(?) b
            LEFT JOIN lemma_by_surface lm
              ON b.modifier_surface=lm.surface AND b.modifier_pos=lm.pos
            LEFT JOIN lemma_by_surface lh
              ON b.head_surface=lh.surface AND b.head_pos=lh.pos
            {ner_join}
            WHERE (b.modifier_pos='ADJ' AND b.head_pos='NOUN')
               OR (b.modifier_pos='NOUN' AND b.head_pos='NOUN')
        """
        annotation_path = final_dir / "bigram_annotations.parquet"
        statistics["bigram_annotations"] = _write_query_table(
            connection, annotation_path, annotation_query, [paths["bigrams"]],
        )
        _reject_null_identifiers(
            connection, annotation_path, ["modifier", "head"])
        annotation_export = tables_dir / "bigram_annotations.parquet"
        _link_or_copy(annotation_path, annotation_export)
        audit_records["bigram_annotations"] = feature_table(
            annotation_export,
            ["modifier", "head", "time", "modifier_surface", "head_surface",
             "modifier_pos", "head_pos", "modifier_ner", "head_ner", "pair_ner",
             "modifier_ner_confidence", "head_ner_confidence",
             "pair_ner_confidence", "same_entity_span_fraction", "ner_coverage",
             "classified_ner_count", "sentence_start_fraction",
             "sentence_end_fraction", "ner_distribution_json", "count"],
            "eligible_ADJ_NOUN_and_NOUN_NOUN_surface_form_occurrences",
            name="tables/bigram_annotations.parquet",
        )
        print(f"Wrote {annotation_path}")

    dependency_path = reduced_dir / "dependencies.parquet"
    if (not annotations_only and args.include_dependencies
            and dependency_path.exists()):
        query = f"""
            SELECT COALESCE(lm.lemma, google_rule_lemma(d.modifier_surface, d.modifier_pos))
                       || '_' || d.modifier_pos AS modifier,
                   COALESCE(lh.lemma, google_rule_lemma(d.head_surface, d.head_pos))
                       || '_' || d.head_pos AS head,
                   d.decade::BIGINT AS time, d.direction,
                   d.count::BIGINT AS count
            FROM partition_input d
            LEFT JOIN lemma_by_surface lm
              ON d.modifier_surface=lm.surface AND d.modifier_pos=lm.pos
            LEFT JOIN lemma_by_surface lh
              ON d.head_surface=lh.surface AND d.head_pos=lh.pos
        """
        path = tables_dir / "dependencies.parquet"
        statistics["dependencies"] = partitioned_sum(
            [dependency_path], path, ["modifier", "head", "time", "direction"],
            query, [*REDUCE_GROUPS["dependencies"], "count"],
            options=options, work_root=work_root / "feature-inputs",
            spill_root=temp_directory / "spill", identity=export_identity,
            projection_database=lookup_database,
            projection_setup=coverage.register,
            required_identifiers=["modifier", "head"],
            time_source_column="decade",
        )
        audit_records["dependencies"] = feature_table(
            path, ["modifier", "head", "time", "direction", "count"],
            "dependency_arc_occurrences", required=False,
            name=f"tables/{path.name}",
        )

    source_inputs = [artifact(reduction_path, "reduction_manifest"),
                     artifact(lexicon, "lemma_lexicon")]
    lexicon_sidecar = Path(str(lexicon) + ".json")
    model = None
    if lexicon_sidecar.exists():
        source_inputs.append(artifact(lexicon_sidecar, "lemma_lexicon_manifest"))
        with open(lexicon_sidecar, "r", encoding="utf-8") as handle:
            model = json.load(handle).get("source_info", {}).get("model")
    parent = reduction_manifest["source_info"]
    vocabulary_value = parent.get("parameters", {}).get("context_vocabulary")
    vocabulary_path = Path(vocabulary_value) if vocabulary_value else None
    vocabulary_sidecar = Path(str(vocabulary_path) + ".json") if vocabulary_path else None
    context_ceiling = None
    if vocabulary_path and vocabulary_path.is_file():
        source_inputs.append(artifact(vocabulary_path, "context_vocabulary"))
    if vocabulary_sidecar and vocabulary_sidecar.is_file():
        source_inputs.append(artifact(vocabulary_sidecar, "context_vocabulary_manifest"))
        with vocabulary_sidecar.open("r", encoding="utf-8") as handle:
            context_ceiling = json.load(handle).get("actual_size")
    manifest = {
        "schema_version": 3,
        "feature_input_schema_version": FEATURE_INPUT_SCHEMA_VERSION,
        "extraction_version": 5,
        "format": "parquet",
        "raw_counts": True, "context_mode": "local5",
        "classification": "all_lexical_adjacent_bigrams; eligible compounds derived before POS pooling",
        "context_dimensions": "lemma_POS; side and distance pooled",
        "context_extraction_ceiling": context_ceiling,
        "row_order": "unspecified; compare logical rows, not Parquet byte order",
        "time_column": "time",
        "word_target_pos": "all_lexical",
        "bigram_target_pos": "all_lexical",
        "count_population": "published fully tagged lexical Google n-gram occurrences, independent of retained contexts",
        "representation_version": 5,
        "table_files": {name: f"{name}.parquet" for name in
                        ["unigrams", "unigram_counts", "bigrams", "bigram_counts"]},
        "tables": {name: table_records[name] for name in
                   ["unigrams", "unigram_counts", "bigrams", "bigram_counts"]},
        "audit_tables": audit_records,
    }
    outputs = [{key: value[key] for key in ["name", "path", "bytes", "sha256"]}
               for value in [*manifest["tables"].values(), *audit_records.values()]]
    manifest["source_info"] = source_info(
        "feature_inputs", parent["corpus"], source_inputs,
        {"context_mode": "local5", "context_side_distance": "pooled by summation",
         "lemmatization": "surface/POS or normalized-form/POS join to the versioned lexicon; "
                          "missing keys computed with its frozen English rules and Google POS; "
                          "invalid keys or unavailable/mismatched rules fail export",
         "lemma_coverage": coverage.identity,
         "target_eligibility": manifest["classification"],
         "proper_nouns": "Google V3 tag inventory has no distinct PROPN tag",
         "unigram_population": "fully annotated lexical tokens excluding punctuation POS '.'",
         "bigram_population": "fully annotated adjacent lexical tokens excluding punctuation POS '.'",
         "word_target_pos": "all_lexical",
         "bigram_target_pos": "all_lexical",
         "context_extraction_ceiling": context_ceiling,
         "annotation_availability": {
             "surface_and_google_pos": "observed for eligible compound candidates",
             "ner": ("occurrence-weighted canonical five-gram estimates"
                     if ner_summary.exists() else "not run; fields null"),
             "document_counts": "not exported; count is Google match_count",
         },
         "context_vocabulary_sha256": parent["parameters"].get("context_vocabulary_sha256"),
         "table_files": manifest["table_files"],
         "row_order": manifest["row_order"]},
        parent["time"], pipeline_software(packages=["duckdb", "pyarrow", "spacy"]),
        model=model or coverage.model_record, outputs=outputs, statistics={"tables": statistics},
        execution={"adapter": "lemma-key compact time/hash partitions with streamed concatenation",
                   "lemma_coverage_cache_types": coverage.cache_size,
                   "lemma_string_store_reset_types": coverage.reset_after,
                   "lemma_coverage_computations_this_run": coverage.computed,
                   "lemma_coverage_cache_hits_this_run": coverage.cache_hits,
                   "threads": max(1, int(args.threads)),
                   "memory_limit": args.memory_limit or "DuckDB automatic limit",
                   "temp_directory": str(temp_directory.resolve()),
                   "work_directory": str(work_root.resolve()),
                   "partitions": options["partitions"],
                   "partition_options": options,
                   "aggregation_workers": options["aggregation_workers"],
                   "time_planning": "bounded source decade-column samples; no lemma joins",
                   "partition_batch_rows": options["batch_rows"],
                   "max_split_depth": options["max_split_depth"],
                   "max_temp_directory_size": options["max_temp_directory_size"],
                   "preserve_insertion_order": False},
        status="partial" if parent.get("status") == "partial" else "complete",
    )
    write_json(manifest_path, manifest)
    print(f"Wrote {manifest_path}")


def classify_pair_entities(doc, modifier_index, head_index):
    modifier = doc[modifier_index]
    head = doc[head_index]
    modifier_ner = modifier.ent_type_ or "NONE"
    head_ner = head.ent_type_ or "NONE"
    pair_ner = "NONE"
    same_span = False
    for entity in doc.ents:
        if entity.start <= modifier_index and entity.end > head_index:
            pair_ner = entity.label_
            same_span = True
            break
    return modifier_ner, head_ner, pair_ner, same_span


def load_target_pairs(path):
    """Read modifier/head pairs from a TSV or CSV file.

    A header containing ``modifier`` and ``head`` is preferred. Without a
    header, the first two columns are used. Additional rating columns are
    ignored.
    """
    if not path:
        return []
    with open(path, "r", encoding="utf-8", newline="") as handle:
        sample = handle.read(4096)
        handle.seek(0)
        delimiter = "\t" if sample.count("\t") >= sample.count(",") else ","
        rows = list(csv.reader(handle, delimiter=delimiter))
    if not rows:
        return []
    header = [value.strip().casefold() for value in rows[0]]
    if "modifier" in header and "head" in header:
        modifier_index = header.index("modifier")
        head_index = header.index("head")
        rows = rows[1:]
    else:
        modifier_index, head_index = 0, 1
    result = set()
    for row in rows:
        if len(row) <= max(modifier_index, head_index):
            continue
        modifier = row[modifier_index].strip()
        head = row[head_index].strip()
        if modifier and head:
            result.add((normalize(modifier), normalize(head)))
    return sorted(result)


def ner_predict(args):
    try:
        import spacy
        from spacy.tokens import Doc
    except ImportError:
        raise RuntimeError("spaCy is required for NER: install the ner extra")
    duckdb = require_duckdb()
    pa, unused_pq = require_pyarrow()
    del unused_pq

    snippets_path = Path(args.output) / "reduced" / "ner_snippets.parquet"
    if not snippets_path.exists():
        raise RuntimeError("reduce five-gram snippets first")
    destination = Path(args.predictions or (Path(args.output) / "reduced" / "ner_predictions.parquet"))
    sink = ParquetSink(destination, "ner_predictions", args.batch_size)
    spacy_model = spacy.load(args.model)
    connection = duckdb.connect()
    connection.execute(f"SET threads = {max(1, int(getattr(args, 'threads', 1)))}")
    connection.execute("SET preserve_insertion_order = false")
    temp_directory = Path(getattr(args, "temp_directory", None)
                          or (Path(args.output) / "duckdb_tmp"))
    temp_directory.mkdir(parents=True, exist_ok=True)
    connection.execute(f"SET temp_directory = '{sql_quote(temp_directory.resolve())}'")
    if getattr(args, "memory_limit", None):
        connection.execute(f"SET memory_limit = '{sql_quote(args.memory_limit)}'")
    target_pairs = sorted({
        pair
        for target_path in args.targets
        for pair in load_target_pairs(target_path)
    })
    target_join = ""
    parameters = [str(snippets_path)]
    if target_pairs:
        connection.execute(
            "CREATE TEMP TABLE ner_targets(modifier_norm VARCHAR, head_norm VARCHAR)"
        )
        connection.executemany("INSERT INTO ner_targets VALUES (?, ?)", target_pairs)
        target_join = """
            JOIN ner_targets t
              ON lower(s.modifier_surface)=t.modifier_norm
             AND lower(s.head_surface)=t.head_norm
        """
    cursor = connection.execute(
        f"""
        SELECT s.tokens_json, s.modifier_index, s.head_index
        FROM read_parquet(?) s
        {target_join}
        GROUP BY s.tokens_json, s.modifier_index, s.head_index
        HAVING SUM(s.count) >= ?
        ORDER BY s.tokens_json, s.modifier_index, s.head_index
        """,
        parameters + [args.min_snippet_count],
    )

    processed = 0
    while True:
        rows = cursor.fetchmany(args.ner_batch_size)
        if not rows:
            break
        docs = []
        keys = []
        for tokens_json, modifier_index, head_index in rows:
            words = json.loads(tokens_json)
            docs.append(Doc(spacy_model.vocab, words=words))
            keys.append((tokens_json, modifier_index, head_index))
        for key, doc in zip(
            keys,
            spacy_model.pipe(docs, batch_size=args.ner_batch_size, n_process=args.ner_processes),
        ):
            modifier_ner, head_ner, pair_ner, same_span = classify_pair_entities(
                doc, key[1], key[2]
            )
            sink.append({
                "tokens_json": key[0], "modifier_index": key[1],
                "head_index": key[2], "modifier_ner": modifier_ner,
                "head_ner": head_ner, "pair_ner": pair_ner,
                "same_entity_span": same_span,
            })
            processed += 1
        print(f"NER predictions: {processed:,}", file=sys.stderr)
    connection.close()
    sink.close()
    metadata = {
        "schema_version": 1, "model": args.model,
        "spacy_version": spacy.__version__, "predictions": processed,
        "targets": args.targets, "target_pairs": len(target_pairs),
        "min_snippet_count": args.min_snippet_count,
    }
    with open(str(destination) + ".json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"Wrote {destination}")


def ner_aggregate(args):
    duckdb = require_duckdb()
    output = Path(args.output)
    reduced = output / "reduced"
    snippets = reduced / "ner_snippets.parquet"
    bigrams = reduced / "bigrams.parquet"
    predictions = Path(args.predictions or (reduced / "ner_predictions.parquet"))
    for path in (snippets, bigrams, predictions):
        if not path.exists():
            raise RuntimeError(f"required file not found: {path}")
    destination = reduced / "pair_ner_summary.parquet"
    temp_destination = destination.with_name(destination.name + ".tmp")
    temp_destination.unlink(missing_ok=True)
    connection = duckdb.connect()
    connection.execute(f"SET threads = {max(1, int(getattr(args, 'threads', 1)))}")
    connection.execute("SET preserve_insertion_order = false")
    temp_directory = Path(getattr(args, "temp_directory", None)
                          or (output / "duckdb_tmp"))
    temp_directory.mkdir(parents=True, exist_ok=True)
    connection.execute(f"SET temp_directory = '{sql_quote(temp_directory.resolve())}'")
    if getattr(args, "memory_limit", None):
        connection.execute(f"SET memory_limit = '{sql_quote(args.memory_limit)}'")
    query = f"""
    WITH pair_totals AS (
        SELECT modifier_surface, modifier_pos, head_surface, head_pos, decade,
               SUM(count)::DOUBLE AS pair_count
        FROM read_parquet('{sql_quote(bigrams)}')
        WHERE (modifier_pos='ADJ' AND head_pos='NOUN')
           OR (modifier_pos='NOUN' AND head_pos='NOUN')
        GROUP BY ALL
    ), classified AS (
        SELECT s.modifier_surface, s.modifier_pos,
               s.head_surface, s.head_pos, s.decade,
               p.modifier_ner, p.head_ner, p.pair_ner, p.same_entity_span,
               s.sentence_start, s.sentence_end, s.count
        FROM read_parquet('{sql_quote(snippets)}') s
        JOIN read_parquet('{sql_quote(predictions)}') p
          USING (tokens_json, modifier_index, head_index)
    ), totals AS (
        SELECT modifier_surface, modifier_pos, head_surface, head_pos, decade,
               SUM(count)::DOUBLE AS classified_ner_count,
               SUM(CASE WHEN same_entity_span THEN count ELSE 0 END)::DOUBLE
                   AS same_span_count,
               SUM(CASE WHEN sentence_start THEN count ELSE 0 END)::DOUBLE
                   AS sentence_start_count,
               SUM(CASE WHEN sentence_end THEN count ELSE 0 END)::DOUBLE
                   AS sentence_end_count
        FROM classified GROUP BY ALL
    ), modifier_mass AS (
        SELECT modifier_surface, modifier_pos, head_surface, head_pos, decade,
               modifier_ner AS label, SUM(count)::DOUBLE AS label_count
        FROM classified GROUP BY ALL
    ), modifier_ranked AS (
        SELECT *, ROW_NUMBER() OVER (
            PARTITION BY modifier_surface, modifier_pos, head_surface, head_pos, decade
            ORDER BY label_count DESC, label
        ) AS rank FROM modifier_mass
    ), head_mass AS (
        SELECT modifier_surface, modifier_pos, head_surface, head_pos, decade,
               head_ner AS label, SUM(count)::DOUBLE AS label_count
        FROM classified GROUP BY ALL
    ), head_ranked AS (
        SELECT *, ROW_NUMBER() OVER (
            PARTITION BY modifier_surface, modifier_pos, head_surface, head_pos, decade
            ORDER BY label_count DESC, label
        ) AS rank FROM head_mass
    ), pair_mass AS (
        SELECT modifier_surface, modifier_pos, head_surface, head_pos, decade,
               pair_ner AS label, SUM(count)::DOUBLE AS label_count
        FROM classified GROUP BY ALL
    ), pair_ranked AS (
        SELECT *, ROW_NUMBER() OVER (
            PARTITION BY modifier_surface, modifier_pos, head_surface, head_pos, decade
            ORDER BY label_count DESC, label
        ) AS rank FROM pair_mass
    ), pair_distribution AS (
        SELECT p.modifier_surface, p.modifier_pos, p.head_surface, p.head_pos,
               p.decade,
               json_group_object(p.label, p.label_count / t.classified_ner_count)
                   AS ner_distribution_json
        FROM pair_mass p
        JOIN totals t USING (
            modifier_surface, modifier_pos, head_surface, head_pos, decade
        )
        GROUP BY p.modifier_surface, p.modifier_pos, p.head_surface, p.head_pos,
                 p.decade
    )
    SELECT t.modifier_surface, t.modifier_pos, t.head_surface, t.head_pos,
           t.decade::BIGINT AS time,
           mr.label AS modifier_ner, hr.label AS head_ner, pr.label AS pair_ner,
           mr.label_count / NULLIF(t.classified_ner_count, 0)
               AS modifier_ner_confidence,
           hr.label_count / NULLIF(t.classified_ner_count, 0)
               AS head_ner_confidence,
           pr.label_count / NULLIF(t.classified_ner_count, 0)
               AS pair_ner_confidence,
           t.same_span_count / NULLIF(t.classified_ner_count, 0)
               AS same_entity_span_fraction,
           t.classified_ner_count / NULLIF(pt.pair_count, 0) AS ner_coverage,
           t.classified_ner_count::BIGINT AS classified_ner_count,
           t.sentence_start_count / NULLIF(t.classified_ner_count, 0)
               AS sentence_start_fraction,
           t.sentence_end_count / NULLIF(t.classified_ner_count, 0)
               AS sentence_end_fraction,
           pd.ner_distribution_json
    FROM totals t
    JOIN modifier_ranked mr USING (
        modifier_surface, modifier_pos, head_surface, head_pos, decade
    )
    JOIN head_ranked hr USING (
        modifier_surface, modifier_pos, head_surface, head_pos, decade
    )
    JOIN pair_ranked pr USING (
        modifier_surface, modifier_pos, head_surface, head_pos, decade
    )
    LEFT JOIN pair_totals pt USING (
        modifier_surface, modifier_pos, head_surface, head_pos, decade
    )
    JOIN pair_distribution pd USING (
        modifier_surface, modifier_pos, head_surface, head_pos, decade
    )
    WHERE mr.rank=1 AND hr.rank=1 AND pr.rank=1
    """
    connection.execute(
        f"COPY ({query}) TO '{sql_quote(temp_destination)}' (FORMAT PARQUET, COMPRESSION ZSTD)"
    )
    connection.close()
    os.replace(str(temp_destination), str(destination))
    print(f"Wrote {destination}")


def read_manifest_entry(path, task_id):
    with open(path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    try:
        return manifest["entries"][task_id]
    except (KeyError, IndexError):
        raise RuntimeError(f"task id {task_id} is outside the manifest")


def parse_orders(value):
    orders = [int(part) for part in value.split(",") if part]
    invalid = [order for order in orders if order not in (0, 1, 2, 3, 4, 5)]
    if invalid:
        raise argparse.ArgumentTypeError(f"invalid ngram orders: {invalid}")
    return orders


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command")
    subparsers.required = True

    manifest = subparsers.add_parser("manifest", help="fetch official shard indexes")
    manifest.add_argument("--corpus", default=DEFAULT_CORPUS)
    manifest.add_argument("--orders", type=parse_orders, default=parse_orders("0,1,2,3,4,5"))
    manifest.add_argument("--output", required=True)
    manifest.add_argument("--timeout", type=int, default=120)

    extract = subparsers.add_parser("extract", help="extract one remote or local shard")
    source_group = extract.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--source", help="local .gz path or shard URL")
    source_group.add_argument("--manifest", help="manifest JSON")
    extract.add_argument("--task-id", type=int, help="zero-based manifest entry")
    extract.add_argument("--order", type=int, choices=(0, 1, 2, 3, 4, 5))
    extract.add_argument("--output", required=True)
    extract.add_argument("--year-min", type=int, default=1800)
    extract.add_argument("--year-max", type=int, default=2019)
    extract.add_argument("--bin-size", type=int, default=10)
    extract.add_argument("--context-pos", default=",".join(sorted(DEFAULT_CONTEXT_POS)))
    extract.add_argument("--context-vocab")
    extract.add_argument("--ner-pair-offset", type=int, default=2)
    extract.add_argument("--batch-size", type=int, default=50000)
    extract.add_argument("--progress-every", type=int, default=250000)
    extract.add_argument("--timeout", type=int, default=120)
    extract.add_argument(
        "--retries", type=int, default=4,
        help="number of retries for truncated or interrupted remote gzip streams",
    )
    extract.add_argument(
        "--retry-delay", type=float, default=5.0,
        help="base retry delay in seconds; attempts use linear backoff",
    )
    extract.add_argument("--force", action="store_true")
    extract.add_argument(
        "--replace-incompatible", action="store_true",
        help="rebuild a task only when its completion settings or output specification differ",
    )

    reduce_parser = subparsers.add_parser("reduce", help="reduce Parquet fragments")
    reduce_parser.add_argument("--output", required=True)
    reduce_parser.add_argument(
        "--threads", type=int, default=1,
        help="total CPU thread budget; divided among aggregation workers",
    )
    reduce_parser.add_argument(
        "--memory-limit",
        help="total DuckDB memory budget such as 192GB; required for parallel workers",
    )
    reduce_parser.add_argument(
        "--temp-directory",
        help="spill directory; defaults to OUTPUT/duckdb_tmp",
    )
    reduce_parser.add_argument(
        "--manifest", action="append", default=[],
        help="manifest whose completion must be verified; may be repeated",
    )
    reduce_parser.add_argument("--allow-incomplete", action="store_true")
    reduce_parser.add_argument(
        "--force", action="store_true",
        help="rebuild even when a matching completed reduction already exists",
    )
    reduce_parser.add_argument(
        "--table", action="append", choices=tuple(REDUCE_GROUPS),
        help="table to reduce; may be repeated (default: every available table)",
    )
    add_partition_arguments(reduce_parser)

    lemmatize = subparsers.add_parser(
        "lemmatize", help="build a word/POS-to-lemma lexicon"
    )
    lemmatize.add_argument("--output", required=True, help="pipeline output directory")
    lemmatize.add_argument("--lexicon", help="output Parquet; defaults under reduced/")
    lemmatize.add_argument("--model", default="en_core_web_trf")
    lemmatize.add_argument("--batch-size", type=int, default=50000)

    vocab = subparsers.add_parser("vocab", help="make a fixed context vocabulary")
    vocab.add_argument("--output", required=True, help="pipeline output directory")
    vocab.add_argument("--vocab-output", required=True)
    vocab.add_argument("--lexicon", help="lemma lexicon; defaults under reduced/")
    vocab.add_argument("--top-k", type=int, default=50000)
    vocab.add_argument("--context-pos", default=",".join(sorted(DEFAULT_CONTEXT_POS)))
    vocab.add_argument("--threads", type=int, default=1)
    vocab.add_argument("--memory-limit", help="optional DuckDB limit such as 48GB")
    vocab.add_argument("--temp-directory", help="DuckDB spill directory")
    vocab.add_argument(
        "--coverage-output",
        help="coverage TSV; defaults beside --vocab-output with _coverage.tsv suffix",
    )

    feature_inputs = subparsers.add_parser(
        "feature-inputs", help="adapt reduced Google tables to the shared feature specification"
    )
    feature_inputs.add_argument("--output", required=True, help="pipeline output directory")
    feature_inputs.add_argument(
        "--feature-output",
        help="destination; defaults to OUTPUT/extracted/feature_inputs/local5",
    )
    feature_inputs.add_argument(
        "--tables-output",
        help="audit-table destination; defaults to OUTPUT/extracted/tables",
    )
    feature_inputs.add_argument("--lexicon", help="lemma lexicon; defaults under reduced/")
    feature_inputs.add_argument(
        "--lemma-model",
        help="rule model for missing lemma joins; defaults to the lexicon sidecar's model; "
             "its rules/version must match the recorded lexicon (never loads the full pipeline)",
    )
    feature_inputs.add_argument("--threads", type=int, default=1,
                               help="total CPU thread budget; divided among aggregation workers")
    feature_inputs.add_argument("--memory-limit",
                               help="total DuckDB memory budget; required for parallel workers")
    feature_inputs.add_argument("--temp-directory", help="DuckDB spill directory")
    add_partition_arguments(feature_inputs)
    feature_inputs.add_argument(
        "--include-dependencies", action="store_true",
        help="also adapt the optional dependency table (not used by shared base features)",
    )
    feature_inputs.add_argument(
        "--allow-partial", action="store_true",
        help="allow an explicitly partial reduction (not recommended for final analysis)",
    )
    feature_inputs.add_argument(
        "--annotations-only", action="store_true",
        help="reuse existing four feature tables and refresh only bigram annotations",
    )

    ner = subparsers.add_parser("ner-predict", help="NER-tag unique cased snippets")
    ner.add_argument("--output", required=True)
    ner.add_argument("--predictions")
    ner.add_argument("--model", default="en_core_web_lg")
    ner.add_argument(
        "--targets", action="append", default=[],
        help="optional TSV/CSV with modifier and head columns; may be repeated",
    )
    ner.add_argument("--min-snippet-count", type=int, default=0)
    ner.add_argument("--batch-size", type=int, default=50000)
    ner.add_argument("--ner-batch-size", type=int, default=256)
    ner.add_argument("--ner-processes", type=int, default=1)
    ner.add_argument("--threads", type=int, default=1)
    ner.add_argument("--memory-limit", help="optional DuckDB limit such as 48GB")
    ner.add_argument("--temp-directory", help="DuckDB spill directory")

    aggregate = subparsers.add_parser("ner-aggregate", help="aggregate NER by pair/decade")
    aggregate.add_argument("--output", required=True)
    aggregate.add_argument("--predictions")
    aggregate.add_argument("--threads", type=int, default=1)
    aggregate.add_argument("--memory-limit", help="optional DuckDB limit such as 48GB")
    aggregate.add_argument("--temp-directory", help="DuckDB spill directory")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.command == "manifest":
        result = download_manifest(args.corpus, args.orders, args.timeout)
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with open(destination, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, sort_keys=True)
            handle.write("\n")
        print(f"Wrote {len(result['entries']):,} tasks to {destination}")

    elif args.command == "extract":
        if args.manifest:
            if args.task_id is None:
                raise RuntimeError("--task-id is required with --manifest")
            entry = read_manifest_entry(args.manifest, args.task_id)
            source, order, shard = entry["source"], int(entry["order"]), entry["shard"]

        else:
            if args.order is None:
                raise RuntimeError("--order is required with --source")
            source, order = args.source, args.order
            shard = os.path.basename(source).replace(".gz", "")
        extract_source_with_retries(args, source, order, shard)

    elif args.command == "reduce":
        reduce_tables(args)

    elif args.command == "lemmatize":
        lemmatize_vocabulary(args)

    elif args.command == "vocab":
        make_vocabulary(args)

    elif args.command == "feature-inputs":
        make_feature_inputs(args)

    elif args.command == "ner-predict":
        ner_predict(args)

    elif args.command == "ner-aggregate":
        ner_aggregate(args)


if __name__ == "__main__":
    main()
