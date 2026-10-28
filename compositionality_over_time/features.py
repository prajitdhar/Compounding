"""Bounded Google and COHA feature preparation and sparse experiments.

The four scientific feature_inputs are read-only. Preparation retains full
population marginals, not just benchmark counts, while storing context vectors
only for the evaluation targets. Exact pair/word type keys are hash-routed
WITHOUT time so wider temporal bins can deduplicate types inside bounded
buckets. Feature jobs never load the original corpus matrices into pandas.
Tagged compounds use ADJ/NOUN-to-NOUN pairs. UnTagged compounds use all
lexical bigram POS combinations before target POS pooling; context tags stay.
"""

import argparse
import contextlib
import fcntl
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy import sparse

DIRECTORY = Path(__file__).resolve().parent
SOURCE_DIRECTORY = DIRECTORY / "feature_helpers"
if str(SOURCE_DIRECTORY) not in sys.path:
    sys.path.append(str(SOURCE_DIRECTORY))

from coha_feature_pipeline import read_ratings, write_features
from coha_features import CONDITIONS, PAIR
from feature_math import (STATIC_COLUMNS, weight_matrix, static_features,
                                 setting_features, temporal_cosines)
from src.partitioned_parquet import (add_partition_arguments, partition_options,
    partitioned_sum, connection_scope, _plan_batches, _arrow_reader,
    _BufferedPartitionWriter, _route_batch)
from pipeline_source_info import (artifact, source_info, software_record,
                              validate_feature_input_manifest, write_json)

CONTEXT_SIZES = (1000, 3000, 5000, 10000, 30000, 50000)
TEMPORAL_BINS = (10000, 10, 20, 50, 100)
CUTOFFS = (0, 10, 50, 100, 500, 1000)
VERSION = 3
BETA_POLICY = "clipped_unit_interval"
TABLE_COLUMNS = {
    "bigrams": ("modifier", "head", "time", "context", "count"),
    "unigrams": ("word", "time", "context", "count"),
    "bigram_counts": ("modifier", "head", "time", "count"),
    "unigram_counts": ("word", "time", "count"),
}
VECTOR_KEYS = ["kind", "target", "context_rank", "time"]
COUNT_KEYS = ["kind", "target", "partner", "time"]
FREQUENCY_COLUMNS = ("comp_freq", "mod_prod", "head_prod", "mod_freq", "head_freq",
    "mod_family_size", "mod_family_size_new", "head_family_size", "head_family_size_new",
    "comp_tf", "log_comp_freq", "mod_tf", "log_mod_freq", "head_tf", "log_head_freq",
    "log_mod_freq_new", "log_head_freq_new", "assoc_pair_freq", "assoc_left_freq",
    "assoc_right_freq", "assoc_population_size", "ppmi", "local_mi", "log_ratio")


def _feature_columns(condition):
    columns = [*PAIR, *STATIC_COLUMNS, *FREQUENCY_COLUMNS]
    if condition == "CompoundAware":
        columns += ["sim_bw_settings_modifier", "perc_token_mod", "perc_type_mod",
                    "sim_bw_settings_head", "perc_token_head", "perc_type_head"]
    elif condition == "CompoundPooled":
        for suffix in ("mod", "head"):
            columns += [f"sim_bw_settings_constituent_{suffix}",
                        f"perc_token_constituent_{suffix}", f"perc_type_constituent_{suffix}"]
    return columns


def _compound_population(args):
    return "all_lexical_bigrams" if args.untagged else "adj_noun_and_noun_noun"


def _quote(value):
    return str(value).replace("'", "''")


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _snapshot(path):
    path = Path(path).resolve()
    stat = path.stat()
    return {"path": str(path), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _read_marker(path):
    try:
        value = json.loads(Path(path).read_text())
        if any(_snapshot(record["path"]) != record for record in value.get("files", [])):
            return None
        return value
    except (OSError, ValueError, KeyError, TypeError):
        return None


@contextlib.contextmanager
def _lock(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f"Another process is using {path}") from None
        yield


def _write_parquet(frame, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_parquet(temporary, index=False, compression="zstd")
    os.replace(temporary, path)


def read_context_vocabulary(path):
    """One row per ranked lemma_POS, independent of surface spelling."""
    frame = pd.read_csv(path, sep="\t", keep_default_na=False)
    if "context" not in frame:
        if not {"lemma", "pos"}.issubset(frame):
            raise ValueError("Vocabulary requires context/rank or lemma/pos/rank columns")
        frame["context"] = frame["lemma"].astype(str) + "_" + frame["pos"].astype(str)
    if "rank" not in frame:
        raise ValueError("The extraction vocabulary must contain frozen ranks")
    frame["rank"] = pd.to_numeric(frame["rank"], errors="raise").astype("int64")
    if (frame["rank"] < 1).any():
        raise ValueError("Vocabulary ranks must be positive")
    metadata = [name for name in ["rank", "context", "score", "decades_observed",
                                  "lemma_match_count"] if name in frame]
    frame = frame[metadata].drop_duplicates().sort_values("rank").reset_index(drop=True)
    if frame.context.duplicated().any() or frame["rank"].duplicated().any():
        raise ValueError("Conflicting ranks or metadata for a lemma_POS vocabulary dimension")
    if frame["rank"].tolist() != list(range(1, len(frame) + 1)):
        raise ValueError("Vocabulary ranks must form a contiguous unique prefix")
    if not frame.context.str.contains(r"_[A-Z]+$", regex=True).all():
        raise ValueError("Context dimensions must retain POS suffixes")
    return frame


def corpus_context(manifest):
    """Identify scientific input semantics independently of annotation metadata."""
    corpus = manifest.get("source_info", {}).get("corpus") or {
        "name": "Google Books Ngrams", "release": "20200217", "subset": "eng"}
    name = str(corpus.get("name", "")).strip().casefold()
    is_coha = name in ("coha", "corpus of historical american english")
    mode = manifest.get("context_mode", manifest.get("mode", "local5"))
    if manifest.get("mode", mode) != mode:
        raise ValueError("Conflicting mode and context_mode in feature-input manifest")
    if not is_coha and mode != "local5":
        raise ValueError("Google features support local5 only, not sentence contexts")
    if mode not in ("local5", "sentence"):
        raise ValueError(f"Unsupported COHA context mode: {mode!r}")
    return {"corpus": corpus, "corpus_id": "coha" if is_coha else "google", "context_mode": mode}


def _inputs(args):
    root = Path(args.inputdir).resolve()
    manifest_path = root / "corpus_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    validate_feature_input_manifest(manifest)
    corpus_context(manifest)
    files, records = {}, {}
    for name, columns in TABLE_COLUMNS.items():
        path = (root / manifest.get("table_files", {}).get(name, f"{name}.parquet")).resolve()
        record = manifest.get("tables", {}).get(name, {})
        if not record:
            record = next((item for item in manifest.get("source_info", {}).get("outputs", [])
                           if Path(item.get("name", "")).name == path.name), {})
        snapshot = _snapshot(path)
        if record.get("bytes") is not None and record["bytes"] != snapshot["bytes"]:
            raise ValueError(f"Input size differs from published manifest: {name}")
        schema = pq.read_schema(path)
        if set(schema.names) != set(columns):
            raise ValueError(f"Unexpected {name} schema: {schema.names}")
        if not pa.types.is_integer(schema.field("time").type) or not pa.types.is_integer(schema.field("count").type):
            raise ValueError("Input time/count columns must contain integers")
        files[name] = path
        records[name] = {**snapshot, "sha256": record.get("sha256"),
                         "sha256_source": "published feature-input manifest"}
    return manifest, files, records


def _specification(args):
    manifest, files, records = _inputs(args)
    vocabulary = read_context_vocabulary(args.context_vocabulary)
    if max(args.context_sizes) > len(vocabulary):
        raise ValueError("Requested vocabulary exceeds the frozen extraction ceiling")
    vocab_record = artifact(args.context_vocabulary, "context_vocabulary")
    extraction = manifest.get("source_info", {}).get("parameters", {})
    expected_hash = extraction.get("context_vocabulary_sha256")
    if expected_hash and expected_hash != vocab_record["sha256"]:
        raise ValueError("Vocabulary hash differs from the vocabulary used for extraction")
    rating_args = SimpleNamespace(tag=not args.untagged,
        **{name: getattr(args, name) for name in ("reddy90", "cordeiro90", "cordeiro100")})
    ratings = read_ratings(rating_args)
    pairs = ratings[["modifier", "head"]].drop_duplicates().sort_values(
        ["modifier", "head"]).reset_index(drop=True)
    pairs.insert(0, "target", np.arange(len(pairs), dtype=np.int64))
    words = pd.DataFrame({"word": sorted(set(pairs.modifier) | set(pairs["head"]))})
    words.insert(0, "target", np.arange(len(words), dtype=np.int64))
    software = software_record([__file__, DIRECTORY / "feature_math.py",
        SOURCE_DIRECTORY / "coha_features.py",
        SOURCE_DIRECTORY / "coha_feature_pipeline.py", SOURCE_DIRECTORY / "table_io.py",
        DIRECTORY / "pipeline_source_info.py"],
        ["duckdb", "pyarrow", "pandas", "numpy", "scipy"])
    options = partition_options(args)
    # Annotation enrichment edits the surrounding manifest, not these tables.
    # Only the scientific core identities enter preparation checkpoints.
    identity = {"version": VERSION, "inputs": records, **corpus_context(manifest),
        "tagged_targets": not args.untagged,
        "compound_population": _compound_population(args),
        "context_vocabulary": vocab_record,
        "ratings": [artifact(getattr(args, name), name)
                    for name in ("reddy90", "cordeiro90", "cordeiro100")],
        "temporal_bins": sorted(set(args.temporal_bins) | {10}), "software": software,
        "layout": {name: options[name] for name in ("partitions", "batch_rows", "arrow_batch_rows",
            "shuffle_buffer_mb", "shuffle_row_group_rows", "shuffle_file_rows", "shuffle_max_open_files")}}
    fingerprint = _digest(identity)
    root = Path(args.cache_directory).resolve() / f"prepare-{fingerprint[:24]}"
    return {"manifest": manifest, "files": files, "identity": identity,
        "fingerprint": fingerprint, "root": root, "vocabulary": vocabulary,
        "ratings": ratings, "pairs": pairs, "words": words, "options": options}


def _setup(specification):
    def install(connection):
        connection.register("evaluation_pairs", pa.Table.from_pandas(specification["pairs"], preserve_index=False))
        connection.register("evaluation_words", pa.Table.from_pandas(specification["words"], preserve_index=False))
        connection.register("vocabulary", pa.Table.from_pandas(
            specification["vocabulary"][["context", "rank"]], preserve_index=False))
    return install


def _target_expression(column, args):
    return column if not args.untagged else f"regexp_replace({column}, '_[A-Z]+$', '')"


def _source_projection(name, args):
    columns = TABLE_COLUMNS[name]
    targets = [column for column in ("word", "modifier", "head") if column in columns]
    expressions = [f"{_target_expression(column, args)} AS {column}" for column in targets]
    expressions += ["CASE WHEN time IS NULL THEN error('Missing input time') ELSE time END AS time",
        "CASE WHEN count IS NULL OR count < 0 THEN error('Invalid raw count') ELSE count END AS count"]
    if "context" in columns:
        expressions.append("context")
    if "modifier" in columns:
        # This population drives compound vectors, independent counts, roles,
        # PPMI backgrounds and exact productivity summaries consistently.
        # Input tables already contain all fully tagged lexical adjacencies.
        if args.untagged:
            expressions.append("TRUE AS eligible")
        else:
            expressions.append("((ends_with(modifier, '_ADJ') OR ends_with(modifier, '_NOUN')) "
                               "AND ends_with(head, '_NOUN')) AS eligible")
    if name == "unigram_counts":
        expressions.append("regexp_matches(word, '_(NOUN|PROPN|ADJ|VERB|NUM|ADV)$') AS context_pos")
        expressions.append("word AS tagged_word")
    return "SELECT " + ", ".join(expressions) + " FROM partition_input"


def _vector_query(name, args):
    source = _source_projection(name, args)
    if name == "bigrams":
        joined = f"""SELECT b.*, v.rank AS context_rank, p.target AS pair_id,
                m.target AS mod_id, h.target AS head_id
            FROM ({source}) b JOIN vocabulary v ON b.context=v.context
            LEFT JOIN evaluation_pairs p ON b.modifier=p.modifier AND b.head=p.head
            LEFT JOIN evaluation_words m ON b.modifier=m.word
            LEFT JOIN evaluation_words h ON b.head=h.word
            WHERE b.eligible AND b.count > 0"""
        dimensions = "(VALUES (0,-1::BIGINT),(2,pair_id),(3,mod_id),(4,head_id))"
    else:
        joined = f"""SELECT b.*, v.rank AS context_rank, w.target AS word_id
            FROM ({source}) b JOIN vocabulary v ON b.context=v.context
            LEFT JOIN evaluation_words w ON b.word=w.word WHERE b.count > 0"""
        dimensions = "(VALUES (1,-1::BIGINT),(5,word_id))"
    # A GROUP BY is bounded by one checkpoint input batch, not the corpus.
    return f"""SELECT kind::INTEGER AS kind, target::BIGINT AS target, context_rank,
            time, SUM(count)::BIGINT AS count
        FROM ({joined}) joined CROSS JOIN LATERAL {dimensions} dims(kind,target)
        WHERE target IS NOT NULL GROUP BY kind,target,context_rank,time"""


def _count_query(name, args, partitions):
    source = _source_projection(name, args)
    if name == "bigram_counts":
        joined = f"""SELECT b.*, p.target AS pair_id, m.target AS mod_id, h.target AS head_id
            FROM ({source}) b
            LEFT JOIN evaluation_pairs p ON b.modifier=p.modifier AND b.head=p.head
            LEFT JOIN evaluation_words m ON b.modifier=m.word
            LEFT JOIN evaluation_words h ON b.head=h.word WHERE b.count > 0"""
        dimensions = """(VALUES
            (0, '', ''), (1, CASE WHEN eligible THEN '' END, ''),
            (3, CASE WHEN eligible THEN modifier END, ''),
            (4, CASE WHEN eligible THEN head END, ''),
            (7, CASE WHEN mod_id IS NOT NULL THEN modifier END, ''),
            (8, CASE WHEN head_id IS NOT NULL THEN head END, ''),
            (9, CAST(pair_id AS VARCHAR), ''),
            (12, CASE WHEN eligible THEN modifier END, head))"""
    else:
        joined = f"""SELECT b.*, w.target AS word_id, v.rank AS context_rank
            FROM ({source}) b LEFT JOIN evaluation_words w ON b.word=w.word
            LEFT JOIN vocabulary v ON b.tagged_word=v.context WHERE b.count > 0"""
        dimensions = """(VALUES (6, '', ''),
            (5, CASE WHEN word_id IS NOT NULL THEN word END, ''),
            (10, CASE WHEN context_pos THEN '' END, ''),
            (11, CAST(context_rank AS VARCHAR), ''))"""
    grouped = f"""SELECT kind::INTEGER AS kind, target::VARCHAR AS target,
            partner::VARCHAR AS partner, time, SUM(count)::BIGINT AS count
        FROM ({joined}) joined CROSS JOIN LATERAL {dimensions} dims(kind,target,partner)
        WHERE target IS NOT NULL GROUP BY kind,target,partner,time"""
    return f"SELECT *, hash(kind,target,partner) % {int(partitions)} AS __bucket FROM ({grouped}) p"


def _route_counts(specification, name, args):
    """Resume bounded scans, with all decades of a type routed to ONE bucket."""
    root = specification["root"] / "counts" / name
    root.mkdir(parents=True, exist_ok=True)
    source = specification["files"][name]
    options = specification["options"]
    schema, batches = _plan_batches([source], list(TABLE_COLUMNS[name]), options["batch_rows"])
    query = _count_query(name, args, options["partitions"])
    routing = {"type": "hash", "buckets": options["partitions"]}
    paths = {}
    with connection_scope(options, args.temp_directory) as connection:
        _setup(specification)(connection)
        connection.register("partition_input", pa.Table.from_batches([], schema=schema))
        output_schema = connection.execute(query + " LIMIT 0").to_arrow_table().schema
        connection.unregister("partition_input")
        for index, units in enumerate(batches):
            directory = root / f"batch-{index:06d}"
            marker = directory / "complete.json"
            complete = _read_marker(marker)
            if complete is None:
                temporary = root / f"batch-{index:06d}.tmp"
                if temporary.exists():
                    shutil.rmtree(temporary)
                temporary.mkdir()
                observed = {"input_rows": 0, "input_count_sum": 0}
                reader = _arrow_reader(schema, units, options["arrow_batch_rows"], observed)
                writer = _BufferedPartitionWriter(temporary, output_schema.remove(
                    output_schema.get_field_index("__bucket")), routing, options)
                connection.register("partition_input", reader)
                projected = None
                try:
                    projected = connection.execute(query).to_arrow_reader(options["arrow_batch_rows"])
                    for batch in projected:
                        _route_batch(writer, batch, routing)
                    fragments = writer.finish()
                finally:
                    if projected is not None:
                        projected.close()
                    writer.close()
                    connection.unregister("partition_input")
                    reader.close()
                # Resolve marker paths AFTER atomic publication, not under .tmp.
                relative = [path.relative_to(temporary) for path in fragments]
                if directory.exists():
                    shutil.rmtree(directory)
                os.replace(temporary, directory)
                complete = {"files": [_snapshot(directory / path) for path in relative],
                            "input_rows": observed["input_rows"]}
                write_json(marker, complete)
                print(f"Prepared {name}: batch {index+1}/{len(batches)} "
                      f"({observed['input_rows']:,} input rows)", flush=True)
            for record in complete["files"]:
                path = Path(record["path"])
                bucket = int(path.parent.name.split("=", 1)[1])
                paths.setdefault(bucket, []).append(path)
    return paths


def _frequency_query(paths, width):
    files = "[" + ",".join("'" + _quote(path) + "'" for path in paths) + "]"
    return f"""WITH binned AS MATERIALIZED (
        SELECT kind,target,partner,floor(time::DOUBLE / {int(width)})::BIGINT * {int(width)} AS time,
            SUM(count)::BIGINT AS count FROM read_parquet({files})
        GROUP BY kind,target,partner,floor(time::DOUBLE / {int(width)})::BIGINT * {int(width)}
    )
    SELECT kind,target,partner,time,count FROM binned
        WHERE kind NOT IN (3,4,12) OR
            (kind IN (3,4) AND target IN (SELECT word FROM evaluation_words))
    UNION ALL
    SELECT CASE kind WHEN 3 THEN 103 WHEN 4 THEN 104 ELSE 112 END,
        '', '', time, COUNT(*)::BIGINT FROM binned WHERE kind IN (3,4,12)
        GROUP BY kind,time
    UNION ALL
    SELECT 113,target,'',time,COUNT(*)::BIGINT FROM binned
        WHERE kind=12 AND target IN (SELECT word FROM evaluation_words) GROUP BY target,time
    UNION ALL
    SELECT 114,partner,'',time,COUNT(*)::BIGINT FROM binned
        WHERE kind=12 AND partner IN (SELECT word FROM evaluation_words) GROUP BY partner,time
    UNION ALL
    SELECT 2,CAST(p.target AS VARCHAR),'',b.time,b.count FROM binned b
        JOIN evaluation_pairs p ON b.target=p.modifier AND b.partner=p.head WHERE b.kind=12"""


def _prepare_frequencies(specification, routed, width, args):
    root = specification["root"] / "frequencies" / f"width-{width}"
    root.mkdir(parents=True, exist_ok=True)
    marker = root / "complete.json"
    saved = _read_marker(marker)
    destination = root / "frequencies.parquet"
    if saved:
        return destination
    parts = []
    for bucket, paths in sorted(routed.items()):
        part = root / f"bucket-{bucket:05d}.parquet"
        part_marker = part.with_suffix(".json")
        if not _read_marker(part_marker):
            with connection_scope(specification["options"], args.temp_directory) as connection:
                _setup(specification)(connection)
                temporary = part.with_name(part.name + ".tmp")
                query = _frequency_query(paths, width)
                connection.execute(f"COPY ({query}) TO '{_quote(temporary)}' "
                                   "(FORMAT PARQUET, COMPRESSION ZSTD)")
                os.replace(temporary, part)
            write_json(part_marker, {"files": [_snapshot(part)]})
            print(f"Frequency width {width}: bucket {bucket+1}/{specification['options']['partitions']}", flush=True)
        parts.append(part)
    with connection_scope(specification["options"], args.temp_directory) as connection:
        if parts:
            quoted = "[" + ",".join("'" + _quote(path) + "'" for path in parts) + "]"
            query = f"SELECT kind,target,partner,time,SUM(count)::BIGINT AS count FROM read_parquet({quoted}) GROUP BY kind,target,partner,time"
            temporary = destination.with_name(destination.name + ".tmp")
            connection.execute(f"COPY ({query}) TO '{_quote(temporary)}' (FORMAT PARQUET, COMPRESSION ZSTD)")
            os.replace(temporary, destination)
        else:
            schema = pa.schema([("kind", pa.int32()), ("target", pa.string()),
                ("partner", pa.string()), ("time", pa.int64()), ("count", pa.int64())])
            pq.write_table(schema.empty_table(), destination, compression="zstd")
    write_json(marker, {"files": [_snapshot(destination)]})
    return destination


def prepare(args):
    specification = _specification(args)
    root = specification["root"]
    root.mkdir(parents=True, exist_ok=True)
    with _lock(root / "prepare.lock"):
        saved = _read_marker(root / "complete.json")
        if saved and saved.get("fingerprint") == specification["fingerprint"]:
            print(f"Reused prepared features: {root}", flush=True)
            return root
        write_json(root / "identity.json", specification["identity"])
        _write_parquet(specification["vocabulary"], root / "contexts.parquet")
        write_json(root / "targets.json", {"pairs": specification["pairs"].to_dict("records"),
                                          "words": specification["words"].to_dict("records")})
        products = [root / "contexts.parquet", root / "targets.json"]
        for name in ("bigrams", "unigrams"):
            destination = root / f"{name}_vectors.parquet"
            partitioned_sum([specification["files"][name]], destination, VECTOR_KEYS,
                _vector_query(name, args), list(TABLE_COLUMNS[name]),
                options=specification["options"], work_root=root / "vector-checkpoints",
                spill_root=args.temp_directory, identity=specification["fingerprint"],
                projection_setup=_setup(specification), preserve_count_mass=False,
                time_source_column="time")
            products.append(destination)
        routed = {}
        for name in ("bigram_counts", "unigram_counts"):
            for bucket, paths in _route_counts(specification, name, args).items():
                routed.setdefault(bucket, []).extend(paths)
        for width in specification["identity"]["temporal_bins"]:
            products.append(_prepare_frequencies(specification, routed, width, args))
        # Input mutation during a long scan must never publish a completion marker.
        for record in specification["identity"]["inputs"].values():
            if _snapshot(record["path"]) != {key: record[key] for key in ("path", "bytes", "mtime_ns")}:
                raise RuntimeError("A core input changed during feature preparation")
        parent = specification["manifest"].get("source_info", {})
        preparation_source_info = source_info("feature_prepare", specification["identity"]["corpus"],
            [{"name": name, **record} for name, record in specification["identity"]["inputs"].items()]
            + [specification["identity"]["context_vocabulary"], *specification["identity"]["ratings"]],
            {"tagged_targets": not args.untagged, "temporal_bins": specification["identity"]["temporal_bins"],
             "feature_definition_version": VERSION,
             "compound_population": _compound_population(args),
             "context_mode": specification["identity"]["context_mode"],
             "context_vocabulary": specification["identity"]["context_vocabulary"],
             "evaluation_filter": "only target vectors are restricted; context and occurrence backgrounds use the complete relevant corpus populations",
             "type_routing": "exact full pair/word keys, hashed without time before temporal deduplication"},
            {"column": "time", "unit": "calendar_year", "binning": "floor(time / temporal) * temporal"},
            specification["identity"]["software"], model=parent.get("model"),
            outputs=[artifact(path) for path in products],
            statistics={"evaluation_pairs": len(specification["pairs"]), "evaluation_words": len(specification["words"]),
                        "context_dimensions": len(specification["vocabulary"])}, execution=specification["options"])
        write_json(root / "complete.json", {"fingerprint": specification["fingerprint"],
            "files": [_snapshot(path) for path in products],
            "evaluation_pairs": len(specification["pairs"]), "evaluation_words": len(specification["words"]),
            "context_dimensions": len(specification["vocabulary"]),
            "execution": specification["options"], "source_info": preparation_source_info})
        print(f"Feature preparation complete: {root}", flush=True)
    return root


def _frequency_maps(frame, time):
    return {(int(row.kind), str(row.target)): int(row.count)
            for row in frame.loc[frame.time == time].itertuples(index=False)}


def _word_counts(values, words, kind, missing=0):
    return np.array([values.get((kind, word), missing) for word in words], dtype=float)


def _association(a, x, y, n):
    a, x, y = [np.asarray(values, dtype=float) for values in (a, x, y)]
    observed = np.column_stack([a, x-a, y-a, n-x-y+a])
    if (observed < 0).any():
        raise ValueError("Inconsistent raw association population")
    expected = np.column_stack([x*y, x*(n-y), (n-x)*y, (n-x)*(n-y)]) / n if n else np.zeros_like(observed)
    with np.errstate(divide="ignore", invalid="ignore"):
        pmi = np.log2(a) + np.log2(n) - np.log2(x) - np.log2(y)
        terms = np.where(observed > 0, observed * np.log(observed/expected), 0)
        local_mi = a*pmi
    return {"assoc_pair_freq": a, "assoc_left_freq": x, "assoc_right_freq": y,
            "assoc_population_size": np.full(len(a), n), "ppmi": np.maximum(0, pmi),
            "local_mi": local_mi, "log_ratio": 2*terms.sum(axis=1)}


def _controls(specification, frequencies, time):
    pairs, words = specification["pairs"], specification["words"].word.tolist()
    word_ids = {word: index for index, word in enumerate(words)}
    mod_ids = np.array([word_ids[word] for word in pairs.modifier], dtype=np.int64)
    head_ids = np.array([word_ids[word] for word in pairs["head"]], dtype=np.int64)
    values = _frequency_maps(frequencies, time)
    roles = (_word_counts(values, words, 3), _word_counts(values, words, 4))
    pooled = roles[0] + roles[1]
    general = _word_counts(values, words, 5, np.nan)
    occurrences = {"CompoundAware": roles, "CompoundPooled": (pooled, pooled),
                   "CompoundAgnostic": (general, general)}
    comp = np.array([values.get((2, str(target)), 0) for target in pairs.target], dtype=float)
    sufficient = [comp]
    for modifiers, heads in occurrences.values():
        sufficient.extend([np.nan_to_num(modifiers[mod_ids], nan=0),
                           np.nan_to_num(heads[head_ids], nan=0)])
    minimum = np.minimum.reduce(sufficient)
    controls = {}
    prod_mod = _word_counts(values, words, 113)[mod_ids]
    prod_head = _word_counts(values, words, 114)[head_ids]
    eligible_mass, all_mass, unigram_mass = [values.get((kind, ""), 0) for kind in (1,0,6)]
    for condition, (modifiers, heads) in occurrences.items():
        result = pairs[["modifier", "head"]].copy()
        result["time"] = time
        result["comp_freq"], result["mod_prod"], result["head_prod"] = comp, prod_mod, prod_head
        result["mod_freq"], result["head_freq"] = modifiers[mod_ids], heads[head_ids]
        with np.errstate(divide="ignore", invalid="ignore"):
            for label, production, other in (("mod", prod_mod, 104), ("head", prod_head, 103)):
                result[f"{label}_family_size"] = np.log2(production / values.get((112,""), 0))
                result[f"{label}_family_size_new"] = np.log2(production / values.get((other,""), 0))
            mass = (unigram_mass if condition == "CompoundAgnostic" else
                    2*eligible_mass if condition == "CompoundPooled" else eligible_mass)
            for prefix, total in (("comp",eligible_mass), ("mod",mass), ("head",mass)):
                result[f"{prefix}_tf"] = np.log2(1 + result[f"{prefix}_freq"])
                result[f"log_{prefix}_freq"] = np.log2(result[f"{prefix}_freq"] / total)
        result["log_mod_freq_new"] = result.log_mod_freq
        result["log_head_freq_new"] = result.log_head_freq
        if condition == "CompoundAgnostic":
            assoc = _association(np.array([values.get((9,str(target)),0) for target in pairs.target]),
                _word_counts(values, words, 7)[mod_ids], _word_counts(values, words, 8)[head_ids], all_mass)
        else:
            assoc = _association(comp, roles[0][mod_ids], roles[1][head_ids], eligible_mass)
        for column, data in assoc.items():
            result[column] = data
        controls[condition] = result
    return controls, occurrences, minimum, mod_ids, head_ids


def _load_vectors(specification, time, temporal, context_size, args):
    paths = [specification["root"] / f"{name}_vectors.parquet" for name in ("bigrams", "unigrams")]
    with connection_scope(specification["options"], args.temp_directory) as connection:
        query = """SELECT kind,target,context_rank,SUM(count)::BIGINT AS count
            FROM read_parquet(?) WHERE context_rank <= ? AND time >= ? AND time < ?
            GROUP BY kind,target,context_rank"""
        table = connection.execute(query, [[str(path) for path in paths], context_size,
                                          time, time+temporal]).to_arrow_table()
    kind = table["kind"].to_numpy()
    target = table["target"].to_numpy()
    context = table["context_rank"].to_numpy().astype(np.int64) - 1
    count = table["count"].to_numpy().astype(float)
    backgrounds, matrices = {}, {}
    for code in (0,1):
        mask = kind == code
        backgrounds[code] = np.zeros(context_size, dtype=float)
        backgrounds[code][context[mask]] = count[mask]
    for code in (2,3,4,5):
        mask = kind == code
        rows = len(specification["pairs"]) if code == 2 else len(specification["words"])
        matrices[code] = sparse.csr_matrix((count[mask], (target[mask], context[mask])),
                                          shape=(rows, context_size))
    return matrices, backgrounds


def _coverage(specification, size):
    frame = pd.read_parquet(specification["root"] / "frequencies/width-10/frequencies.parquet",
                            filters=[("kind", "in", [10,11])])
    rows = []
    for time, part in frame.groupby("time", sort=True):
        total = part.loc[part.kind == 10, "count"].sum()
        selected = part.loc[(part.kind == 11) & (pd.to_numeric(part.target, errors="coerce") <= size)]
        retained = selected["count"].sum()
        rows.append({"time": int(time), "eligible_context_tokens": int(total),
                     "retained_context_tokens": int(retained), "retained_context_types": len(selected),
                     "token_coverage": retained/total if total else np.nan})
    return pd.DataFrame(rows, columns=["time", "eligible_context_tokens", "retained_context_tokens",
                                      "retained_context_types", "token_coverage"])


def _configuration(args):
    if args.grid_index is not None:
        size_index, time_index = divmod(args.grid_index, len(args.temporal_bins))
        if size_index >= len(args.context_sizes):
            raise ValueError("Grid index is outside the requested vocabulary/time grid")
        return args.context_sizes[size_index], args.temporal_bins[time_index]
    if args.context_size is None or args.temporal is None:
        raise ValueError("features requires --grid-index or both --context-size and --temporal")
    return args.context_size, args.temporal


def run_features(args):
    specification = _specification(args)
    saved = _read_marker(specification["root"] / "complete.json")
    if not saved or saved.get("fingerprint") != specification["fingerprint"]:
        raise RuntimeError("Matching feature preparation is incomplete; run prepare with the same arguments first")
    size, temporal = _configuration(args)
    if size > len(specification["vocabulary"]):
        raise ValueError("Requested vocabulary exceeds the frozen extraction ceiling")
    if temporal not in specification["identity"]["temporal_bins"]:
        raise ValueError("Temporal bin was not included in feature preparation")
    tag = ("UnTagged" if args.untagged else "Tagged") + f"_ctx{size}"
    output = Path(args.outputdir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    parameters = {"context_size": size, "temporal": temporal, "cutoffs": args.cutoffs,
                  "weights": args.weights, "families": args.families, "conditions": args.conditions}
    fingerprint = _digest({"prepare": specification["fingerprint"], "parameters": parameters})
    marker = output / ".checkpoints" / f"configuration-{fingerprint[:24]}.json"
    with _lock(output / ".checkpoints" / f"{tag}-{temporal}.lock"):
        complete = _read_marker(marker)
        if complete and complete.get("fingerprint") == fingerprint:
            print(f"Reused feature configuration: {tag}, temporal={temporal}", flush=True)
            return complete
        return _run_configuration(args, specification, size, temporal, tag, output, marker, fingerprint, parameters)


def _run_configuration(args, specification, size, temporal, tag, output, marker, fingerprint, parameters):
    pairs, words = specification["pairs"], specification["words"]
    frequencies = pd.read_parquet(specification["root"] / f"frequencies/width-{temporal}/frequencies.parquet",
                                 filters=[("kind", "not in", [10,11])])
    times = sorted(frequencies.loc[frequencies.kind == 1, "time"].unique())
    pieces = {(family,weight,condition,cutoff): [] for family in args.families
              for weight in args.weights for condition in args.conditions for cutoff in args.cutoffs}
    constituents = {weight: [] for weight in args.weights}
    previous, previous_minimum, previous_time = None, None, None
    for time in times:
        time = int(time)
        raw, background = _load_vectors(specification, time, temporal, size, args)
        controls, occurrence, minimum, mod_ids, head_ids = _controls(specification, frequencies, time)
        pooled = raw[3] + raw[4]
        weighted = {}
        for weight in args.weights:
            use_ppmi = weight == "PPMI"
            compound = weight_matrix(raw[2], background[0], use_ppmi)
            role_mod = weight_matrix(raw[3], background[0], use_ppmi)
            role_head = weight_matrix(raw[4], background[0], use_ppmi)
            role_pooled = weight_matrix(pooled, 2*background[0], use_ppmi)
            general = weight_matrix(raw[5], background[1], use_ppmi)
            matrices = {"CompoundAware": (role_mod,role_head),
                        "CompoundPooled": (role_pooled,role_pooled),
                        "CompoundAgnostic": (general,general)}
            weighted[weight] = (compound, matrices)
            for condition in args.conditions:
                modifiers, heads = matrices[condition]
                current = pairs[["modifier", "head"]].copy()
                current["time"] = time
                comparisons = []
                if "features" in args.families and condition != "CompoundAgnostic":
                    raw_roles = (raw[3],raw[4]) if condition == "CompoundAware" else (pooled,pooled)
                    for position, label in enumerate(("modifier","head") if condition == "CompoundAware" else ("constituent","constituent")):
                        comparisons.append(setting_features(matrices[condition][position], general,
                            raw_roles[position],raw[5],occurrence[condition][position],occurrence["CompoundAgnostic"][position],label))
                    if condition == "CompoundPooled":
                        table = pd.concat([words[["word"]], comparisons[0]], axis=1)
                        table["time"] = time
                        constituents[weight].append(table)
                static = None
                if "features" in args.families:
                    static = pd.concat([current, static_features(compound,modifiers[mod_ids],heads[head_ids])], axis=1)
                    static = static.merge(controls[condition], on=PAIR, how="left", validate="one_to_one")
                    for position, comparison in enumerate(comparisons):
                        ids = mod_ids if position == 0 else head_ids
                        values = comparison.iloc[ids].reset_index(drop=True)
                        if condition == "CompoundPooled":
                            values = values.add_suffix("_mod" if position == 0 else "_head")
                        static = pd.concat([static,values],axis=1)
                change = None
                if "temporal" in args.families:
                    change = current.copy()
                    if previous is not None and previous_time == time-temporal:
                        prev_comp, prev_matrices = previous[weight]
                        prev_mod, prev_head = prev_matrices[condition]
                        change["change_comp"] = temporal_cosines(compound,prev_comp)
                        change["change_mod"] = temporal_cosines(modifiers,prev_mod)[mod_ids]
                        change["change_head"] = temporal_cosines(heads,prev_head)[head_ids]
                    else:
                        for column in ("change_comp","change_mod","change_head"):
                            change[column] = np.nan
                for cutoff in args.cutoffs:
                    selected = minimum > cutoff
                    if static is not None:
                        pieces[("features",weight,condition,cutoff)].append(static.loc[selected].copy())
                    if change is not None:
                        result = change.loc[selected].copy()
                        if previous is not None and previous_time == time-temporal:
                            prior_selected = previous_minimum > cutoff
                            modifier_present = np.zeros(len(words),dtype=bool)
                            head_present = np.zeros(len(words),dtype=bool)
                            modifier_present[mod_ids[prior_selected]] = True
                            head_present[head_ids[prior_selected]] = True
                            result.loc[~prior_selected[selected],"change_comp"] = np.nan
                            result.loc[~modifier_present[mod_ids[selected]],"change_mod"] = np.nan
                            result.loc[~head_present[head_ids[selected]],"change_head"] = np.nan
                        pieces[("temporal",weight,condition,cutoff)].append(result)
        previous, previous_minimum, previous_time = weighted, minimum, time
        print(f"Computed {tag}, temporal={temporal}, time={time}", flush=True)
    output_records = []
    vocabulary = specification["vocabulary"].loc[specification["vocabulary"]["rank"] <= size].copy()
    coverage = _coverage(specification,size)
    inputs = [{"name":name, **record} for name,record in specification["identity"]["inputs"].items()]
    inputs += [specification["identity"]["context_vocabulary"], *specification["identity"]["ratings"]]
    parent = specification["manifest"].get("source_info", {})
    corpus = specification["identity"]["corpus"]
    for family in args.families:
        for weight in args.weights:
            products = []
            for suffix, frame in (("contexts",vocabulary),("context_coverage",coverage)):
                path = output / f"{family}_{weight}_{tag}_{temporal}_{suffix}.parquet"
                _write_parquet(frame,path)
                products.append(artifact(path))
            for condition in args.conditions:
                for cutoff in args.cutoffs:
                    frames = pieces[(family,weight,condition,cutoff)]
                    if frames:
                        frame = pd.concat(frames,ignore_index=True)
                    else:
                        columns = (PAIR + ["change_comp","change_mod","change_head"]
                                   if family == "temporal" else _feature_columns(condition))
                        frame = pd.DataFrame(columns=columns)
                    selected = frame[PAIR].drop_duplicates()
                    path = output / f"{family}_{condition}_withSetting_{weight}_{tag}_{temporal}_{cutoff}_na.parquet"
                    # The shared COHA writer preserves rating metadata and its
                    # exact long/wide column layout, including eligibility.
                    write_features(frame,specification["ratings"],selected,path)
                    products.extend([artifact(path),artifact(path.with_name(path.stem+"_long.parquet"))])
            if family == "features" and "CompoundPooled" in args.conditions:
                frame = (pd.concat(constituents[weight],ignore_index=True).drop_duplicates()
                         if constituents[weight] else pd.DataFrame(columns=["word","time"]))
                path = output / f"setting_constituents_{weight}_{tag}_{temporal}.parquet"
                _write_parquet(frame,path)
                products.append(artifact(path))
            semantics = {**parameters, "conditions":args.conditions,"format":"parquet",
                "corpus":specification["identity"]["corpus_id"],
                "tagged_targets":not args.untagged,"ppmi":weight=="PPMI",
                "feature_definition_version":VERSION,"beta_policy":BETA_POLICY,
                "compound_population":_compound_population(args),
                "compound_vectors":"identical observations from compound_population in all conditions",
                "association_population":{"CompoundAware":"bigram_counts" if args.untagged else "eligible",
                    "CompoundPooled":"bigram_counts" if args.untagged else "eligible",
                    "CompoundAgnostic":"bigram_counts"},
                "ppmi_background":"full per-condition context-event population after vocabulary and time aggregation, before cutoff or rating filtering",
                "frequency_source":"independent raw occurrence counts","cutoff_comparison":"strictly greater than; shared across all three conditions",
                "context_vocabulary":{"requested_size":size,"actual_size":len(vocabulary),"selection":"ranked prefix of the frozen extraction lemma_POS vocabulary",
                    "sha256":specification["identity"]["context_vocabulary"]["sha256"],"fixed_across_time_bins_and_settings":True},
                "temporal_vector_availability":"both periods must retain the vector under the same shared raw cutoff; exact adjacent bins only",
                "setting_features":{"cosine":"full vector support","token_ratio":"raw role incidences / raw unigram occurrences",
                    "type_ratio":"positive raw context types before PPMI","pooled_pair_columns":"separate constituent_mod and constituent_head"},
                "imputation":"none; fit only inside training folds",
                "context_mode":specification["identity"]["context_mode"]}
            envelope = source_info(family,corpus,inputs,semantics,
                {"column":"time","unit":"calendar_year","bin_sizes":[temporal],
                 "binning":"floor(time / temporal) * temporal","pooled_bin":{"temporal":10000,"time":0}},
                specification["identity"]["software"],model=parent.get("model"),outputs=products,
                statistics={"evaluation_pairs":len(pairs),"evaluation_words":len(words),"output_files":len(products)},
                execution={"prepare_directory":str(specification["root"]),"options":specification["options"]})
            manifest = {"schema_version":3,**semantics,"source_info":envelope,"prepare_fingerprint":specification["fingerprint"]}
            path = output / f"{family}_{weight}_{tag}_{temporal}_manifest.json"
            write_json(path,manifest)
            output_records.extend(products)
            output_records.append(artifact(path))
    for record in specification["identity"]["inputs"].values():
        if _snapshot(record["path"]) != {key:record[key] for key in ("path","bytes","mtime_ns")}:
            raise RuntimeError("A core input changed during feature computation")
    complete = {"fingerprint":fingerprint,"parameters":parameters,
                "feature_definition_version":VERSION,"beta_policy":BETA_POLICY,
                "files":[_snapshot(record["path"]) for record in output_records],
                "outputs":output_records}
    write_json(marker,complete)
    print(f"Completed {tag}, temporal={temporal}: {len(output_records)} outputs",flush=True)
    return complete


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command",required=True)
    for command in ("prepare","features"):
        child = subparsers.add_parser(command)
        child.add_argument("--inputdir",required=True)
        child.add_argument("--outputdir",required=True)
        child.add_argument("--cache-directory",required=True)
        child.add_argument("--temp-directory",required=True)
        child.add_argument("--context-vocabulary",required=True)
        for name in ("reddy90","cordeiro90","cordeiro100"):
            child.add_argument(f"--{name}",required=True)
        child.add_argument("--untagged",action="store_true",help="Use all lexical bigram POS combinations and pool target POS; contexts stay tagged")
        child.add_argument("--context-sizes",type=int,nargs="+",default=list(CONTEXT_SIZES))
        child.add_argument("--temporal-bins",type=int,nargs="+",default=list(TEMPORAL_BINS))
        child.add_argument("--threads",type=int,default=4)
        child.add_argument("--memory-limit",default="48GB")
        add_partition_arguments(child)
        child.set_defaults(max_temp_directory_size="256GB",shuffle_buffer_mb=512)
        if command == "features":
            child.add_argument("--grid-index",type=int)
            child.add_argument("--context-size",type=int)
            child.add_argument("--temporal",type=int)
            child.add_argument("--cutoffs",type=int,nargs="+",default=list(CUTOFFS))
            child.add_argument("--weights",choices=("RAW","PPMI"),nargs="+",default=["RAW","PPMI"])
            child.add_argument("--families",choices=("features","temporal"),nargs="+",default=["features","temporal"])
            child.add_argument("--conditions",choices=CONDITIONS,nargs="+",default=list(CONDITIONS))
    args = parser.parse_args(argv)
    for name in ("context_sizes","temporal_bins"):
        values = list(dict.fromkeys(getattr(args,name)))
        if not values or min(values) < 1:
            parser.error("Vocabulary sizes and temporal bins must be positive")
        setattr(args,name,values)
    if args.threads < 1:
        parser.error("--threads must be positive")
    if args.command == "features":
        if args.grid_index is not None and (args.grid_index < 0 or args.context_size is not None or args.temporal is not None):
            parser.error("Use a nonnegative --grid-index OR an explicit --context-size/--temporal pair")
        if args.grid_index is None and (args.context_size is None or args.temporal is None):
            parser.error("features requires --grid-index or both --context-size and --temporal")
        if ((args.context_size is not None and args.context_size < 1)
                or (args.temporal is not None and args.temporal < 1)):
            parser.error("Vocabulary sizes and temporal bins must be positive")
        if min(args.cutoffs) < 0:
            parser.error("Cutoffs must be nonnegative")
        for name in ("cutoffs","weights","families","conditions"):
            setattr(args,name,list(dict.fromkeys(getattr(args,name))))
    partition_options(args)
    return args


def main(argv=None):
    args = parse_args(argv)
    return prepare(args) if args.command == "prepare" else run_features(args)


if __name__ == "__main__":
    main()
