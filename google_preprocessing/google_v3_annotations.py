"""Annotation-only, resumable enrichment of completed Google V3 feature inputs.

Hashes route records; joins always use complete cased surface/token keys. This
module deliberately does not modify the extraction or scientific reducers.
"""

import argparse
import contextlib
import fcntl
import importlib.util
import json
import os
import shutil
import tempfile
import time
from pathlib import Path

from src import partitioned_parquet as pp
from src.source_info import sha256_file, software_record


VERSION = 1
PAIR = ["modifier_surface", "modifier_pos", "head_surface", "head_pos", "decade"]
TOKEN = ["tokens_json", "modifier_index", "head_index"]
OUTCOMES = ["modifier_ner", "head_ner", "pair_ner", "same_entity_span",
            "either_token_entity", "sentence_start", "sentence_end"]
BASE_NER_COLUMNS = ["modifier_ner", "head_ner", "pair_ner", "modifier_ner_confidence",
           "head_ner_confidence", "pair_ner_confidence", "same_entity_span_fraction",
           "ner_coverage", "classified_ner_count", "sentence_start_fraction",
           "sentence_end_fraction", "ner_distribution_json"]
COVERAGE_NER_COLUMNS = ["either_token_entity_fraction", "modifier_entity_fraction",
           "head_entity_fraction", "available_ner_count", "snippet_classification_coverage",
           "ner_status", "ner_coverage_exceeds_one"]
DEPENDENCY = ["dependency_head_to_dependent_count", "dependency_dependent_to_head_count",
              "dependency_evidence_observed", "compound_candidate_pos"]


def _json(path):
    return json.loads(Path(path).read_text())


def _write_json(path, value):
    """Unique temporary names also make concurrent array jobs safe."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp",
                                              dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


@contextlib.contextmanager
def _lock(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        yield


def _settings(args):
    root = Path(args.output).resolve()
    work = Path(args.work_directory or root / "annotation_work").resolve()
    spill = Path(args.temp_directory or work / "spill").resolve()
    options = pp.partition_options(args)
    work.mkdir(parents=True, exist_ok=True)
    return root, work, spill, options


def _identity(kind, files, parameters):
    import duckdb
    import pyarrow
    return {"stage": kind, "version": VERSION,
            "sources": [pp._file_record(path) for path in files],
            "parameters": parameters, "code_sha256": sha256_file(__file__),
            "partition_backend_sha256": sha256_file(pp.__file__),
            "duckdb": duckdb.__version__, "pyarrow": pyarrow.__version__}


def _workspace(work, identity):
    path = work / identity["stage"] / pp._digest(identity)[:24]
    path.mkdir(parents=True, exist_ok=True)
    record = path / "identity.json"
    if record.exists() and _json(record) != identity:
        raise RuntimeError(f"Workspace identity mismatch: {path}")
    _write_json(record, identity)
    return path


def _paths_sql(paths):
    if not paths:
        raise ValueError("Empty Parquet source list")
    return "read_parquet([" + ",".join("'" + pp._quote(path) + "'" for path in paths) + \
           "], hive_partitioning=false)"


def _columns(names, alias=None):
    return ", ".join((alias + "." if alias else "") + '"' + n + '"' for n in names)


def _join(left, right, keys=PAIR):
    return " AND ".join(f'{left}."{name}"={right}."{name}"' for name in keys)


def _route_query(connection, query, destination, route_keys, columns, options,
                 partitions, by_time=False):
    """Route one bounded SQL result without retaining a whole Python table."""
    schema = connection.execute(f"SELECT {_columns(columns)} FROM ({query}) q LIMIT 0").to_arrow_table().schema
    routing = {"type": "time", "column": "decade"} if by_time else {"type": "hash"}
    writer = pp._BufferedPartitionWriter(destination, schema, routing, options)
    hashed = f"hash(0, {_columns(route_keys)}) % {int(partitions)}"
    reader = connection.execute(f"SELECT {_columns(columns)}, {hashed} AS __bucket FROM ({query}) q").to_arrow_reader(options["arrow_batch_rows"])
    rows, mass = 0, 0
    import pyarrow.compute as pc
    try:
        for batch in reader:
            rows += batch.num_rows
            if "count" in columns:
                mass += int(pc.sum(batch.column(batch.schema.get_field_index("count"))).as_py() or 0)
            pp._route_batch(writer, batch, routing)
        files = writer.finish()
    finally:
        reader.close()
        writer.close()
    return {"rows": rows, "count_sum": mass,
            "files": [pp._file_record(path, destination) for path in files]}, schema


def _bucket_paths(root, records):
    result = {}
    for record in records:
        path = Path(root) / record["path"]
        key = path.parent.relative_to(root).as_posix()
        # All routed files have the same final directory convention.
        key = "/".join(Path(key).parts[-2:]) if "decade=" in key else path.parent.name
        result.setdefault(key, []).append(path)
    return result


def _route_table(sources, columns, query, route_keys, root, spill, options,
                 partitions, by_time=False):
    """Single source scan, resumable at row-group batch boundaries."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    plan_path = root / "plan.json"
    if plan_path.exists():
        batches = _json(plan_path)["batches"]
        with pq.ParquetFile(sources[0]) as parquet:
            schema = pa.schema([parquet.schema_arrow.field(c) for c in columns])
    else:
        schema, batches = pp._plan_batches(sources, columns, options["batch_rows"])
        _write_json(plan_path, {"batches": batches})
    paths, rows, mass = {}, 0, 0
    with pp.connection_scope(options, spill) as connection:
        for index, units in enumerate(batches):
            folder = root / f"batch-{index:06d}"
            saved = pp._read_checkpoint(folder / "complete.json", folder)
            if saved is None:
                folder.mkdir(parents=True, exist_ok=True)
                observed = {"input_rows": 0, "input_count_sum": 0}
                reader = pp._arrow_reader(schema, units, options["arrow_batch_rows"], observed)
                connection.register("annotation_input", reader)
                try:
                    # Partial work uses a new private directory. Only completed
                    # file records are later read: interrupted fragments never count.
                    with tempfile.TemporaryDirectory(prefix="routing-", dir=folder) as temporary:
                        saved, unused = _route_query(connection, query, temporary, route_keys,
                                                    _output_columns(connection, query), options,
                                                    partitions, by_time)
                        published = folder / "data"
                        if published.exists():
                            # Only our own previously uncommitted batch product.
                            shutil.rmtree(published)
                        os.replace(temporary, published)
                        saved["files"] = [pp._file_record(published / r["path"], folder)
                                          for r in saved["files"]]
                    _write_json(folder / "complete.json", saved)
                    print(f"Routed {root.parent.name}: {index + 1}/{len(batches)} batches, "
                          f"{saved['rows']:,} rows", flush=True)
                finally:
                    connection.unregister("annotation_input")
                    reader.close()
            for key, files in _bucket_paths(folder, saved["files"]).items():
                paths.setdefault(key, []).extend(files)
            rows += saved["rows"]
            mass += saved["count_sum"]
    return paths, {"rows": rows, "count_sum": mass}


def _output_columns(connection, query):
    return [row[0] for row in connection.execute(f"DESCRIBE ({query})").fetchall()]


def prepare(args):
    root, work, spill, options = _settings(args)
    source = root / "reduced/ner_snippets.parquet"
    if not source.is_file():
        raise RuntimeError(f"Missing canonical cased snippets: {source}")
    identity = _identity("prepare", [source], {"shards": args.shards,
                         "batch_rows": options["batch_rows"],
                         "arrow_batch_rows": options["arrow_batch_rows"]})
    pointer = work / "prepare.json"
    if pointer.exists():
        previous = _json(pointer)
        if _compatible_prepare_identity(identity, previous["identity"]):
            previous_folder = Path(previous["workspace"])
            completed = pp._read_checkpoint(previous_folder / "complete.json", previous_folder)
            if completed is not None and completed["identity"] == previous["identity"]:
                print(f"Reusing compatible prepared snippets in {previous_folder}", flush=True)
                return completed
    folder = _workspace(work, identity)
    with _lock(folder / "stage.lock"):
        saved = pp._read_checkpoint(folder / "complete.json", folder)
        if saved is None:
            import pyarrow.parquet as pq
            with pq.ParquetFile(source) as parquet:
                columns = parquet.schema_arrow.names
            needed = PAIR + TOKEN + ["sentence_start", "sentence_end", "count"]
            if set(needed).difference(columns):
                raise RuntimeError("NER snippet schema does not preserve exact cased token/index keys")
            paths, statistics = _route_table([source], needed,
                f"SELECT {_columns(needed)} FROM annotation_input", TOKEN,
                folder / "occurrences", spill, options, args.shards)
            shards, records = [], []
            for index in range(args.shards):
                originals = paths.get(f"__bucket={index}", [])
                unique, stats = [], {"rows": 0, "count_sum": 0}
                if originals:
                    unique, stats = pp._aggregate_bucket(originals, TOKEN,
                        folder / f"unique/shard-{index:05d}", options, spill, 0)
                shards.append({"index": index, "occurrences": [str(p) for p in originals],
                               "unique": [str(p) for p in unique], "statistics": stats})
                records.extend(pp._file_record(p, folder) for p in [*originals, *unique])
            saved = {"identity": identity, "shards": shards, "statistics": statistics,
                     "files": records}
            saved["statistics"]["unique_snippets"] = sum(shard["statistics"]["rows"] for shard in shards)
            _write_json(folder / "complete.json", saved)
        _write_json(work / "prepare.json", {"workspace": str(folder), "identity": identity})
    print(f"Prepared {args.shards} NER shards in {folder}; "
          f"{saved['statistics']['unique_snippets']:,} unique token/index inputs before cutoff", flush=True)
    return saved


def _compatible_prepare_identity(current, saved):
    """Preparation can be reused only with the same source and code identity."""
    return current.get("stage") == "prepare" and current == saved


def _prepared(args):
    root, work, spill, options = _settings(args)
    pointer = work / "prepare.json"
    if not pointer.exists():
        raise RuntimeError("Run prepare before predicting or aggregating NER")
    record = _json(pointer)
    identity = _identity("prepare", [root / "reduced/ner_snippets.parquet"],
                        record["identity"]["parameters"])
    if not _compatible_prepare_identity(identity, record["identity"]) or identity["parameters"]["shards"] != args.shards:
        raise RuntimeError("Prepared snippets/source/code/shards changed; rerun prepare")
    folder = Path(record["workspace"])
    # Per-shard validation below avoids restatting every file in every array task.
    saved = _json(folder / "complete.json")
    if saved.get("identity") != record["identity"]:
        raise RuntimeError("Prepared checkpoint and pointer identities disagree; rerun prepare")
    return folder, saved


def ensure_spacy_compatibility():
    """Support srsly registry APIs used by installed transformer plugins.

    Some plugin releases call the registry as a decorator while newer srsly
    exposes register(name, func). Do not replace registry objects or encoders.
    """
    import srsly
    for name in ("msgpack_encoders", "msgpack_decoders"):
        registry = getattr(srsly, name, None)
        if registry is not None and not callable(registry):
            def decorate(self, key, func=None):
                def install(function):
                    self.register(key, function)
                    return function
                return install(func) if func is not None else install
            type(registry).__call__ = decorate


def _model_path(model):
    path = Path(model)
    if path.exists():
        candidates = [path.resolve()]
    else:
        spec = importlib.util.find_spec(model)
        if spec is None or not spec.origin:
            raise RuntimeError(f"NER model {model!r} is not installed")
        path = Path(spec.origin).parent
        candidates = [path]
    candidates += [p for p in path.iterdir() if p.is_dir()]
    for candidate in candidates:
        if (candidate / "config.cfg").is_file() and (candidate / "meta.json").is_file():
            return candidate.resolve()
    raise RuntimeError(f"No spaCy model data found under {model!r}")


def _model_identity(args):
    import importlib.metadata as metadata
    unused, work, unused, unused = _settings(args)
    path = _model_path(args.model)
    paths = sorted(p for p in path.rglob("*") if p.is_file())
    stats = [pp._file_record(p, path) for p in paths]
    versions = {}
    for package in ("spacy", "srsly", "thinc", "torch", "transformers", "spacy-transformers",
                    "spacy-curated-transformers", "curated-transformers", "numpy",
                    "cupy", "cupy-cuda11x", "cupy-cuda12x", "cupy-cuda13x"):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    cache = work / "model-fingerprints" / (pp._digest({"path": str(path), "files": stats,
                                                      "versions": versions}) + ".json")
    with _lock(cache.with_suffix(".lock")):
        if cache.exists():
            return _json(cache)
        result = {"path": str(path), "files": stats, "versions": versions,
                  "files_sha256": {p.relative_to(path).as_posix(): sha256_file(p) for p in paths},
                  "metadata": _json(path / "meta.json")}
        _write_json(cache, result)
        return result


def _prediction_workspace(args, prepared, model=None):
    unused, work, unused, unused = _settings(args)
    identity = _identity("predict", [], {"prepare": pp._digest(prepared["identity"]),
        "model": model or _model_identity(args), "min_snippet_count": args.min_snippet_count,
        "shards": args.shards, "classifier": "exact_token_spans_v1",
        "device": getattr(args, "device", "cpu"),
        "precision": "float32_highest_no_tf32_matmul"})
    return _workspace(work, identity), identity


def _validate_prepared_files(prepared_folder, prepared, paths):
    expected = {str(prepared_folder / record["path"]): record for record in prepared["files"]}
    for path in paths:
        if pp._file_record(path, prepared_folder) != expected.get(str(path)):
            raise RuntimeError("Prepared snippet partition changed; rerun prepare")


def classify_pair_entities(doc, modifier_index, head_index):
    if not (0 <= modifier_index < len(doc) and 0 <= head_index < len(doc)):
        raise ValueError("Modifier/head index is outside preserved token list")
    modifier = doc[modifier_index].ent_type_ or "NONE"
    head = doc[head_index].ent_type_ or "NONE"
    pair, same = "NONE", False
    for entity in doc.ents:
        if entity.start <= modifier_index < entity.end and entity.start <= head_index < entity.end:
            pair, same = entity.label_, True
            break
    return {"modifier_ner": modifier, "head_ner": head, "pair_ner": pair,
            "same_entity_span": same, "either_token_entity": modifier != "NONE" or head != "NONE"}


def _configure_device(args):
    """Select the backend explicitly; a CUDA request must never fall back."""
    ensure_spacy_compatibility()
    import spacy
    device = getattr(args, "device", "cpu")
    gpu_id = getattr(args, "gpu_id", 0)
    if gpu_id < 0:
        raise ValueError("--gpu-id must be nonnegative (Slurm normally exposes logical device 0)")
    if device not in ("cpu", "cuda"):
        raise ValueError("--device must be cpu or cuda")
    try:
        import torch
        torch.set_num_threads(max(1, args.threads))
        torch.set_float32_matmul_precision("highest")
    except ImportError as error:
        if device == "cuda":
            raise RuntimeError("CUDA NER requires a CUDA-enabled PyTorch installation") from error
        torch = None
    runtime = {"device": device, "threads": args.threads,
               "torch_version": str(torch.__version__) if torch else None,
               "cuda_build": torch.version.cuda if torch else None,
               "precision": "float32_highest_no_tf32_matmul"}
    if device == "cpu":
        spacy.require_cpu()
        return runtime
    if not torch.version.cuda:
        raise RuntimeError("CUDA was requested but PyTorch is CPU-only. Activate an environment with a CUDA-compatible PyTorch build.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but PyTorch cannot access a GPU. Request a GPU on your site's GPU partition and check the NVIDIA driver.")
    if gpu_id >= torch.cuda.device_count():
        raise RuntimeError(f"Logical GPU {gpu_id} is not visible; Slurm normally masks allocated GPUs so --gpu-id 0 is correct")
    try:
        import cupy
    except ImportError as error:
        raise RuntimeError("CUDA spaCy NER also requires a compatible CuPy wheel, e.g. cupy-cuda12x") from error
    try:
        spacy.require_gpu(gpu_id)
        from thinc.api import set_gpu_allocator
        # Share the PyTorch allocator rather than keeping two competing pools.
        set_gpu_allocator("pytorch")
        properties = torch.cuda.get_device_properties(gpu_id)
        runtime.update({"gpu_id": gpu_id, "gpu_name": properties.name,
                        "gpu_memory_bytes": int(properties.total_memory),
                        "compute_capability": [int(properties.major), int(properties.minor)],
                        "cupy_version": cupy.__version__,
                        "cuda_driver_version": int(cupy.cuda.runtime.driverGetVersion()),
                        "cuda_runtime_version": int(cupy.cuda.runtime.runtimeGetVersion())})
    except Exception as error:
        raise RuntimeError(f"Could not activate CUDA for spaCy (no CPU fallback): {error}") from error
    return runtime


def _load_model(args):
    args._ner_runtime = _configure_device(args)
    import spacy
    path = _model_path(args.model)
    components = _json(path / "meta.json").get("pipeline", [])
    # Keep the shared encoder, but exclude parser, POS prediction and lemmatizer.
    exclude = [name for name in components if name in ("tagger", "parser", "attribute_ruler",
                                                       "lemmatizer", "senter", "sentencizer")]
    model = spacy.load(path, exclude=exclude)
    if "ner" not in model.pipe_names:
        raise RuntimeError("The chosen spaCy model has no ner component")
    if not hasattr(model, "memory_zone"):
        raise RuntimeError("Annotation NER requires spaCy >=3.8 with memory_zone() to bound vocabulary memory")
    return model


def gpu_check(args):
    """Check CUDA kernels and optionally one model prediction, without datasets."""
    model = _load_model(args)
    runtime = dict(args._ner_runtime, pipeline=list(model.pipe_names))
    if args.device == "cuda":
        import torch
        import cupy
        try:
            # CuPy arithmetic exercises NVRTC/PTX, which driver version strings
            # and torch.cuda.is_available() alone cannot validate.
            torch_value = torch.arange(16, dtype=torch.float32, device=f"cuda:{args.gpu_id}").sum().item()
            cupy_value = float(cupy.arange(16, dtype=cupy.float32).sum().get())
            if torch_value != 120 or cupy_value != 120:
                raise RuntimeError("CUDA kernel results did not match their expected sums")
            runtime["cuda_kernel_check"] = "passed"
        except Exception as error:
            raise RuntimeError(f"CUDA kernel check failed; verify matching driver/toolkit/PyTorch/CuPy: {error}") from error
    if args.smoke_test:
        from spacy.tokens import Doc
        with model.memory_zone():
            doc = next(model.pipe([Doc(model.vocab, words=["visit", "New", "York", "in", "spring"])],
                                  batch_size=1, n_process=1))
            runtime["smoke_prediction"] = classify_pair_entities(doc, 1, 2)
    print(json.dumps(runtime, indent=2, sort_keys=True), flush=True)
    return runtime


def predict(args, predictor=None, model_identity=None):
    import pyarrow as pa
    import pyarrow.parquet as pq
    prepared_folder, prepared = _prepared(args)
    unused, work, spill, options = _settings(args)
    if not 0 <= args.shard_index < args.shards:
        raise ValueError("--shard-index must be in [0, --shards)")
    folder, identity = _prediction_workspace(args, prepared, model_identity)
    shard = prepared["shards"][args.shard_index]
    node = folder / f"shard-{args.shard_index:05d}"
    node.mkdir(parents=True, exist_ok=True)
    schema = pa.schema([("tokens_json", pa.string()), ("modifier_index", pa.int8()),
        ("head_index", pa.int8()), ("modifier_ner", pa.string()), ("head_ner", pa.string()),
        ("pair_ner", pa.string()), ("same_entity_span", pa.bool_()), ("either_token_entity", pa.bool_())])
    with _lock(node / "stage.lock"):
        saved = pp._read_checkpoint(node / "complete.json", node)
        if saved is not None:
            print(f"Reusing NER shard {args.shard_index}", flush=True)
            return saved
        _validate_prepared_files(prepared_folder, prepared, shard["unique"])
        plan_path = node / "plan.json"
        if plan_path.exists():
            plan_record = _json(plan_path)
            plan = plan_record["batches"]
            checkpoint_rows = plan_record["checkpoint_rows"]
        elif shard["unique"]:
            unused, plan = pp._plan_batches(shard["unique"], TOKEN + ["count"],
                                            args.prediction_batch_rows)
            checkpoint_rows = args.prediction_batch_rows
            _write_json(plan_path, {"batches": plan, "checkpoint_rows": checkpoint_rows})
        else:
            plan = []
            checkpoint_rows = args.prediction_batch_rows
            _write_json(plan_path, {"batches": plan, "checkpoint_rows": checkpoint_rows})
        model = None
        records, predicted, candidates = [], 0, 0
        if plan:
            with pq.ParquetFile(shard["unique"][0]) as parquet:
                table_schema = pa.schema([parquet.schema_arrow.field(c) for c in TOKEN + ["count"]])
        for index, units in enumerate(plan):
            stats = {"input_rows": 0, "input_count_sum": 0}
            reader = pp._arrow_reader(table_schema, units, checkpoint_rows, stats)
            try:
                for chunk_index, batch in enumerate(reader):
                    # Checkpoint at actual Arrow boundaries, not just row groups:
                    # a failure never loses earlier completed inference chunks.
                    batch_dir = node / f"batch-{index:06d}-{chunk_index:06d}"
                    batch_dir.mkdir(exist_ok=True)
                    batch_saved = pp._read_checkpoint(batch_dir / "complete.json", batch_dir)
                    if batch_saved is None:
                        started = time.monotonic()
                        result = batch_dir / "predictions.parquet"
                        temporary = batch_dir / "predictions.parquet.tmp"
                        selected = [row for row in batch.to_pylist()
                                    if row["count"] >= args.min_snippet_count]
                        results = []
                        if selected and predictor is None:
                            from spacy.tokens import Doc
                            if model is None:
                                model = _load_model(args)
                            with model.memory_zone():
                                docs = (Doc(model.vocab, words=json.loads(r["tokens_json"])) for r in selected)
                                for row, doc in zip(selected, model.pipe(docs, batch_size=args.ner_batch_size, n_process=1), strict=True):
                                    results.append({**{k: row[k] for k in TOKEN},
                                        **classify_pair_entities(doc, row["modifier_index"], row["head_index"])})
                        elif selected:
                            results = [{**{k: row[k] for k in TOKEN}, **predictor(row)} for row in selected]
                        try:
                            pq.write_table(pa.Table.from_pylist(results, schema=schema), temporary, compression="zstd")
                            os.replace(temporary, result)
                        finally:
                            temporary.unlink(missing_ok=True)
                        batch_saved = {"rows": len(selected), "unique_candidates": batch.num_rows,
                            "execution": {"ner_batch_size": args.ner_batch_size,
                                          "checkpoint_rows": checkpoint_rows,
                                          "seconds": time.monotonic() - started,
                                          "device": getattr(args, "_ner_runtime", {"device": args.device})},
                            "files": [pp._file_record(result, batch_dir)]}
                        _write_json(batch_dir / "complete.json", batch_saved)
                        seconds = batch_saved["execution"]["seconds"]
                        print(f"NER shard {args.shard_index}: checkpoint {index+1}.{chunk_index+1}, "
                              f"{len(selected):,} classified unique snippets in {seconds:.1f}s "
                              f"({len(selected)/max(seconds, 1e-9):.1f}/s; {args.device})", flush=True)
                    records.extend(pp._file_record(batch_dir / r["path"], node) for r in batch_saved["files"])
                    predicted += batch_saved["rows"]
                    candidates += batch_saved["unique_candidates"]
            finally:
                reader.close()
        saved = {"identity": identity, "rows": predicted, "unique_candidates": candidates,
                 "execution": getattr(args, "_ner_runtime", {"device": args.device}),
                 "files": records}
        _write_json(node / "complete.json", saved)
    return saved


def aggregate(args):
    prepared_folder, prepared = _prepared(args)
    unused, work, spill, options = _settings(args)
    predictions, pred_identity = _prediction_workspace(args, prepared)
    markers = []
    for index in range(args.shards):
        node = predictions / f"shard-{index:05d}"
        saved = pp._read_checkpoint(node / "complete.json", node)
        if saved is None:
            raise RuntimeError(f"NER shard {index} is unfinished; finish predict before aggregate")
        markers.append((node, saved))
    identity = _identity("aggregate", [], {"predict": pp._digest(pred_identity),
                                           "pair_partitions": args.partitions})
    folder = _workspace(work, identity)
    with _lock(folder / "stage.lock"):
        saved = pp._read_checkpoint(folder / "complete.json", folder)
        if saved is None:
            files, total_mass, rows = [], 0, 0
            for shard, (pred_node, pred_saved) in zip(prepared["shards"], markers, strict=True):
                node = folder / f"shard-{shard['index']:05d}"
                node.mkdir(parents=True, exist_ok=True)
                batch_saved = pp._read_checkpoint(node / "complete.json", node)
                if batch_saved is None:
                    _validate_prepared_files(prepared_folder, prepared, shard["occurrences"])
                    if not shard["occurrences"]:
                        batch_saved = {"rows": 0, "count_sum": 0, "files": []}
                    else:
                        with pp.connection_scope(options, spill) as connection:
                            prediction_files = [pred_node / r["path"] for r in pred_saved["files"]]
                            if prediction_files:
                                connection.execute(f"CREATE TEMP TABLE predictions AS SELECT * FROM {_paths_sql(prediction_files)}")
                            else:
                                connection.execute("CREATE TEMP TABLE predictions(tokens_json VARCHAR, modifier_index TINYINT, head_index TINYINT, modifier_ner VARCHAR, head_ner VARCHAR, pair_ner VARCHAR, same_entity_span BOOLEAN, either_token_entity BOOLEAN)")
                            connection.execute("CREATE TEMP VIEW occurrences AS SELECT * FROM " + _paths_sql(shard["occurrences"]))
                            selected = _columns(PAIR[:-1], "s") + ", s.decade::BIGINT AS decade, " + _columns(OUTCOMES[:5], "p") + ", s.sentence_start, s.sentence_end"
                            query = f"SELECT {selected}, SUM(s.count)::BIGINT AS count FROM occurrences s LEFT JOIN predictions p ON {_join('s', 'p', TOKEN)} GROUP BY ALL"
                            with tempfile.TemporaryDirectory(prefix="routing-", dir=node) as temporary:
                                batch_saved, unused = _route_query(connection, query, temporary,
                                    PAIR, PAIR + OUTCOMES + ["count"], options, args.partitions, True)
                                published = node / "data"
                                if published.exists():
                                    shutil.rmtree(published)
                                os.replace(temporary, published)
                                batch_saved["files"] = [pp._file_record(published / r["path"], node)
                                                        for r in batch_saved["files"]]
                    _write_json(node / "complete.json", batch_saved)
                    print(f"Aggregated NER evidence shard {shard['index']}", flush=True)
                files.extend(pp._file_record(node / r["path"], folder) for r in batch_saved["files"])
                rows += batch_saved["rows"]
                total_mass += batch_saved["count_sum"]
            if total_mass != prepared["statistics"]["count_sum"]:
                raise RuntimeError("NER join changed original snippet count mass")
            saved = {"identity": identity, "model": pred_identity["parameters"]["model"],
                     "prediction_device": pred_identity["parameters"]["device"],
                     "prediction_execution": [{"shard_index": index, **record.get("execution", {})}
                                              for index, (node, record) in enumerate(markers)],
                     "statistics": {"rows": rows, "count_sum": total_mass}, "files": files}
            _write_json(folder / "complete.json", saved)
        _write_json(work / "aggregate.json", {"workspace": str(folder), "identity": identity})
    return saved


def _ner_ctes():
    keys = _columns(PAIR)
    # Rows with NULL labels are unassessed; NONE is a genuine model outcome.
    return f"""
        events AS (SELECT {keys}, {_columns(OUTCOMES)}, SUM(count)::BIGINT AS count
                   FROM evidence GROUP BY ALL),
        totals AS (SELECT {keys}, SUM(count)::BIGINT AS available_ner_count,
            COALESCE(SUM(count) FILTER (WHERE modifier_ner IS NOT NULL),0)::BIGINT AS classified_ner_count,
            SUM(CASE WHEN same_entity_span THEN count ELSE 0 END) AS same_count,
            SUM(CASE WHEN either_token_entity THEN count ELSE 0 END) AS either_count,
            SUM(CASE WHEN modifier_ner != 'NONE' THEN count ELSE 0 END) AS modifier_entity_count,
            SUM(CASE WHEN head_ner != 'NONE' THEN count ELSE 0 END) AS head_entity_count,
            SUM(CASE WHEN modifier_ner IS NOT NULL AND sentence_start THEN count ELSE 0 END) AS start_count,
            SUM(CASE WHEN modifier_ner IS NOT NULL AND sentence_end THEN count ELSE 0 END) AS end_count
            FROM events GROUP BY {keys}),
        labels AS (SELECT {keys}, kind, label, SUM(count)::BIGINT AS weight
            FROM events, LATERAL (VALUES ('modifier',modifier_ner),('head',head_ner),('pair',pair_ner)) l(kind,label)
            WHERE label IS NOT NULL GROUP BY ALL),
        ranked AS (SELECT *, ROW_NUMBER() OVER (PARTITION BY {keys},kind ORDER BY weight DESC,label) AS rank FROM labels),
        modes AS (SELECT {keys},
            MAX(label) FILTER (WHERE kind='modifier') AS modifier_ner,
            MAX(label) FILTER (WHERE kind='head') AS head_ner,
            MAX(label) FILTER (WHERE kind='pair') AS pair_ner,
            MAX(weight) FILTER (WHERE kind='modifier') AS modifier_mode_count,
            MAX(weight) FILTER (WHERE kind='head') AS head_mode_count,
            MAX(weight) FILTER (WHERE kind='pair') AS pair_mode_count
            FROM ranked WHERE rank=1 GROUP BY {keys}),
        distributions AS (SELECT {_columns(PAIR,'l')},
            json_group_object(label,weight::DOUBLE/NULLIF(t.classified_ner_count,0))::VARCHAR AS ner_distribution_json
            FROM labels l JOIN totals t ON {_join('l','t')}
            WHERE kind='pair' GROUP BY {_columns(PAIR,'l')}),
        summary AS (SELECT t.*, m.modifier_ner,m.head_ner,m.pair_ner,
            m.modifier_mode_count,m.head_mode_count,m.pair_mode_count,d.ner_distribution_json
            FROM totals t LEFT JOIN modes m ON {_join('t','m')}
            LEFT JOIN distributions d ON {_join('t','d')})
    """


def _export_query(dependencies_only, baseline_columns):
    base = "a.modifier, a.head, a.decade::BIGINT AS time, a.modifier_surface, a.head_surface, a.modifier_pos, a.head_pos"
    if dependencies_only:
        fields = [f"a.{name}" if name in baseline_columns else
                  f"NULL::{'VARCHAR' if name in ['modifier_ner','head_ner','pair_ner','ner_distribution_json'] else 'BIGINT' if name == 'classified_ner_count' else 'DOUBLE'} AS {name}"
                  for name in BASE_NER_COLUMNS]
        fields += [f"a.{name}" if name in baseline_columns else
                   f"NULL::{'VARCHAR' if name=='ner_status' else 'BOOLEAN' if name=='ner_coverage_exceeds_one' else 'BIGINT' if name=='available_ner_count' else 'DOUBLE'} AS {name}"
                   for name in COVERAGE_NER_COLUMNS if name != "ner_status"]
        fields += ["a.ner_status" if "ner_status" in baseline_columns else
                   "CASE WHEN a.classified_ner_count>0 THEN 'assessed_existing' ELSE 'not_run' END AS ner_status"
                   if "classified_ner_count" in baseline_columns else "'not_run' AS ner_status"]
        ner = ", ".join(fields)
        ctes, ner_join = "", ""
    else:
        fraction = lambda column: f"n.{column}::DOUBLE/NULLIF(n.classified_ner_count,0)"
        ner = f"""n.modifier_ner,n.head_ner,n.pair_ner,
            {fraction('modifier_mode_count')} AS modifier_ner_confidence,
            {fraction('head_mode_count')} AS head_ner_confidence,
            {fraction('pair_mode_count')} AS pair_ner_confidence,
            {fraction('same_count')} AS same_entity_span_fraction,
            n.classified_ner_count::DOUBLE/NULLIF(a.count,0) AS ner_coverage,
            n.classified_ner_count,
            {fraction('start_count')} AS sentence_start_fraction,
            {fraction('end_count')} AS sentence_end_fraction,
            n.ner_distribution_json,
            {fraction('either_count')} AS either_token_entity_fraction,
            {fraction('modifier_entity_count')} AS modifier_entity_fraction,
            {fraction('head_entity_count')} AS head_entity_fraction,
            n.available_ner_count,
            n.classified_ner_count::DOUBLE/NULLIF(n.available_ner_count,0) AS snippet_classification_coverage,
            CASE WHEN n.available_ner_count IS NULL THEN 'unavailable'
                 WHEN n.classified_ner_count=0 THEN 'below_threshold'
                 WHEN n.classified_ner_count<n.available_ner_count THEN 'partial'
                 ELSE 'assessed' END AS ner_status,
            n.classified_ner_count>a.count AS ner_coverage_exceeds_one"""
        ctes, ner_join = _ner_ctes() + ",", "LEFT JOIN summary n ON " + _join("a", "n")
    return f"""WITH {ctes}
        dep AS (SELECT {_columns(PAIR)},
            COALESCE(SUM(count) FILTER (WHERE direction='head=>dependent'),0)::BIGINT AS forward_count,
            COALESCE(SUM(count) FILTER (WHERE direction='dependent<=head'),0)::BIGINT AS reverse_count
            FROM dependencies GROUP BY {_columns(PAIR)})
        SELECT {base}, {ner},
            COALESCE(d.forward_count,0)::BIGINT AS dependency_head_to_dependent_count,
            COALESCE(d.reverse_count,0)::BIGINT AS dependency_dependent_to_head_count,
            COALESCE(d.forward_count>0 OR d.reverse_count>0,FALSE) AS dependency_evidence_observed,
            a.modifier_pos || '-' || a.head_pos AS compound_candidate_pos,
            a.count::BIGINT AS count
        FROM annotations a {ner_join} LEFT JOIN dep d ON {_join('a','d')}
        WHERE a.modifier_pos IN ('ADJ','NOUN') AND a.head_pos='NOUN'"""


def _publish_link(source, destination):
    source, destination = Path(source), Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and os.path.samefile(source, destination):
        return
    descriptor, temporary = tempfile.mkstemp(prefix=destination.name+".", suffix=".tmp", dir=destination.parent)
    os.close(descriptor)
    Path(temporary).unlink()
    try:
        try:
            os.link(source, temporary)
        except OSError:
            shutil.copyfile(source, temporary)
        os.replace(temporary, destination)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _baseline(root, work):
    """Pin immutable original annotations before replacing canonical aliases."""
    import pyarrow.parquet as pq
    node = work / "baseline"
    node.mkdir(parents=True, exist_ok=True)
    manifest = node / "baseline.json"
    with _lock(node / "stage.lock"):
        if manifest.exists():
            record = _json(manifest)
            if Path(record["original_path"]) != root / "extracted/tables/bigram_annotations.parquet":
                raise RuntimeError("This annotation work directory belongs to a different output dataset")
            path = Path(record["file"]["path"])
            if pp._file_record(path) != record["file"]:
                raise RuntimeError("Pinned annotation baseline changed; do not reuse this annotation work directory")
            return path, record
        source = root / "extracted/tables/bigram_annotations.parquet"
        if not source.is_file():
            raise RuntimeError(f"Missing completed annotation baseline: {source}")
        path = node / "bigram_annotations.parquet"
        _publish_link(source, path)
        with pq.ParquetFile(path) as parquet:
            record = {"file": pp._file_record(path), "rows": parquet.metadata.num_rows,
                      "columns": parquet.schema_arrow.names, "original_path": str(source)}
        _write_json(manifest, record)
        return path, record


def _feature_snapshot(root):
    """Validate, but do not reread/hash/recount, scientific feature tables."""
    records = {}
    for name in ("bigrams", "bigram_counts", "unigrams", "unigram_counts"):
        path = root / f"extracted/feature_inputs/local5/{name}.parquet"
        if not path.is_file():
            raise RuntimeError(f"Missing completed scientific feature table: {path}")
        records[name] = pp._file_record(path)
    manifest_path = root / "extracted/feature_inputs/local5/corpus_manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError(f"Missing completed corpus manifest: {manifest_path}")
    manifest = _json(manifest_path)
    if manifest is not None:
        for name, actual in records.items():
            record = manifest.get("tables", {}).get(name)
            if not isinstance(record, dict) or not record.get("sha256"):
                raise RuntimeError(f"Missing saved scientific feature record/hash: {name}")
            if Path(record.get("path", "")).resolve() != Path(actual["path"]):
                raise RuntimeError(f"Scientific feature record path no longer matches this dataset: {name}")
            if record.get("bytes") != actual["bytes"]:
                raise RuntimeError(f"Feature table size no longer matches source information: {name}")
            reference = manifest.get("table_files", {}).get(name)
            if reference is None or (manifest_path.parent / reference).resolve() != Path(actual["path"]):
                raise RuntimeError(f"Scientific feature table_files path mismatch: {name}")
    return records, manifest_path, manifest


def export(args):
    import pyarrow as pa
    import pyarrow.parquet as pq
    root, work, spill, options = _settings(args)
    with _lock(root / "extracted/tables/.annotation-enrichment.lock"), _lock(work / "export.lock"):
        existing_source_info = root / "extracted/tables/bigram_annotations.parquet.json"
        if args.dependencies_only and existing_source_info.exists():
            existing = _json(existing_source_info)
            if existing.get("stage") == "annotation_enrichment" and existing.get("model"):
                raise RuntimeError("NER annotations are already published; dependencies-only would erase them. "
                                   "Rerun full export to preserve NER evidence instead.")
        snapshot, manifest_path, manifest = _feature_snapshot(root)
        baseline, baseline_record = _baseline(root, work)
        dependency = root / "reduced/dependencies.parquet"
        if not dependency.is_file():
            raise RuntimeError(f"Missing reduced dependency evidence: {dependency}")
        evidence_folder, evidence = None, None
        if not args.dependencies_only:
            pointer = work / "aggregate.json"
            if not pointer.exists():
                raise RuntimeError("Run aggregate before export, or use --dependencies-only")
            pointer = _json(pointer)
            prepared_folder, prepared = _prepared(args)
            unused, prediction_identity = _prediction_workspace(args, prepared)
            expected = _identity("aggregate", [], {"predict": pp._digest(prediction_identity),
                                                   "pair_partitions": args.partitions})
            if pointer["identity"] != expected:
                raise RuntimeError("Aggregate settings/model changed; rerun aggregate")
            evidence_folder = Path(pointer["workspace"])
            evidence = pp._read_checkpoint(evidence_folder / "complete.json", evidence_folder)
            if evidence is None:
                raise RuntimeError("Aggregated NER evidence is incomplete or changed")
        # Reuse the expensive dependency and baseline shuffle across NER models.
        route_identity = _identity("pair-routing", [baseline, dependency],
            {"partitions": args.partitions, "batch_rows": options["batch_rows"],
             "arrow_batch_rows": options["arrow_batch_rows"]})
        routes = _workspace(work, route_identity)
        columns = [c for c in baseline_record["columns"] if c != "time"] + ["time"]
        select = ",".join('"'+c+'"' for c in baseline_record["columns"] if c != "time")
        annotations, input_stats = _route_table([baseline], columns,
            f"SELECT {select}, time::BIGINT AS decade FROM annotation_input", PAIR,
            routes / "annotations", spill, options, args.partitions, True)
        deps, unused = _route_table([dependency], PAIR + ["direction", "count"],
            f"SELECT {_columns(PAIR[:-1])}, decade::BIGINT AS decade,direction,count FROM annotation_input",
            PAIR, routes / "dependencies", spill, options, args.partitions, True)
        ner_paths = _bucket_paths(evidence_folder, evidence["files"]) if evidence else {}
        identity = _identity("export", [], {"routing": pp._digest(route_identity),
            "aggregate": pp._digest(evidence["identity"]) if evidence else None,
            "dependencies_only": args.dependencies_only})
        folder = _workspace(work, identity)
        files, rows, mass, output_schema = [], 0, 0, None
        for key, paths in sorted(annotations.items()):
            node = folder / key
            node.mkdir(parents=True, exist_ok=True)
            saved = pp._read_checkpoint(node / "complete.json", node)
            if saved is None:
                destination = node / "annotations.parquet"
                temporary = node / "annotations.parquet.tmp"
                with pp.connection_scope(options, spill) as connection:
                    connection.execute("CREATE TEMP VIEW annotations AS SELECT * FROM " + _paths_sql(paths))
                    if key in deps:
                        connection.execute("CREATE TEMP VIEW dependencies AS SELECT * FROM " + _paths_sql(deps[key]))
                    else:
                        connection.execute("CREATE TEMP TABLE dependencies(modifier_surface VARCHAR,modifier_pos VARCHAR,head_surface VARCHAR,head_pos VARCHAR,decade BIGINT,direction VARCHAR,count BIGINT)")
                    if not args.dependencies_only:
                        if key in ner_paths:
                            connection.execute("CREATE TEMP VIEW evidence AS SELECT * FROM " + _paths_sql(ner_paths[key]))
                        else:
                            connection.execute("CREATE TEMP TABLE evidence(modifier_surface VARCHAR,modifier_pos VARCHAR,head_surface VARCHAR,head_pos VARCHAR,decade BIGINT,modifier_ner VARCHAR,head_ner VARCHAR,pair_ner VARCHAR,same_entity_span BOOLEAN,either_token_entity BOOLEAN,sentence_start BOOLEAN,sentence_end BOOLEAN,count BIGINT)")
                    unknown_direction = connection.execute("SELECT direction FROM dependencies WHERE direction IS NULL OR direction NOT IN ('head=>dependent','dependent<=head') LIMIT 1").fetchone()
                    if unknown_direction:
                        raise RuntimeError(f"Unknown dependency arrow: {unknown_direction[0]}")
                    query = _export_query(args.dependencies_only, baseline_record["columns"])
                    connection.execute(f"COPY ({query}) TO '{pp._quote(temporary)}' (FORMAT PARQUET,COMPRESSION ZSTD)")
                    output_rows, output_mass = connection.execute("SELECT COUNT(*),COALESCE(SUM(count),0) FROM read_parquet(?)", [str(temporary)]).fetchone()
                    eligible_rows, eligible_mass = connection.execute("SELECT COUNT(*),COALESCE(SUM(count),0) FROM annotations WHERE modifier_pos IN ('ADJ','NOUN') AND head_pos='NOUN'").fetchone()
                    if (output_rows, output_mass) != (eligible_rows, eligible_mass):
                        raise RuntimeError("Annotation join changed eligible row cardinality or adjacency count")
                os.replace(temporary, destination)
                saved = {"rows": int(output_rows), "count_sum": int(output_mass),
                         "files": [pp._file_record(destination, node)]}
                _write_json(node / "complete.json", saved)
                print(f"Enriched {key}: {output_rows:,} rows", flush=True)
            files.extend(node / r["path"] for r in saved["files"])
            rows += saved["rows"]
            mass += saved["count_sum"]
        if files:
            with pq.ParquetFile(files[0]) as parquet:
                output_schema = parquet.schema_arrow
        else:
            raise RuntimeError("No eligible baseline annotations to export")
        result = folder / "bigram_annotations.parquet"
        saved = pp._read_checkpoint(folder / "complete.json", folder)
        if saved is None:
            pp._concatenate(files, result, output_schema, options["arrow_batch_rows"])
            digest = sha256_file(result)
            saved = {"identity": identity, "rows": rows, "count_sum": mass,
                     "sha256": digest, "files": [pp._file_record(result, folder)]}
            _write_json(folder / "complete.json", saved)
        if _feature_snapshot(root)[0] != snapshot:
            raise RuntimeError("Scientific feature tables changed during annotation export; refusing publication")
        target = root / "extracted/tables/bigram_annotations.parquet"
        # Both aliases are repaired on a resumed publication, even if interrupted.
        _publish_link(result, target)
        _publish_link(result, root / "final/bigram_annotations.parquet")
        source_info = {"schema_version": 1, "stage": "annotation_enrichment", "status": "complete",
            "corpus": (manifest or {}).get("source_info", {}).get("corpus", {"name": "Google Books Ngrams"}),
            "time": (manifest or {}).get("source_info", {}).get("time", {"column": "time", "unit": "decade"}),
            "inputs": [dict(baseline_record["file"], name="pinned_annotation_baseline"),
                       dict(pp._file_record(dependency), name="unlabeled_dependencies")]
                       + ([dict(pp._file_record(root / "reduced/ner_snippets.parquet"),
                                name="cased_canonical_snippets")] if evidence else []),
            "identity": identity, "parameters": {
                "ner": "not run/preserved existing evidence" if args.dependencies_only else "weighted cased canonical five-gram predictions",
                "min_snippet_count": args.min_snippet_count if evidence else None,
                "confidence": "weighted modal-label share, NOT calibrated model probability",
                "coverage": "classified canonical snippet count / exact surface/POS/decade adjacent count; unclamped",
                "dependency_arrows": "separate source-notation counts, not positional directions or compound probabilities",
                "compound_label": "POS candidacy and observed dependency evidence only; no gold is_compound",
                "unknown": "NULL is unassessed; NONE is a model prediction"},
            "model": evidence["model"] if evidence else None,
            "software": software_record([__file__, pp.__file__], ["duckdb","pyarrow","spacy"]),
            "baseline": baseline_record,
            "outputs": [{"name": "tables/bigram_annotations.parquet", "path": str(target),
                         "bytes": target.stat().st_size, "sha256": saved["sha256"]}],
            "statistics": {"rows": rows, "count_sum": mass},
            "execution": {"work_directory": str(work), "temp_directory": str(spill),
                          "prediction_device": evidence.get("prediction_device") if evidence else None,
                          "prediction_execution": evidence.get("prediction_execution") if evidence else None,
                          "partition_options": options, "scientific_feature_hash_validation": "saved paths/sizes plus before-after stat snapshots; not full rehash"},
            "scientific_features_unchanged": snapshot}
        _write_json(str(target) + ".json", source_info)
        if manifest is not None:
            audit = manifest.setdefault("audit_tables", {}).setdefault("bigram_annotations", {})
            audit.update(source_info["outputs"][0])
            audit.update({"columns": output_schema.names, "format": "parquet", "required": True,
                          "count_unit": "eligible_ADJ_NOUN_and_NOUN_NOUN_surface_form_occurrences"})
            manifest["annotation_enrichment"] = source_info
            # Keep the original extraction/software record. Only audit records
            # and availability change; all core table hashes/statistics survive.
            feature_source_info = manifest.get("source_info", {})
            availability = feature_source_info.setdefault("parameters", {}).setdefault("annotation_availability", {})
            availability["ner"] = source_info["parameters"]["ner"]
            availability["dependencies"] = "separate unlabeled source-arrow counts; observed evidence only"
            for record in feature_source_info.get("outputs", []):
                if Path(record.get("path", "")).name == "bigram_annotations.parquet":
                    record.update(source_info["outputs"][0])
            table_stats = feature_source_info.get("statistics", {}).get("tables", {})
            if "bigram_annotations" in table_stats:
                table_stats["bigram_annotations"].update({"rows": rows, "count_sum": mass})
            _write_json(manifest_path, manifest)
        print(f"Published {target}; scientific feature files unchanged", flush=True)
        return source_info


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    stages = result.add_subparsers(dest="stage", required=True)
    for stage in ("prepare", "predict", "aggregate", "export", "gpu-check"):
        p = stages.add_parser(stage)
        p.add_argument("--output", required=True)
        p.add_argument("--temp-directory")
        p.add_argument("--model", default="en_core_web_trf")
        p.add_argument("--device", choices=("cpu", "cuda"), default="cpu",
                       help="NER prediction backend; preparation/finalization stay CPU-only")
        p.add_argument("--gpu-id", type=int, default=0,
                       help="logical CUDA device within CUDA_VISIBLE_DEVICES (normally 0 in Slurm)")
        p.add_argument("--shards", type=int, default=128)
        p.add_argument("--threads", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
        p.add_argument("--memory-limit", default="48GB")
        p.add_argument("--min-snippet-count", type=int, default=20)
        p.add_argument("--ner-batch-size", type=int, default=32)
        p.add_argument("--prediction-batch-rows", type=int, default=10000)
        p.add_argument("--work-directory", help="resumable annotations; default OUTPUT/annotation_work")
        p.add_argument("--partitions", type=int, default=128, help="pair hash buckets per decade")
        p.add_argument("--partition-batch-rows", type=int, default=50000000,
                       help="maximum input rows per checkpointed routing batch (row-group aligned)")
        p.add_argument("--arrow-batch-rows", type=int, default=65536)
        p.add_argument("--max-split-depth", type=int, default=4,
                       help="recursive OOM subdivisions during unique-snippet deduplication")
        p.add_argument("--max-temp-directory-size", default="1TB",
                       help="DuckDB spill cap per connection, not total persistent scratch")
        p.add_argument("--shuffle-buffer-mb", type=int, default=256)
        p.add_argument("--shuffle-row-group-rows", type=int, default=131072)
        p.add_argument("--shuffle-file-rows", type=int, default=2000000)
        p.add_argument("--shuffle-max-open-files", type=int, default=512)
        p.add_argument("--aggregation-workers", type=int, choices=(1,), default=1,
                       help="annotation partitions run sequentially; SQL uses --threads")
        if stage == "predict":
            p.add_argument("--shard-index", type=int, required=True)
        if stage == "export":
            p.add_argument("--dependencies-only", action="store_true")
        if stage == "gpu-check":
            p.add_argument("--smoke-test", action="store_true", help="also predict entities in one five-token snippet")
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    if args.gpu_id < 0:
        raise ValueError("--gpu-id must be nonnegative")
    if min(args.shards, args.threads, args.ner_batch_size, args.prediction_batch_rows) < 1 or args.min_snippet_count < 0:
        raise ValueError("Shard/thread/batch sizes must be positive and count threshold nonnegative")
    return {"prepare": prepare, "predict": predict, "aggregate": aggregate,
            "export": export, "gpu-check": gpu_check}[args.stage](args)


if __name__ == "__main__":
    main()
