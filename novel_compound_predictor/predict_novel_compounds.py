#!/usr/bin/env python3
"""Train and use a noun-compound plausibility classifier.

The pooled model is Linear(2D, H) -> ReLU -> Linear(H, 2, bias=False).
The optional temporal model first encodes each role's chronological D-dimensional
decade sequence with a separate, one-layer LSTM. Its hidden size defaults to D;
this and end-to-end supervised LSTM training are explicit implementation choices.

Input embeddings: headered CSV, TSV or Parquet with modifier/head (or word/token),
optional decade, and dim_0..dim_299. Numeric 0..299 or embedding0..299 names are
also accepted. All embeddings must share the representation builder's SVD basis.
Only years <= --cutoff-year are used. A per-decade table requires a cutoff ending
in 9: its aggregated vectors cannot be split into partial decades.
Pooled tables must already have been built
from pre-cutoff counts: no downstream program can remove leakage from their SVD.
Pooled mode averages a supplied per-decade table; a table built
by summing historical counts before SVD is another, explicitly different input.

Positive pairs are normally newly attested 2000s compounds, with constituent
vectors restricted to pre-2000 data. --positive-flag novel can select those rows
from compound_metadata.csv. Split positive pairs BEFORE corrupting one role;
negatives are unique across splits and exclude EVERY pair in --attested, including
future positives. This is supervised disambiguation with randomly held-out pairs,
not a claim of unsupervised forecasting.

Ranking scores are logit(plausible) - logit(implausible), NOT probabilities.
For ranking, --attested should contain historical pairs only, so genuinely new
future pairs remain eligible. --candidates avoids a large Cartesian search.
"""

from __future__ import annotations

import argparse
import csv
import heapq
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
import random
import re
from typing import Iterable, Iterator

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from src.source_info import sha256_file, source_identity_paths


FORMAT_VERSION = 2


def implementation_identity() -> dict[str, str]:
    """Identify the predictor and its shared support code."""
    return {path.name: sha256_file(path) for path in [Path(__file__), *source_identity_paths()]}


def read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    # Detect the delimiter from the header rather than assuming it from the suffix.
    with path.open(encoding="utf-8-sig") as stream:
        first = stream.readline()
    return pd.read_csv(path, sep="\t" if "\t" in first else ",")


def truth_mask(series: pd.Series) -> pd.Series:
    values = series.astype(str).str.lower().str.strip()
    if not values.isin(["true", "false", "1", "0", "1.0", "0.0"]).all():
        raise ValueError(f"Flag {series.name!r} must contain only true/false or 1/0")
    return values.isin(["true", "1", "1.0"])


def read_pairs(path: str | Path, flag: str | None = None) -> list[tuple[str, str]]:
    table = read_table(path)
    if not {"modifier", "head"}.issubset(table.columns):
        raise ValueError(f"{path}: expected named modifier and head columns")
    if flag:
        if flag not in table:
            raise ValueError(f"{path}: no flag column {flag!r}")
        table = table.loc[truth_mask(table[flag])]
    if table[["modifier", "head"]].isna().any().any():
        raise ValueError(f"{path}: missing modifier/head values")
    pairs = {(str(m), str(h)) for m, h in table[["modifier", "head"]].itertuples(index=False, name=None)}
    if any(not m.strip() or not h.strip() for m, h in pairs):
        raise ValueError(f"{path}: empty modifier/head values")
    return sorted(pairs)


def dimension_columns(table: pd.DataFrame, dimensions: int) -> list:
    for prefix in ("dim_", "embedding", ""):
        columns = [f"{prefix}{i}" for i in range(dimensions)]
        if set(columns).issubset(table.columns):
            candidates = [c for c in table.columns if re.fullmatch(re.escape(prefix) + r"\d+", str(c))]
            if len(candidates) != dimensions:
                raise ValueError(f"Expected exactly {dimensions} embedding columns, found {len(candidates)}")
            return columns
    # Parquet may preserve integer dimension column names.
    if set(range(dimensions)).issubset(table.columns):
        numeric = [c for c in table.columns if isinstance(c, int)]
        if len(numeric) == dimensions:
            return list(range(dimensions))
    raise ValueError(f"Expected {dimensions} dimensions named dim_0..dim_{dimensions - 1}")


@dataclass
class Embeddings:
    tokens: list[str]
    values: torch.Tensor
    dimensions: int
    decades: list[int]

    def __post_init__(self):
        self.index = {token: i for i, token in enumerate(self.tokens)}


def load_embeddings(path: str | Path, role: str, dimensions: int, encoder: str,
                    start_year: int, cutoff_year: int) -> Embeddings:
    table = read_table(path)
    token_column = next((column for column in (role, "token", "word") if column in table), None)
    if token_column is None:
        raise ValueError(f"{path}: expected {role}, token or word column")
    columns = dimension_columns(table, dimensions)
    if table[token_column].isna().any():
        raise ValueError(f"{path}: missing tokens")
    table = table.copy()
    table[token_column] = table[token_column].astype(str)
    if table[token_column].str.strip().eq("").any():
        raise ValueError(f"{path}: empty tokens")
    if "decade" in table:
        if cutoff_year % 10 != 9:
            raise ValueError("Per-decade embeddings contain complete decades; --cutoff-year must end in 9 "
                             "because partial-decade aggregates cannot be separated")
        years = pd.to_numeric(table["decade"], errors="raise")
        if not np.isfinite(years).all() or not (years % 10 == 0).all():
            raise ValueError(f"{path}: decade must be a finite multiple of ten")
        table["decade"] = years.astype(int)
        # In particular discard 2000s vectors BEFORE grouping, vocabulary and tensors.
        table = table.loc[table["decade"].between(start_year, cutoff_year)]
    elif encoder == "lstm":
        raise ValueError("LSTM mode requires per-decade embeddings with a decade column")
    if table.empty:
        raise ValueError(f"{path}: no embeddings within the requested historical interval")
    try:
        table[columns] = table[columns].astype(np.float32)
    except (ValueError, TypeError) as error:
        raise ValueError(f"{path}: embedding values must be numeric") from error
    if not np.isfinite(table[columns].to_numpy()).all():
        raise ValueError(f"{path}: embedding values contain NaN or infinity")
    keys = [token_column, "decade"] if "decade" in table else [token_column]
    if table.duplicated(keys).any():
        raise ValueError(f"{path}: duplicate {keys}; choose one shared embedding per token/decade")
    tokens = sorted(table[token_column].unique().tolist())
    if encoder == "pooled":
        grouped = table.groupby(token_column, sort=True)[columns].mean().reindex(tokens)
        values = torch.from_numpy(grouped.to_numpy(dtype=np.float32, copy=True))
        return Embeddings(tokens, values, dimensions, [])
    decades = list(range(start_year, cutoff_year // 10 * 10 + 1, 10))
    # A missing decade means no observed vector; preserve its zero-filled position.
    matrix = np.zeros((len(tokens), len(decades), dimensions), dtype=np.float32)
    token_indices = pd.Index(tokens).get_indexer(table[token_column])
    decade_indices = (table["decade"].to_numpy() - start_year) // 10
    matrix[token_indices, decade_indices] = table[columns].to_numpy(dtype=np.float32)
    return Embeddings(tokens, torch.from_numpy(matrix), dimensions, decades)


class CompoundNNM(nn.Module):
    def __init__(self, dimensions: int = 300, hidden_size: int = 300,
                 encoder: str = "pooled", lstm_hidden_size: int = 300):
        super().__init__()
        self.encoder = encoder
        if encoder == "lstm":
            self.modifier_encoder = nn.LSTM(dimensions, lstm_hidden_size, batch_first=True)
            self.head_encoder = nn.LSTM(dimensions, lstm_hidden_size, batch_first=True)
            encoded_size = lstm_hidden_size
        else:
            encoded_size = dimensions
        self.fc1 = nn.Linear(2 * encoded_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 2, bias=False)

    def encode(self, values: torch.Tensor, role: str) -> torch.Tensor:
        if self.encoder == "pooled":
            return values
        encoder = self.modifier_encoder if role == "modifier" else self.head_encoder
        _, (hidden, _) = encoder(values)
        return hidden[-1]

    def compose(self, modifiers: torch.Tensor, heads: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.relu(self.fc1(torch.cat((modifiers, heads), dim=1))))

    def forward(self, modifiers: torch.Tensor, heads: torch.Tensor) -> torch.Tensor:
        return self.compose(self.encode(modifiers, "modifier"), self.encode(heads, "head"))


def seed_everything(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False


def split_positives(pairs: list[tuple[str, str]], seed: int, validation_fraction: float,
                    test_fraction: float) -> dict[str, list[tuple[str, str]]]:
    if not 0 < validation_fraction < 1 or not 0 < test_fraction < 1:
        raise ValueError("Validation and test fractions must each be between zero and one")
    if validation_fraction + test_fraction >= 1:
        raise ValueError("Validation and test fractions must leave a training split")
    pairs = sorted(set(pairs))
    random.Random(seed).shuffle(pairs)
    n_validation = max(1, math.floor(len(pairs) * validation_fraction))
    n_test = max(1, math.floor(len(pairs) * test_fraction))
    n_train = len(pairs) - n_validation - n_test
    if n_train < 1:
        raise ValueError("Need at least three available unique positive pairs")
    return {"train": pairs[:n_train], "validation": pairs[n_train:n_train + n_validation],
            "test": pairs[n_train + n_validation:]}


def corrupt_splits(splits: dict[str, list[tuple[str, str]]], modifiers: list[str], heads: list[str],
                   attested: set[tuple[str, str]], corrupt: str, seed: int) -> pd.DataFrame:
    rng = random.Random(seed)
    used: set[tuple[str, str]] = set()
    positives = {pair for pairs in splits.values() for pair in pairs}
    forbidden = attested | positives
    vocabulary = sorted(set(heads if corrupt == "head" else modifiers))
    if not vocabulary:
        raise ValueError("No attested replacement constituents have historical embeddings")
    rows = []
    for split, pairs in splits.items():
        for modifier, head in pairs:
            def candidate(replacement):
                return (modifier, replacement) if corrupt == "head" else (replacement, head)
            negative = None
            for _ in range(100):
                pair = candidate(rng.choice(vocabulary))
                if pair not in forbidden and pair not in used:
                    negative = pair
                    break
            if negative is None:
                # Bounded fallback scans ONE role vocabulary, never a Cartesian array.
                available = [candidate(word) for word in vocabulary
                             if candidate(word) not in forbidden and candidate(word) not in used]
                if available:
                    negative = rng.choice(available)
            if negative is None:
                raise ValueError(f"Cannot generate a unique Corrupt{corrupt.title()} negative for "
                                 f"{modifier!r} {head!r}; use a broader vocabulary or the other role")
            used.add(negative)
            rows.append((modifier, head, 1, split, modifier, head))
            rows.append((*negative, 0, split, modifier, head))
    return pd.DataFrame(rows, columns=["modifier", "head", "label", "split", "source_modifier", "source_head"])


def make_loader(examples: pd.DataFrame, modifiers: Embeddings, heads: Embeddings,
                batch_size: int, shuffle: bool = False, seed: int = 1991) -> DataLoader:
    modifier_indices = torch.tensor([modifiers.index[m] for m in examples.modifier], dtype=torch.long)
    head_indices = torch.tensor([heads.index[h] for h in examples["head"]], dtype=torch.long)
    labels = torch.tensor(examples.label.to_numpy(), dtype=torch.long)
    return DataLoader(TensorDataset(modifier_indices, head_indices, labels), batch_size=batch_size,
                      shuffle=shuffle, generator=torch.Generator().manual_seed(seed), num_workers=0)


def evaluate(model: CompoundNNM, loader: DataLoader, modifiers: Embeddings, heads: Embeddings,
             device: torch.device) -> tuple[dict, list[float]]:
    model.eval()
    correct = total = 0
    loss_sum = 0.0
    scores = []
    with torch.inference_mode():
        for modifier_indices, head_indices, labels in loader:
            labels = labels.to(device)
            logits = model(modifiers.values[modifier_indices].to(device), heads.values[head_indices].to(device))
            loss_sum += nn.functional.cross_entropy(logits, labels, reduction="sum").item()
            correct += (logits.argmax(1) == labels).sum().item()
            total += len(labels)
            scores.extend((logits[:, 1] - logits[:, 0]).cpu().tolist())
    return {"examples": total, "accuracy": correct / total, "loss": loss_sum / total}, scores


def representation_manifests(modifier_path: str | Path, head_path: str | Path,
                             cutoff_year: int, dimensions: int) -> dict:
    """Check available builder metadata, including the dates used to FIT the SVD."""
    records = {}
    for role, path in [("modifiers", modifier_path), ("heads", head_path)]:
        manifest_path = Path(path).resolve().parent / "representation_manifest.json"
        if not manifest_path.exists():
            continue
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("status") != "complete":
            raise ValueError(f"{manifest_path}: representations are not marked complete")
        end_exclusive = manifest.get("time_window", {}).get("end_exclusive")
        if not isinstance(end_exclusive, int) or end_exclusive > cutoff_year + 1:
            raise ValueError(f"{manifest_path}: SVD training window includes future data "
                             f"(end_exclusive={end_exclusive}, cutoff={cutoff_year})")
        parameters = manifest.get("parameters", {})
        if parameters.get("train_before", end_exclusive) != end_exclusive:
            raise ValueError(f"{manifest_path}: inconsistent SVD training dates")
        if parameters.get("dimensions", dimensions) != dimensions:
            raise ValueError(f"{manifest_path}: embedding dimensions do not match --dimensions")
        records[role] = {"path": str(manifest_path), "sha256": sha256_file(manifest_path), "content": manifest}
    if len(records) == 1:
        raise ValueError("One input has a representation manifest and the other does not; "
                         "use both role tables from the same shared SVD output")
    if len(records) == 2 and records["modifiers"]["content"] != records["heads"]["content"]:
        raise ValueError("Role representation manifests differ; use a single shared SVD basis for both roles")
    return records


def choose_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable; use --device cpu")
    return device


def train(args: argparse.Namespace) -> dict:
    seed_everything(args.seed)
    if (args.epochs < 1 or args.batch_size < 1 or not math.isfinite(args.learning_rate)
            or args.learning_rate <= 0 or min(args.dimensions, args.hidden_size, args.lstm_hidden_size) < 1):
        raise ValueError("Epochs, batch size, dimensions, hidden sizes and learning rate must be positive and finite")
    if args.start_year % 10 or args.cutoff_year < args.start_year:
        raise ValueError("Start year must be a decade boundary and no later than cutoff year")
    output_dir = Path(args.output_dir)
    if (output_dir / "model.pt").exists() and not args.overwrite:
        raise ValueError(f"{output_dir}/model.pt exists; choose another --output-dir or use --overwrite")
    device = choose_device(args.device)
    manifests = representation_manifests(args.modifiers, args.heads, args.cutoff_year, args.dimensions)
    modifiers = load_embeddings(args.modifiers, "modifier", args.dimensions, args.encoder,
                                args.start_year, args.cutoff_year)
    heads = load_embeddings(args.heads, "head", args.dimensions, args.encoder,
                            args.start_year, args.cutoff_year)
    all_positives = read_pairs(args.positives, args.positive_flag)
    attested = set(read_pairs(args.attested))
    missing_attested = set(all_positives) - attested
    if missing_attested:
        raise ValueError("--attested must include all positive pairs, including future positives; "
                         f"missing {len(missing_attested)}")
    positives = [pair for pair in all_positives if pair[0] in modifiers.index and pair[1] in heads.index]
    dropped = sorted(set(all_positives) - set(positives))
    splits = split_positives(positives, args.seed, args.validation_fraction, args.test_fraction)
    # Replacement constituents must come from attested compounds, in the same role.
    replacement_modifiers = sorted({m for m, _ in attested if m in modifiers.index})
    replacement_heads = sorted({h for _, h in attested if h in heads.index})
    examples = corrupt_splits(splits, replacement_modifiers, replacement_heads, attested, args.corrupt, args.seed)
    subsets = {split: examples.loc[examples.split.eq(split)].reset_index(drop=True) for split in splits}
    loaders = {split: make_loader(rows, modifiers, heads, args.batch_size, split == "train", args.seed)
               for split, rows in subsets.items()}
    model_config = {"dimensions": args.dimensions, "hidden_size": args.hidden_size, "encoder": args.encoder,
                    "lstm_hidden_size": args.lstm_hidden_size}
    model = CompoundNNM(**model_config).to(device)
    optimizer_class = torch.optim.SGD if args.optimizer == "sgd" else torch.optim.Adam
    optimizer = optimizer_class(model.parameters(), lr=args.learning_rate)
    history = []
    best_validation_loss = float("inf")
    best_state = None
    best_epoch = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_examples = 0
        for modifier_indices, head_indices, labels in loaders["train"]:
            optimizer.zero_grad(set_to_none=True)
            logits = model(modifiers.values[modifier_indices].to(device), heads.values[head_indices].to(device))
            loss = nn.functional.cross_entropy(logits, labels.to(device))
            if not torch.isfinite(loss):
                raise ValueError("Training loss is non-finite; check vectors and reduce the learning rate")
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(labels)
            total_examples += len(labels)
        validation, _ = evaluate(model, loaders["validation"], modifiers, heads, device)
        row = {"epoch": epoch, "train_loss": total_loss / total_examples, "validation": validation}
        history.append(row)
        print(json.dumps(row), flush=True)
        if validation["loss"] < best_validation_loss:
            best_validation_loss = validation["loss"]
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            best_epoch = epoch
    if best_state is None:
        raise ValueError("Validation loss is non-finite; no valid checkpoint was produced")
    model.load_state_dict(best_state)
    # The test set is evaluated only after validation-based model selection.
    test_metrics, test_scores = evaluate(model, loaders["test"], modifiers, heads, device)
    validation_metrics, _ = evaluate(model, loaders["validation"], modifiers, heads, device)
    embedding_config = {"start_year": args.start_year, "cutoff_year": args.cutoff_year}
    checkpoint = {"format_version": FORMAT_VERSION, "model_config": model_config, "state_dict": best_state,
                  "code_sha256": implementation_identity(),
                  "embedding_config": embedding_config,
                  "representation_manifests": manifests,
                  "embedding_sha256": {"modifiers": sha256_file(args.modifiers), "heads": sha256_file(args.heads)},
                  "seed": args.seed, "best_epoch": best_epoch, "torch_version": str(torch.__version__),
                  "score_definition": "logit_plausible_minus_logit_implausible"}
    metrics = {"model_config": model_config, "embedding_config": embedding_config,
               "code_sha256": checkpoint["code_sha256"],
               "representation_manifests": manifests,
               "embedding_source_info_warning": None if manifests else
               "No representation manifest: independently verify that the embedding SVD was fitted only "
               "to pre-cutoff counts. Dropping future rows cannot remove leakage from an SVD basis.",
               "embedding_sha256": checkpoint["embedding_sha256"], "seed": args.seed,
               "positive_pairs_requested": len(all_positives), "positive_pairs_used": len(positives),
               "positive_pairs_missing_historical_constituents": len(dropped),
               "corrupt": args.corrupt, "optimizer": args.optimizer, "learning_rate": args.learning_rate,
               "epochs": args.epochs, "batch_size": args.batch_size, "best_epoch": best_epoch,
               "validation": validation_metrics, "test": test_metrics, "history": history,
               "interpretation": "Supervised held-out-pair disambiguation; scores are uncalibrated logit margins.",
               "temporal_encoder_choices": "Separate role LSTMs, one layer, zero-filled missing decades, "
               "final hidden state, trained jointly with NNM." if args.encoder == "lstm" else None,
               "input_paths": {name: str(Path(getattr(args, name)).resolve())
                               for name in ["modifiers", "heads", "positives", "attested"]}}
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_tmp = output_dir / "model.pt.tmp"
    torch.save(checkpoint, checkpoint_tmp)
    checkpoint_tmp.replace(output_dir / "model.pt")
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    examples.to_csv(output_dir / "split_examples.csv", index=False)
    pd.DataFrame(dropped, columns=["modifier", "head"]).to_csv(output_dir / "missing_constituents.csv", index=False)
    predictions = subsets["test"].copy()
    predictions["score"] = test_scores
    predictions["prediction"] = predictions.score.gt(0).astype(int)
    predictions.to_csv(output_dir / "test_predictions.csv", index=False)
    print(json.dumps({"checkpoint": str(output_dir / "model.pt"), "test": test_metrics}), flush=True)
    return metrics


def load_checkpoint(path: str | Path, device: torch.device) -> tuple[CompoundNNM, dict]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("format_version") != FORMAT_VERSION:
        raise ValueError("Unsupported checkpoint format; train a model with the current predictor")
    if checkpoint.get("code_sha256") != implementation_identity():
        raise ValueError("Checkpoint implementation differs from the current predictor or shared support code")
    model = CompoundNNM(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint


def encoded_table(model: CompoundNNM, embeddings: Embeddings, role: str,
                  batch_size: int, device: torch.device) -> torch.Tensor:
    with torch.inference_mode():
        return torch.cat([model.encode(embeddings.values[start:start + batch_size].to(device), role).cpu()
                          for start in range(0, len(embeddings.tokens), batch_size)])


def batched(pairs: Iterable[tuple[str, str]], batch_size: int) -> Iterator[list[tuple[str, str]]]:
    batch = []
    for pair in pairs:
        batch.append(pair)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def rank_pairs(model: CompoundNNM, modifiers: Embeddings, heads: Embeddings,
               pairs: Iterable[tuple[str, str]], excluded: set[tuple[str, str]],
               top_k: int, batch_size: int, device: torch.device) -> tuple[list[tuple[float, str, str]], dict]:
    modifier_vectors = encoded_table(model, modifiers, "modifier", batch_size, device)
    head_vectors = encoded_table(model, heads, "head", batch_size, device)
    heap = []
    counts = {"scored": 0, "excluded_attested": 0, "missing_constituents": 0}

    def eligible():
        for m, h in pairs:
            if (m, h) in excluded:
                counts["excluded_attested"] += 1
            elif m not in modifiers.index or h not in heads.index:
                counts["missing_constituents"] += 1
            else:
                yield (m, h)

    with torch.inference_mode():
        for batch in batched(eligible(), batch_size):
            m_indices = [modifiers.index[m] for m, _ in batch]
            h_indices = [heads.index[h] for _, h in batch]
            logits = model.compose(modifier_vectors[m_indices].to(device), head_vectors[h_indices].to(device))
            scores = (logits[:, 1] - logits[:, 0]).cpu().tolist()
            for (m, h), score in zip(batch, scores):
                if not math.isfinite(score):
                    raise ValueError("Non-finite ranking score")
                item = (score, m, h)
                if len(heap) < top_k:
                    heapq.heappush(heap, item)
                elif item > heap[0]:
                    heapq.heapreplace(heap, item)
            counts["scored"] += len(batch)
    return sorted(heap, reverse=True), counts


def rank(args: argparse.Namespace) -> dict:
    if args.top_k < 1 or args.batch_size < 1 or args.max_pairs < 0:
        raise ValueError("Top-k and batch size must be positive; max-pairs must be non-negative")
    device = choose_device(args.device)
    model, checkpoint = load_checkpoint(args.checkpoint, device)
    seed_everything(checkpoint["seed"])
    for role in ["modifiers", "heads"]:
        if sha256_file(getattr(args, role)) != checkpoint["embedding_sha256"][role]:
            raise ValueError(f"{role} embeddings differ from training; use the original files/basis")
    config = checkpoint["model_config"]
    interval = checkpoint["embedding_config"]
    representation_manifests(args.modifiers, args.heads, interval["cutoff_year"], config["dimensions"])
    modifiers = load_embeddings(args.modifiers, "modifier", config["dimensions"], config["encoder"], **interval)
    heads = load_embeddings(args.heads, "head", config["dimensions"], config["encoder"], **interval)
    excluded = set(read_pairs(args.attested))
    if args.candidates:
        pairs = read_pairs(args.candidates)
        candidate_count = len(pairs)
    else:
        def token_subset(path, embeddings):
            if path is None:
                return embeddings.tokens
            tokens = sorted({line.strip() for line in Path(path).read_text().splitlines() if line.strip()})
            missing = set(tokens) - set(embeddings.index)
            if missing:
                raise ValueError(f"{path}: {len(missing)} tokens have no historical embeddings")
            return tokens
        modifier_tokens = token_subset(args.modifier_list, modifiers)
        head_tokens = token_subset(args.head_list, heads)
        candidate_count = len(modifier_tokens) * len(head_tokens)
        pairs = ((m, h) for m in modifier_tokens for h in head_tokens)
    if args.max_pairs and candidate_count > args.max_pairs:
        raise ValueError(f"Search has {candidate_count:,} pairs, over --max-pairs {args.max_pairs:,}. "
                         "Use --candidates or --modifier-list/--head-list, or explicitly raise --max-pairs.")
    rows, counts = rank_pairs(model, modifiers, heads, pairs, excluded, args.top_k, args.batch_size, device)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["modifier", "head", "score"])
        writer.writerows((m, h, score) for score, m, h in rows)
    result = {"output": str(output), "candidate_pairs": candidate_count, **counts, "returned": len(rows),
              "score_definition": checkpoint["score_definition"]}
    output.with_suffix(output.suffix + ".json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
    return result


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = result.add_subparsers(dest="command", required=True)
    training = commands.add_parser("train", help="Train with unique pair-level train/validation/test splits")
    training.add_argument("--modifiers", required=True)
    training.add_argument("--heads", required=True)
    training.add_argument("--positives", required=True, help="Pairs newly attested in evaluation period")
    training.add_argument("--positive-flag", help="Boolean column selecting positives (e.g. novel)")
    training.add_argument("--attested", required=True, help="ALL attested pairs, historical AND future")
    training.add_argument("--output-dir", required=True)
    training.add_argument("--encoder", choices=["pooled", "lstm"], default="pooled")
    training.add_argument("--dimensions", type=int, default=300)
    training.add_argument("--hidden-size", type=int, default=300)
    training.add_argument("--lstm-hidden-size", type=int, default=300)
    training.add_argument("--start-year", type=int, default=1800)
    training.add_argument("--cutoff-year", type=int, default=1999)
    training.add_argument("--corrupt", choices=["head", "modifier"], default="head")
    training.add_argument("--epochs", type=int, default=50)
    training.add_argument("--batch-size", type=int, default=100)
    training.add_argument("--optimizer", choices=["sgd", "adam"], default="sgd")
    training.add_argument("--learning-rate", type=float, default=0.001)
    training.add_argument("--validation-fraction", type=float, default=0.1)
    training.add_argument("--test-fraction", type=float, default=0.2)
    training.add_argument("--seed", type=int, default=1991)
    training.add_argument("--device", default="auto")
    training.add_argument("--overwrite", action="store_true")
    training.set_defaults(func=train)
    ranking = commands.add_parser("rank", help="Rank unattested pairs with bounded batches and a top-k heap")
    ranking.add_argument("--checkpoint", required=True)
    ranking.add_argument("--modifiers", required=True)
    ranking.add_argument("--heads", required=True)
    ranking.add_argument("--attested", required=True, help="Historical pairs to exclude from generated results")
    ranking.add_argument("--candidates", help="Optional headered table of modifier/head pairs")
    ranking.add_argument("--modifier-list", help="Optional text file of candidate modifiers, one per line")
    ranking.add_argument("--head-list", help="Optional text file of candidate heads, one per line")
    ranking.add_argument("--top-k", type=int, default=100)
    ranking.add_argument("--batch-size", type=int, default=1024)
    ranking.add_argument("--max-pairs", type=int, default=10_000_000, help="Search guard; zero explicitly disables it")
    ranking.add_argument("--device", default="auto")
    ranking.add_argument("--output", required=True)
    ranking.set_defaults(func=rank)
    return result


def main(argv=None):
    arguments = parser().parse_args(argv)
    try:
        arguments.func(arguments)
    except (ValueError, FileNotFoundError) as error:
        raise SystemExit(f"error: {error}") from error


if __name__ == "__main__":
    main()
