"""Sparse numerical features sharing the corrected COHA definitions.

Callers build complete population marginals before selecting evaluated targets.
Matrices here contain the same fixed, tagged context dimensions in each column;
rows are already aligned to compounds or constituents by the caller. Neither raw
occurrence counts nor PPMI backgrounds can be reconstructed from these selected
matrices. No function densifies a target-by-context matrix or mutates its inputs.
"""

import numpy as np
import pandas as pd
from scipy import sparse


STATIC_COLUMNS = (
    "sim_with_modifier", "sim_with_head", "sim_bw_constituents", "beta",
    "geom_mean_sim", "arith_mean_sim", "sim_cpf_0", "sim_cpf_25",
    "sim_cpf_50", "sim_cpf_75", "sim_cpf_100", "sim_cpf_beta",
)
_ENTRY_BATCH = 131_072


def _csr(matrix):
    result = sparse.csr_matrix(matrix, dtype=np.float64, copy=False)
    if not result.has_canonical_format:
        result = result.copy()
        result.sum_duplicates()
        result.sort_indices()
    if not np.isfinite(result.data).all():
        raise ValueError("Feature matrices must contain finite values")
    return result


def _aligned(*matrices):
    result = tuple(_csr(matrix) for matrix in matrices)
    if any(matrix.shape != result[0].shape for matrix in result[1:]):
        raise ValueError("Feature matrices must have identical row/column shapes")
    return result


def _entry_rows(matrix, start, stop):
    # Repeated indptr values are empty rows; right-sided search assigns entries
    # to the correct subsequent nonempty row without a full nnz-sized row array.
    return np.searchsorted(matrix.indptr, np.arange(start, stop), side="right") - 1


def _row_reduce(matrix, transform):
    result = np.zeros(matrix.shape[0], dtype=np.float64)
    for start in range(0, matrix.nnz, _ENTRY_BATCH):
        stop = min(start + _ENTRY_BATCH, matrix.nnz)
        rows = _entry_rows(matrix, start, stop)
        boundaries = np.r_[0, np.flatnonzero(np.diff(rows)) + 1]
        values = transform(matrix.data[start:stop])
        result[rows[boundaries]] += np.add.reduceat(values, boundaries)
    return result


def _norms(matrix):
    return np.sqrt(_row_reduce(matrix, np.square))


def _row_dots(left, right):
    result = np.zeros(left.shape[0], dtype=np.float64)
    pointers = left.indptr.astype(np.int64) + right.indptr
    start = 0
    while start < left.shape[0]:
        stop = int(np.searchsorted(pointers, pointers[start] + _ENTRY_BATCH,
                                   side="right")) - 1
        stop = min(left.shape[0], start + 4096, max(start + 1, stop))
        product = left[start:stop].multiply(right[start:stop])
        result[start:stop] = np.asarray(product.sum(axis=1)).ravel()
        start = stop
    return result


def _cosines(left, right, left_norm, right_norm):
    denominator = left_norm * right_norm
    result = np.full(left.shape[0], np.nan, dtype=np.float64)
    np.divide(_row_dots(left, right), denominator, out=result,
              where=denominator > 0)
    return np.clip(result, -1.0, 1.0)


def weight_matrix(matrix, background, ppmi=False):
    """Return RAW or PPMI weights using full-population context marginals.

    ``background`` is a one-dimensional column-total vector for the COMPLETE
    condition population within this time bin and selected vocabulary. The input
    matrix may contain only evaluated targets; its individual row sums still
    supply complete target marginals. Backgrounds precede cutoff/rating filtering.
    """
    result = _csr(matrix).copy()
    if (result.data < 0).any():
        raise ValueError("Weighting requires nonnegative raw counts")
    background = np.asarray(background, dtype=np.float64)
    if background.ndim != 1 or len(background) != result.shape[1]:
        raise ValueError("Background must provide one total per context column")
    if not np.isfinite(background).all() or (background < 0).any():
        raise ValueError("Background counts must be finite and nonnegative")
    result.eliminate_zeros()
    if not ppmi or result.nnz == 0:
        return result
    total = float(background.sum())
    if total <= 0 or not np.isfinite(total):
        raise ValueError("Positive matrix counts require a positive finite background")
    row_totals = np.asarray(result.sum(axis=1)).ravel()
    log_rows = np.zeros(len(row_totals), dtype=np.float64)
    np.log2(row_totals, out=log_rows, where=row_totals > 0)
    log_background = np.zeros(len(background), dtype=np.float64)
    np.log2(background, out=log_background, where=background > 0)
    log_total = np.log2(total)
    for start in range(0, result.nnz, _ENTRY_BATCH):
        stop = min(start + _ENTRY_BATCH, result.nnz)
        rows = _entry_rows(result, start, stop)
        columns = result.indices[start:stop]
        if np.any(background[columns] <= 0):
            raise ValueError("Positive matrix counts require positive context backgrounds")
        values = result.data[start:stop]
        np.log2(values, out=values)
        values += log_total
        values -= log_rows[rows]
        values -= log_background[columns]
        np.maximum(values, 0.0, out=values)
    result.eliminate_zeros()
    return result


def cosine_rows(left, right):
    """Full-support row cosines: missing/zero -> NaN; disjoint -> zero."""
    left, right = _aligned(left, right)
    return _cosines(left, right, _norms(left), _norms(right))


def _composition_cosines(cm, ch, mh):
    values = tuple(np.asarray(value, dtype=np.float64) for value in (cm, ch, mh))
    if any(value.shape != values[0].shape for value in values[1:]):
        raise ValueError("Composition cosines must have identical shapes")
    if any(np.isinf(value).any() for value in values):
        raise ValueError("Composition cosines must contain finite values or NaN")
    return values


def bounded_beta_from_cosines(cm, ch, mh):
    """Return the [0, 1] HEAD weight for nonnegative RAW/PPMI vectors.

    Inputs are compound-modifier, compound-head and modifier-head cosines with
    identical shapes. The analytic stationary point is clipped to the admissible
    interval, yielding a convex mixture of normalized constituents. Degenerate
    or missing inputs retain the 0.5 default; clipping does not change
    missing-vector policy. NaNs are allowed and inputs are never modified.
    """
    cm, ch, mh = _composition_cosines(cm, ch, mh)
    denominator = (cm + ch) * (1.0 - mh)
    beta = np.full(cm.shape, 0.5, dtype=np.float64)
    np.divide(ch - cm * mh, denominator, out=beta,
              where=np.isfinite(denominator) & (np.abs(denominator) > 1e-12))
    return np.clip(beta, 0.0, 1.0)


def composition_cosine_from_cosines(cm, ch, mh, weight):
    """Cosine to ``weight * unit_head + (1 - weight) * unit_modifier``.

    The three cosine arrays must have identical shapes. ``weight`` can be a
    scalar or have exactly the same shape. Missing cosines/weights and a zero
    composed norm yield NaN, preserving the requirement that both constituent
    vectors exist even at the endpoints. Callers provide the bounded beta when
    computing ``sim_cpf_beta``; this helper does not silently alter weights.
    """
    cm, ch, mh = _composition_cosines(cm, ch, mh)
    weight = np.asarray(weight, dtype=np.float64)
    if weight.ndim != 0 and weight.shape != cm.shape:
        raise ValueError("Composition weight must be scalar or match cosine shapes")
    if np.isinf(weight).any():
        raise ValueError("Composition weight must contain finite values or NaN")
    # Equivalent to w²+(1-w)²+2w(1-w)*mh, but avoids cancellation
    # for identical or nearly identical constituents.
    with np.errstate(invalid="ignore"):
        composed_norm = np.sqrt((1.0 + mh) / 2.0 +
                                (1.0 - mh) / 2.0 * (2.0 * weight - 1.0) ** 2)
    values = np.full(cm.shape, np.nan, dtype=np.float64)
    valid = np.isfinite(cm) & np.isfinite(ch) & np.isfinite(mh)
    np.divide(weight * ch + (1.0 - weight) * cm, composed_norm, out=values,
              where=valid & (composed_norm > 0))
    return np.clip(values, -1.0, 1.0)


def static_features(compounds, modifiers, heads):
    """The twelve COHA cosine/composition features for aligned pair rows.

    Constituent composition uses unit vectors and a HEAD weight. Analytic dot
    products avoid allocating six composed sparse matrices. The analytic beta is
    clipped to [0, 1], so its composition is a convex mixture, not extrapolation.
    Degenerate/missing inputs retain beta=0.5 and missing composition similarities.
    """
    compounds, modifiers, heads = _aligned(compounds, modifiers, heads)
    comp_norm, mod_norm, head_norm = map(_norms, (compounds, modifiers, heads))
    cm = _cosines(compounds, modifiers, comp_norm, mod_norm)
    ch = _cosines(compounds, heads, comp_norm, head_norm)
    mh = _cosines(modifiers, heads, mod_norm, head_norm)
    beta = bounded_beta_from_cosines(cm, ch, mh)
    with np.errstate(invalid="ignore"):
        result = {
            "sim_with_modifier": cm, "sim_with_head": ch,
            "sim_bw_constituents": mh, "beta": beta,
            "geom_mean_sim": np.sqrt(cm * ch), "arith_mean_sim": (cm + ch) / 2.0,
        }
    for suffix, weight in zip(("0", "25", "50", "75", "100", "beta"),
                              (0.0, 0.25, 0.5, 0.75, 1.0, beta)):
        result[f"sim_cpf_{suffix}"] = composition_cosine_from_cosines(cm, ch, mh, weight)
    return pd.DataFrame(result, columns=STATIC_COLUMNS)


def _counts(values, size, name):
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 1 or len(result) != size:
        raise ValueError(f"{name} must provide one occurrence count per row")
    if np.isinf(result).any() or (result < 0).any():
        raise ValueError(f"{name} must contain nonnegative counts or NaN")
    return result


def setting_features(role, general, raw_role, raw_general,
                     role_counts, general_counts, label):
    """Setting-to-Agnostic cosine plus raw occurrence and raw type ratios.

    Raw context types are measured before PPMI. Occurrence counts are supplied
    independently. Empty numerators over observed positive denominators yield
    zero; missing/zero denominators yield NaN. Pooled incidence ratios may exceed
    one and are never capped.
    """
    role, general, raw_role, raw_general = _aligned(role, general, raw_role, raw_general)
    numerator = _counts(role_counts, role.shape[0], "role_counts")
    denominator = _counts(general_counts, role.shape[0], "general_counts")
    numerator = np.where(np.isnan(numerator), 0.0, numerator)
    token = np.full(role.shape[0], np.nan, dtype=np.float64)
    np.divide(numerator, denominator, out=token,
              where=np.isfinite(denominator) & (denominator > 0))
    role_types = _row_reduce(raw_role, lambda values: values > 0)
    general_types = _row_reduce(raw_general, lambda values: values > 0)
    types = np.full(role.shape[0], np.nan, dtype=np.float64)
    np.divide(role_types, general_types, out=types, where=general_types > 0)
    token_label = "mod" if label == "modifier" else label
    return pd.DataFrame({
        f"sim_bw_settings_{label}": cosine_rows(role, general),
        f"perc_token_{token_label}": token,
        f"perc_type_{token_label}": types,
    })


def temporal_cosines(current, previous):
    """Adjacent-bin cosine; callers supply only exact eligible preceding bins."""
    return cosine_rows(current, previous)
