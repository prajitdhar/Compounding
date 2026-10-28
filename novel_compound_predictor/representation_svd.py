"""Row-bounded randomized right-singular subspace fitting for shared embeddings.

All compound, modifier and head rows participate in the same SVD objective.
Power iteration applies X.T @ X in sparse row batches; no rows-by-components
array is allocated. Float64 accumulation limits loss of precision from the
normal-operator products. This is a different randomized solver from sklearn's
range finder: the fitted approximation and numeric coordinates can differ.
The seed, iterations, oversampling and solver must accompany exported vectors.
"""

from contextlib import nullcontext
from dataclasses import dataclass


def _size(value, name, minimum=0):
    if int(value) != value or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    if value > 2**63 - 1:
        raise ValueError(f"{name} exceeds signed 64-bit sparse index capacity")
    return int(value)


def estimate_csr_memory(n_rows, n_contexts, nnz, batch_size=10000):
    """CSR bytes and construction peak, including the builder's int64 row counts.

    The 64-bit switch includes nnz, not just shape: CSR indptr stores cumulative
    offsets. The Arrow conversion allowance is bounded by batch_size. DuckDB's
    allocation and the SVD working set are separate from these values.
    """
    n_rows = _size(n_rows, "n_rows", 1)
    n_contexts = _size(n_contexts, "n_contexts", 1)
    nnz = _size(nnz, "nnz")
    batch_size = _size(batch_size, "batch_size", 1)
    index_bytes = 8 if max(n_rows, n_contexts, nnz) >= 2**31 else 4
    csr_bytes = nnz * (4 + index_bytes) + (n_rows + 1) * index_bytes
    row_lengths_bytes = n_rows * 8
    conversion_bytes = batch_size * 64
    return {"rows": n_rows, "contexts": n_contexts, "nonzeros": nnz,
            "data_dtype": "float32", "sparse_index_bits": index_bytes * 8,
            "csr_bytes": csr_bytes, "row_lengths_bytes": row_lengths_bytes,
            "batch_conversion_allowance_bytes": conversion_bytes,
            "estimated_construction_peak_bytes": csr_bytes + row_lengths_bytes + conversion_bytes}


def estimate_svd_workspace(n_contexts, k, batch_size=10000, max_block_nnz=None,
                           oversamples=10, sparse_index_bits=64, details=False):
    """Conservative solver workspace bytes, excluding the resident CSR/DuckDB.

    Six feature-by-sketch buffers allow for accumulation and LAPACK QR copies.
    If supplied, max_block_nnz allows the previous float64 block, a new float32
    CSR row-slice, and its float64 moment copy to coexist during assignment.
    This estimate is independent of the corpus row count; it is not a strict
    allocator bound for every BLAS implementation.
    """
    n_contexts = _size(n_contexts, "n_contexts", 1)
    k = _size(k, "k", 1)
    batch_size = _size(batch_size, "batch_size", 1)
    oversamples = _size(oversamples, "oversamples")
    if k > n_contexts:
        raise ValueError("k exceeds the number of contexts")
    if sparse_index_bits not in {32, 64}:
        raise ValueError("sparse_index_bits must be 32 or 64")
    width = min(n_contexts, k + oversamples)
    feature_buffers = 6 * n_contexts * width * 8
    batch_buffers = 4 * batch_size * width * 8
    small_buffers = 6 * width * width * 8
    moment_buffers = 8 * n_contexts * 8
    sparse_buffers = 0
    if max_block_nnz is not None:
        max_block_nnz = _size(max_block_nnz, "max_block_nnz")
        # scipy astype(float64) copies indices/indptr as well as data; Python
        # keeps the previous block alive while evaluating the next assignment.
        sparse_buffers = max_block_nnz * (4 + 8 + 8 + 3 * (sparse_index_bits // 8))
        sparse_buffers += 3 * (batch_size + 1) * (sparse_index_bits // 8)
    plan = {"solver": "streaming_gram_subspace", "accumulation_dtype": "float64",
            "sketch_columns": width, "batch_rows": batch_size,
            "feature_buffers_bytes": feature_buffers, "batch_buffers_bytes": batch_buffers,
            "small_buffers_bytes": small_buffers, "moment_buffers_bytes": moment_buffers,
            "sparse_block_allowance_bytes": sparse_buffers,
            "estimated_workspace_bytes": feature_buffers + batch_buffers + small_buffers
                                         + moment_buffers + sparse_buffers}
    return plan if details else plan["estimated_workspace_bytes"]


@dataclass
class StreamingSVD:
    components_: object
    singular_values_: object
    explained_variance_: object
    explained_variance_ratio_: object
    n_features_in_: int
    solver: str = "streaming_gram_subspace"
    n_iter: int = 10
    random_state: int = 1991
    n_oversamples: int = 10

    def transform(self, matrix):
        import numpy as np
        if len(matrix.shape) != 2 or matrix.shape[1] != self.n_features_in_:
            raise ValueError(f"Expected {self.n_features_in_} context features")
        return np.asarray(matrix @ self.components_.T)


def _merge_moments(count, mean, m2, length, batch_mean, batch_variance):
    delta = batch_mean - mean
    total = count + length
    m2 += batch_variance * length + delta**2 * count * length / total
    mean += delta * length / total
    return total


def fit_streaming_svd(matrix, k, *, n_iter=10, seed=1991, batch_size=10000,
                      oversamples=10, reporter=None):
    """Fit one approximate truncated SVD with row-bounded float64 passes.

    reporter may be a ProgressReporter object (stages and stored batch updates)
    or a callback receiving ``reporter(event, **metadata)`` at pass boundaries. The
    operator X.T @ X squares singular values; float64 and QR at every pass help
    stability, but very weak directions can be less accurate than direct SVD.
    All arrays indexed by corpus rows have at most batch_size rows. Randomness
    is over context features only; batches never fit separate SVD bases.
    """
    import numpy as np
    from scipy import sparse
    from sklearn.utils.sparsefuncs import mean_variance_axis

    if not sparse.isspmatrix_csr(matrix) and not isinstance(matrix, sparse.csr_array):
        raise ValueError("Streaming shared SVD requires a CSR matrix or CSR array")
    k = _size(k, "k", 1)
    n_iter = _size(n_iter, "n_iter", 1)
    batch_size = _size(batch_size, "batch_size", 1)
    oversamples = _size(oversamples, "oversamples")
    if k > min(matrix.shape):
        raise ValueError("Requested dimensions exceed the matrix shape")
    n_rows, n_features = matrix.shape
    width = min(n_features, k + oversamples)

    def report(event, **metadata):
        if reporter is not None:
            if callable(reporter):
                reporter(event, **metadata)
            else:
                reporter.log(event, **metadata)

    def stage(name):
        return (reporter.stage(name, total=n_rows)
                if reporter is not None and hasattr(reporter, "stage") else nullcontext())

    def update(start, detail):
        if reporter is not None and hasattr(reporter, "update"):
            reporter.update(detail, completed=min(start + batch_size, n_rows), total=n_rows)

    report("svd_initialize", rows=n_rows, contexts=n_features, dimensions=k,
           sketch_columns=width, n_iter=n_iter, batch_rows=batch_size,
           solver="streaming_gram_subspace", accumulation_dtype="float64")
    random = np.random.default_rng(seed)
    q, _ = np.linalg.qr(random.standard_normal((n_features, width)), mode="reduced")
    for iteration in range(n_iter):
        report("svd_power_pass_start", pass_number=iteration + 1, total_passes=n_iter)
        gram_projection = np.zeros((n_features, width), dtype=np.float64)
        with stage(f"SVD power pass {iteration + 1}/{n_iter}"):
            for start in range(0, n_rows, batch_size):
                block = matrix[start:start + batch_size]
                if iteration == 0 and not np.isfinite(block.data).all():
                    raise ValueError("Non-finite sparse count/weight in the shared SVD matrix")
                gram_projection += block.T @ (block @ q)
                update(start, "accumulated context subspace")
            if not np.isfinite(gram_projection).all():
                raise ValueError("Non-finite Gram projection; input weights overflow float64 accumulation")
            q, _ = np.linalg.qr(gram_projection, mode="reduced")
        del gram_projection
        report("svd_power_pass_complete", pass_number=iteration + 1, total_passes=n_iter)

    report("svd_rayleigh_pass_start")
    small_gram = np.zeros((width, width), dtype=np.float64)
    with stage("SVD projected Gram pass"):
        for start in range(0, n_rows, batch_size):
            projected = matrix[start:start + batch_size] @ q
            small_gram += projected.T @ projected
            update(start, "accumulated projected Gram matrix")
        if not np.isfinite(small_gram).all():
            raise ValueError("Non-finite projected Gram matrix; input weights overflow float64 accumulation")
    eigenvalues, eigenvectors = np.linalg.eigh((small_gram + small_gram.T) * 0.5)
    order = np.argsort(eigenvalues)[::-1][:k]
    components = (q @ eigenvectors[:, order]).T
    singular_values = np.sqrt(np.maximum(eigenvalues[order], 0))
    # Use the same right-vector sign convention as TruncatedSVD.
    signs = np.sign(components[np.arange(k), np.abs(components).argmax(axis=1)])
    signs[signs == 0] = 1
    components *= signs[:, None]
    del q, small_gram, projected, eigenvectors
    report("svd_rayleigh_pass_complete")
    model = StreamingSVD(components, singular_values, None, None, n_features,
                         n_iter=n_iter, random_state=seed, n_oversamples=oversamples)

    report("svd_variance_pass_start")
    total = 0
    mean = np.zeros(k, dtype=np.float64)
    m2 = np.zeros(k, dtype=np.float64)
    feature_mean = np.zeros(n_features, dtype=np.float64)
    feature_m2 = np.zeros(n_features, dtype=np.float64)
    with stage("SVD explained variance pass"):
        for start in range(0, n_rows, batch_size):
            block = matrix[start:start + batch_size].astype(np.float64, copy=False)
            transformed = model.transform(block)
            length = block.shape[0]
            batch_feature_mean, batch_feature_variance = mean_variance_axis(block, axis=0)
            _merge_moments(total, feature_mean, feature_m2, length,
                           batch_feature_mean, batch_feature_variance)
            total = _merge_moments(total, mean, m2, length,
                                   transformed.mean(axis=0), transformed.var(axis=0))
            update(start, "merged projection and sparse-feature moments")
    model.explained_variance_ = m2 / total
    full_variance = feature_m2.sum() / total
    model.explained_variance_ratio_ = (model.explained_variance_ / full_variance
                                     if full_variance > 0 else np.zeros(k, dtype=np.float64))
    report("svd_variance_pass_complete", rows=total,
           explained_variance_ratio_sum=float(model.explained_variance_ratio_.sum()))
    return model
