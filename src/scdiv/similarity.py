"""Similarity computation helpers for dense and sparse matrices."""

import numpy as np
import numpy.typing as npt
import scipy.sparse


def _to_dense(x: npt.NDArray | scipy.sparse.sparray) -> npt.NDArray:
    if scipy.sparse.issparse(x):
        return np.asarray(x.todense())  # ty: ignore[unresolved-attribute]`
    return np.asarray(x, dtype=float)


def l2_normalize_rows(x: npt.NDArray) -> npt.NDArray:
    """L2-normalize each row. Rows with zero norm are left as zeros."""
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return x / norms


def feature_transform(x: npt.NDArray, alpha: float = 1.0) -> npt.NDArray:
    """Map non-negative rows to L2-normalized probability-geometric features.

    Each row is raised elementwise to the power ``alpha``, then
    L2-normalized. Feeding the result into the cosine matvec yields the
    probability-geometric similarity family:

    - ``alpha=1``: plain cosine similarity of the expression vectors.
    - ``alpha=0.5``: Bhattacharyya (Hellinger) similarity.
    - smaller ``alpha``: progressively down-weights highly expressed genes,
      so a handful of high-count genes no longer dominate the similarity.

    Args:
        x: Non-negative matrix of shape (n, d).
        alpha: Exponent of the probability-geometric family.

    Returns:
        L2-row-normalized features, shape (n, d).

    """
    if alpha == 1.0:
        return l2_normalize_rows(x)
    return l2_normalize_rows(x**alpha)


def cosine_similarity_matrix(x: npt.NDArray, alpha: float = 1.0) -> npt.NDArray:
    """Compute the probability-geometric similarity matrix from row vectors.

    Args:
        x: Matrix of shape (n, d).
        alpha: Exponent of the probability-geometric family (see
            :func:`feature_transform`). ``alpha=1`` is cosine similarity.

    Returns:
        Similarity matrix of shape (n, n) with values in [-1, 1].

    """
    x_norm = feature_transform(x, alpha)
    return x_norm @ x_norm.T


def weighted_cosine_similarities(
    x_norm: npt.NDArray, distribution: npt.NDArray
) -> npt.NDArray:
    """Compute S @ p without materializing S, where S is cosine similarity.

    Uses the identity: S @ p = X_norm @ (X_norm.T @ p) where X_norm has
    L2-normalized rows.

    Args:
        x_norm: L2-row-normalized matrix, shape (n, d).
        distribution: Weight vector, shape (n,).

    Returns:
        Vector of weighted similarities, shape (n,).

    """
    return x_norm @ (x_norm.T @ distribution)


def _mean_expression_per_type(
    x: npt.NDArray, labels: npt.NDArray, cell_types: npt.NDArray
) -> npt.NDArray:
    """Compute mean expression vector for each cell type.

    Args:
        x: Expression matrix, shape (n_cells, n_genes).
        labels: Cell type label for each cell, shape (n_cells,).
        cell_types: Unique cell types to compute means for. Must be
            sorted and contain every label (true when
            ``cell_types = np.unique(labels)``).

    Returns:
        Mean expression per type, shape (n_types, n_genes).

    """
    n_cells = len(labels)
    n_types = len(cell_types)
    row_idx = np.searchsorted(cell_types, labels)
    counts = np.bincount(row_idx, minlength=n_types)
    weights = 1.0 / counts[row_idx]
    indicator = scipy.sparse.csr_matrix(
        (weights, (row_idx, np.arange(n_cells))),
        shape=(n_types, n_cells),
    )
    return indicator @ x


def cell_type_similarity(
    x: npt.NDArray | scipy.sparse.sparray,
    labels: npt.NDArray,
    alpha: float = 1.0,
) -> tuple[npt.NDArray, npt.NDArray]:
    """Compute the similarity matrix between cell types.

    Pipeline: compute mean expression per type, then probability-geometric
    similarity between the mean vectors.

    Args:
        x: Expression matrix, shape (n_cells, n_genes). Can be sparse.
        labels: Cell type label for each cell, shape (n_cells,).
        alpha: Exponent of the probability-geometric family (see
            :func:`feature_transform`). ``alpha=1`` is cosine similarity.

    Returns:
        (similarity_matrix, cell_types) where similarity_matrix has
        shape (n_types, n_types) and cell_types is a sorted array of
        unique labels.

    """
    cell_types = np.unique(labels)
    x_dense = _to_dense(x)
    means = _mean_expression_per_type(x_dense, labels, cell_types)
    return cosine_similarity_matrix(means, alpha), cell_types
