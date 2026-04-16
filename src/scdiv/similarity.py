"""Similarity computation helpers for dense and sparse matrices."""

import numpy as np
import numpy.typing as npt
import scipy.sparse


def _to_dense(x: npt.NDArray | scipy.sparse.sparray) -> npt.NDArray:
    if scipy.sparse.issparse(x):
        return np.asarray(x.todense()) # type: ignore[union-attr]`
    return np.asarray(x, dtype=float)


def l2_normalize_rows(x: npt.NDArray) -> npt.NDArray:
    """L2-normalize each row. Rows with zero norm are left as zeros."""
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return x / norms


def cosine_similarity_matrix(x: npt.NDArray) -> npt.NDArray:
    """Compute the cosine similarity matrix from row vectors.

    Args:
        x: Matrix of shape (n, d).

    Returns:
        Cosine similarity matrix of shape (n, n) with values in [-1, 1].

    """
    x_norm = l2_normalize_rows(x)
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
        cell_types: Unique cell types to compute means for.

    Returns:
        Mean expression per type, shape (n_types, n_genes).

    """
    means = np.empty((len(cell_types), x.shape[1]))
    for i, ct in enumerate(cell_types):
        means[i] = x[labels == ct].mean(axis=0)
    return means


def cell_type_similarity(
    x: npt.NDArray | scipy.sparse.sparray,
    labels: npt.NDArray,
) -> tuple[npt.NDArray, npt.NDArray]:
    """Compute cosine similarity matrix between cell types.

    Pipeline: compute mean expression per type, then cosine similarity
    between the mean vectors.

    Args:
        x: Expression matrix, shape (n_cells, n_genes). Can be sparse.
        labels: Cell type label for each cell, shape (n_cells,).

    Returns:
        (similarity_matrix, cell_types) where similarity_matrix has
        shape (n_types, n_types) and cell_types is a sorted array of
        unique labels.

    """
    cell_types = np.unique(labels)
    x_dense = _to_dense(x)
    means = _mean_expression_per_type(x_dense, labels, cell_types)
    return cosine_similarity_matrix(means), cell_types
