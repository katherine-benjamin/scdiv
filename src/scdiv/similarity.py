"""Similarity computation helpers for dense and sparse matrices."""

import numpy as np
import numpy.typing as npt
import scipy.sparse


def _to_dense(x: npt.NDArray | scipy.sparse.sparray) -> npt.NDArray:
    if scipy.sparse.issparse(x):
        return np.asarray(x.todense()) # type: ignore[union-attr]`
    return np.asarray(x, dtype=float)


def normalize_columns(x: npt.NDArray | scipy.sparse.sparray) -> npt.NDArray:
    """L1-normalize each column (gene) of an expression matrix.

    Divides each gene's expression by its total across all cells, so that
    all genes contribute equally regardless of abundance. Returns a dense
    array. Columns with zero total are left as zeros.
    """
    x = _to_dense(x)
    col_sums = np.abs(x).sum(axis=0, keepdims=True)
    col_sums[col_sums == 0] = 1
    return x / col_sums


def _l2_normalize_rows(x: npt.NDArray) -> npt.NDArray:
    """L2-normalize each row. Rows with zero norm are left as zeros."""
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return x / norms


def cosine_similarity_matrix(x: npt.NDArray) -> npt.NDArray:
    """Compute the cosine similarity matrix from row vectors.

    Intended for small matrices (e.g. one row per cell type).

    Args:
        x: Matrix of shape (n, d) where each row is a vector.

    Returns:
        Cosine similarity matrix of shape (n, n) with values in [-1, 1].

    """
    x_norm = _l2_normalize_rows(x)
    return x_norm @ x_norm.T


def weighted_cosine_similarities(
    x_norm: npt.NDArray, distribution: npt.NDArray
) -> npt.NDArray:
    """Compute S @ p without materializing S, where S is cosine similarity.

    Uses the identity: S @ p = X_norm @ (X_norm.T @ p)
    where X_norm has L2-normalized rows. This is O(n*d) instead of O(n^2).

    Args:
        x_norm: L2-row-normalized matrix, shape (n, d).
        distribution: Weight vector, shape (n,).

    Returns:
        Vector of weighted similarities, shape (n,).

    """
    return x_norm @ (x_norm.T @ distribution)
