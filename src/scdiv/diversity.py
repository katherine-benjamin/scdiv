"""Similarity-sensitive diversity measures for transcriptomics data."""

from typing import Literal

import numpy as np
import numpy.typing as npt
import scipy.stats

import scdiv.similarity

Mode = Literal["alpha", "alpha_norm", "gamma"]
_MODES: tuple[Mode, ...] = ("alpha", "alpha_norm", "gamma")


def diversity_from_weighted_similarities(
    weighted_similarities: npt.NDArray,
    order: float,
    distribution: npt.NDArray,
) -> float:
    """Compute diversity from pre-computed weighted similarities.

    Args:
        weighted_similarities:
            The vector S @ p, where S is the similarity matrix and p is the
            distribution. Shape (n,).
        order:
            The order of the power mean used to average the diversity.
        distribution:
            The relative abundances. Shape (n,).

    Returns:
        The diversity of the data set.

    """
    if np.isposinf(order):
        return 1 / weighted_similarities.max()

    if np.isneginf(order):
        return 1 / weighted_similarities.min()

    return scipy.stats.pmean(1 / weighted_similarities, 1 - order, weights=distribution)


def diversity(
    similarity: npt.NDArray, order: float, distribution: None | npt.NDArray = None
) -> float:
    """Return the diversity of a single-cell data set.

    Args:
        similarity:
            The similarity matrix of the data set.
        order:
            The order of the diversity.
        distribution:
            The relative abundances of each sample. If None then a uniform
            distribution is assumed.

    Returns:
        The similarity-sensitive diversity of the given order.

    """
    num_species = len(similarity)

    if num_species <= 0:
        msg = "Similarity matrix should not be empty."
        raise ValueError(msg)

    if distribution is None:
        distribution = np.ones(num_species) / num_species

    weighted_similarities = similarity @ distribution
    return diversity_from_weighted_similarities(
        weighted_similarities, order, distribution
    )


def distribution_from_labels(
    labels: npt.NDArray,
) -> tuple[npt.NDArray, npt.NDArray]:
    """Compute relative abundance of each type from labels.

    Args:
        labels: Type label for each observation, shape (n,).

    Returns:
        (distribution, cell_types) where distribution sums to 1 and
        cell_types is a sorted array of unique labels.

    """
    cell_types, counts = np.unique(labels, return_counts=True)
    return counts / counts.sum(), cell_types


def _power_mean(values: npt.NDArray, weights: npt.NDArray, order: float) -> float:
    """Weighted power mean of ``values`` of order ``1 - order``.

    Used to aggregate per-subcommunity diversities into a single
    metacommunity-level scalar (Reeve et al. 2016).
    """
    if np.isposinf(order):
        return float(values.min())
    if np.isneginf(order):
        return float(values.max())
    return float(scipy.stats.pmean(values, 1 - order, weights=weights))


def partition_diversity(  # noqa: PLR0913
    similarity: npt.NDArray,
    distributions: npt.NDArray,
    weights: npt.NDArray,
    order: float,
    *,
    mode: Mode = "alpha_norm",
    aggregate: bool = False,
) -> tuple[npt.NDArray, float | None]:
    """Per-subcommunity diversity in the style of Reeve et al. (2016).

    Args:
        similarity:
            Similarity matrix Z, shape (M, M).
        distributions:
            Within-subcommunity proportions P^(j) as columns, shape (M, N).
            Each column must sum to 1.
        weights:
            Subcommunity weights w_j, shape (N,), summing to 1.
        order:
            Order q of the diversity.
        mode:
            One of:
              - ``"alpha_norm"``: standalone Leinster-Cobbold diversity of
                each subcommunity, ``D_q(P^(j); Z P^(j))``.
              - ``"alpha"``: ``alpha_norm / w_j`` for each subcommunity.
              - ``"gamma"``: ``D_q(P^(j); Z p_pooled)``, where
                ``p_pooled = distributions @ weights``.
        aggregate:
            If True, also return the metacommunity-level scalar (the
            ``w_j``-weighted power mean of order ``1 - q`` of the per-group
            values; for ``gamma`` this equals the diversity of the pooled
            distribution).

    Returns:
        ``(per_group, meta)`` where ``per_group`` has shape (N,) and
        ``meta`` is the aggregate scalar or None.

    """
    if mode not in _MODES:
        msg = f"mode must be one of {_MODES}, got {mode!r}."
        raise ValueError(msg)

    n_groups = distributions.shape[1]
    pooled = distributions @ weights

    if mode == "gamma":
        zp_pooled = similarity @ pooled
        per_group = np.array(
            [
                diversity_from_weighted_similarities(
                    zp_pooled, order, distributions[:, j]
                )
                for j in range(n_groups)
            ]
        )
        meta = (
            diversity_from_weighted_similarities(zp_pooled, order, pooled)
            if aggregate
            else None
        )
        return per_group, meta

    zp = similarity @ distributions
    alpha_norm = np.array(
        [
            diversity_from_weighted_similarities(zp[:, j], order, distributions[:, j])
            for j in range(n_groups)
        ]
    )
    per_group = alpha_norm / weights if mode == "alpha" else alpha_norm
    meta = _power_mean(per_group, weights, order) if aggregate else None
    return per_group, meta


def partition_diversity_singleton(
    x_norm: npt.NDArray,
    group_indices: npt.NDArray,
    order: float,
    *,
    mode: Mode = "alpha_norm",
    aggregate: bool = False,
) -> tuple[npt.NDArray, float | None]:
    """Reeve partition diversity in singleton mode (each row is its own type).

    Uses the factored cosine identity ``S @ p = X_norm @ (X_norm.T @ p)`` to
    avoid materialising the n x n cell-cell similarity matrix. Same
    semantics as :func:`partition_diversity` for the ``mode``/``aggregate``
    arguments.

    Args:
        x_norm:
            L2-row-normalized matrix, shape ``(n_total, d)``. Each row
            represents one "species".
        group_indices:
            Integer subcommunity index in ``[0, N)`` for each row of
            ``x_norm``. ``N = group_indices.max() + 1``.
        order:
            Order q of the diversity.
        mode:
            One of:
                - ``"alpha_norm"``: standalone Leinster-Cobbold diversity of
                  each subcommunity, ``D_q(P^(j); Z P^(j))``.
                - ``"alpha"``: ``alpha_norm / w_j`` for each subcommunity.
                - ``"gamma"``: ``D_q(P^(j); Z p_pooled)``, where
                  ``p_pooled = distributions @ weights``.
        aggregate:
            If True, also return the metacommunity-level scalar (the
            ``w_j``-weighted power mean of order ``1 - q`` of the per-group
             values; for ``gamma`` this equals the diversity of the pooled
            distribution).

    Returns:
        ``(per_group, meta)`` where ``per_group`` has shape ``(N,)`` and
        ``meta`` is the aggregate scalar or ``None``.

    """
    if mode not in _MODES:
        msg = f"mode must be one of {_MODES}, got {mode!r}."
        raise ValueError(msg)

    n_total = x_norm.shape[0]
    n_groups = int(group_indices.max()) + 1
    p_pooled = np.ones(n_total) / n_total

    if mode == "gamma":
        zp_pooled = scdiv.similarity.weighted_cosine_similarities(x_norm, p_pooled)

    per_sub = np.empty(n_groups)
    weights = np.empty(n_groups)
    for j in range(n_groups):
        within = group_indices == j
        n_group = int(within.sum())
        weights[j] = n_group / n_total
        p_j = np.ones(n_group) / n_group
        if mode == "gamma":
            zp_j = zp_pooled[within]
        else:
            zp_j = scdiv.similarity.weighted_cosine_similarities(x_norm[within], p_j)
        per_sub[j] = diversity_from_weighted_similarities(zp_j, order, p_j)

    if mode == "gamma":
        meta = (
            diversity_from_weighted_similarities(zp_pooled, order, p_pooled)
            if aggregate
            else None
        )
        return per_sub, meta

    per_group = per_sub / weights if mode == "alpha" else per_sub
    meta = _power_mean(per_group, weights, order) if aggregate else None
    return per_group, meta


def diversity_from_counts(x: npt.NDArray, labels: npt.NDArray, order: float) -> float:
    """Compute diversity directly from a count matrix and cell type labels.

    Args:
        x: Expression matrix, shape (n_cells, n_genes). Can be sparse.
        labels: Cell type label for each cell, shape (n_cells,).
        order: The order of the diversity.

    Returns:
        The similarity-sensitive diversity.

    """
    sim, _ = scdiv.similarity.cell_type_similarity(x, labels)
    dist, _ = distribution_from_labels(labels)
    return diversity(sim, order, dist)
