"""Similarity-sensitive diversity measures for transcriptomics data."""

import numpy as np
import numpy.typing as npt
import scipy.stats

import scdiv.similarity


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

    return scipy.stats.pmean(
        1 / weighted_similarities, 1 - order, weights=distribution
    )


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


def diversity_from_counts(
    x: npt.NDArray, labels: npt.NDArray, order: float
) -> float:
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
