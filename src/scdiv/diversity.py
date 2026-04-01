"""Similarity-sensitive diversity measures for transcriptomics data."""

import numpy as np
import numpy.typing as npt
import scipy.stats


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


def distance_to_similarity(distance: npt.NDArray) -> npt.NDArray:
    """Convert a distance matrix to a similarity matrix."""
    return np.exp(-distance)
