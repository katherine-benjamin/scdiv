"""Similarity-sensitive diversity measures for transcriptomics data."""

import numpy as np
import numpy.typing as npt
import scipy.stats


def diversity(
    similarity: npt.NDArray, order: float, distribution: None | npt.NDArray = None
) -> float:
    """Return the diversity of a single-cell data set.

    Args:
        similarity:
            The similarity matrix of the data set.
        order:
            The order of the power mean used to average the diversity over all
            samples.
        distribution:
            The relative abundances of each sample. If None then a uniform
            distribution is assumed. Should be None if the samples are single
            cells rather than cell clusters.

    Returns:
        The diversity of the data set.

    """
    num_species = len(similarity)

    if num_species <= 0:
        msg = "Similarity matrix should not be empty."
        raise ValueError(msg)

    if distribution is None:
        distribution = np.ones(num_species) / num_species

    if np.isposinf(order):
        return 1 / (similarity @ distribution).max()

    if np.isneginf(order):
        return 1 / (similarity @ distribution).min()

    return scipy.stats.pmean(
        1 / (similarity @ distribution), 1 - order, weights=distribution
    )


def distance_to_similarity(distance: npt.NDArray) -> npt.NDArray:
    """Convert a distance matrix to a similarity matrix."""
    return np.exp(-distance)
