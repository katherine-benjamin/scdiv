import hypothesis.strategies as st
import numpy as np
import numpy.testing as npt
import pytest
import scipy.stats
from hypothesis import given
from hypothesis.extra.numpy import arrays

import scdiv.diversity

MAX_VALUE = np.iinfo("int64").max
RTOL = 1e-06

dists = st.lists(
    st.integers(min_value=1, max_value=MAX_VALUE), min_size=1, max_size=100
)

orders = st.floats(min_value=0, max_value=1000, allow_nan=False)


def freq_to_dist(freq):
    freq = np.array(freq, dtype="d")
    return freq / freq.sum()


@st.composite
def similarity_matrices(draw):
    n = draw(st.integers(min_value=1, max_value=10))
    m = draw(arrays("d", (n, n), elements=st.floats(0, 1), fill=st.nothing()))
    for i in range(n):
        m[i, i] = 1
    return m


@st.composite
def similarities_and_dists(draw):
    n = draw(st.integers(min_value=1, max_value=10))
    m = draw(arrays("d", (n, n), elements=st.floats(0, 1), fill=st.nothing()))
    for i in range(n):
        m[i, i] = 1
    dist = draw(
        st.lists(st.integers(min_value=1, max_value=MAX_VALUE), min_size=n, max_size=n)
    )

    return m, dist


@given(dists, orders)
def test_all_similar_gives_one(dist, p):
    dist = freq_to_dist(dist)
    num_types = len(dist)
    similarity = np.ones((num_types, num_types))
    diversity = scdiv.diversity.diversity(similarity, p, dist)
    npt.assert_allclose(diversity, 1.0, rtol=RTOL)


@given(dists, orders)
def test_naive_gives_hill_number(dist, p):
    dist = freq_to_dist(dist)
    num_types = len(dist)
    similarity = np.identity(num_types)

    div = scdiv.diversity.diversity(similarity, p, dist)

    hill = scipy.stats.pmean(1 / dist, 1 - p, weights=dist)

    npt.assert_allclose(div, hill, rtol=RTOL)


@given(dists)
def test_order_infinity_naive(dist):
    dist = freq_to_dist(dist)
    num_types = len(dist)
    similarity = np.identity(num_types)

    div = scdiv.diversity.diversity(similarity, np.inf, dist)

    npt.assert_allclose(div, 1 / np.max(dist), rtol=RTOL)


@given(similarities_and_dists())
def test_order_two(sim_and_dist):
    similarity, dist = sim_and_dist
    dist = freq_to_dist(dist)

    div = scdiv.diversity.diversity(similarity, 2, dist)

    npt.assert_allclose(div, 1 / (dist.T @ (similarity @ dist)), rtol=RTOL)


@given(similarities_and_dists(), orders, orders)
def test_decreasing_in_order(sim_and_dist, order1, order2):
    similarity, dist = sim_and_dist
    dist = freq_to_dist(dist)

    div1 = scdiv.diversity.diversity(similarity, order1, dist)
    div2 = scdiv.diversity.diversity(similarity, order2, dist)

    if order1 <= order2:
        assert div2 <= div1
    else:
        assert div1 <= div2


def test_empty_similarity_raises():
    with pytest.raises(ValueError, match="empty"):
        scdiv.diversity.diversity(np.empty((0, 0)), 1)


@given(similarities_and_dists(), orders)
def test_range_of_diversities(sim_and_dist, order):
    similarity, dist = sim_and_dist
    dist = freq_to_dist(dist)

    div = scdiv.diversity.diversity(similarity, order, dist)

    assert 1 <= div <= len(dist) * (1 + RTOL)
