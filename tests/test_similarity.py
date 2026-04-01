import hypothesis.strategies as st
import numpy as np
import numpy.testing as npt
import scipy.sparse
from hypothesis import given
from hypothesis.extra.numpy import arrays

from scdiv.similarity import (
    _l2_normalize_rows,
    cosine_similarity_matrix,
    normalize_columns,
    weighted_cosine_similarities,
)

expression_matrices = st.integers(min_value=1, max_value=10).flatmap(
    lambda n: st.integers(min_value=1, max_value=10).flatmap(
        lambda d: arrays(
            "d",
            (n, d),
            elements=st.floats(0, 100, allow_nan=False, allow_infinity=False),
            fill=st.nothing(),
        )
    )
)


@given(expression_matrices)
def test_normalize_columns_sums_to_one_or_zero(x):
    result = normalize_columns(x)
    col_sums = result.sum(axis=0)
    for s, orig_sum in zip(col_sums, np.abs(x).sum(axis=0), strict=True):
        if orig_sum == 0:
            npt.assert_allclose(s, 0.0, atol=1e-10)
        else:
            npt.assert_allclose(s, 1.0, rtol=1e-6)


@given(expression_matrices)
def test_normalize_columns_preserves_nonnegativity(x):
    result = normalize_columns(x)
    assert np.all(result >= 0)


@given(expression_matrices)
def test_normalize_columns_sparse_matches_dense(x):
    dense_result = normalize_columns(x)
    sparse_result = normalize_columns(scipy.sparse.csr_matrix(x))
    npt.assert_allclose(sparse_result, dense_result, rtol=1e-10)


@given(expression_matrices)
def test_l2_normalize_rows_unit_norm_or_zero(x):
    result = _l2_normalize_rows(x)
    norms = np.linalg.norm(result, axis=1)
    for norm, orig_norm in zip(norms, np.linalg.norm(x, axis=1), strict=True):
        if orig_norm == 0:
            npt.assert_allclose(norm, 0.0, atol=1e-10)
        else:
            npt.assert_allclose(norm, 1.0, rtol=1e-6)


@given(expression_matrices)
def test_cosine_similarity_diagonal_is_one_or_zero(x):
    sim = cosine_similarity_matrix(x)
    for i in range(sim.shape[0]):
        row_norm = np.linalg.norm(x[i])
        if row_norm == 0:
            npt.assert_allclose(sim[i, i], 0.0, atol=1e-10)
        else:
            npt.assert_allclose(sim[i, i], 1.0, rtol=1e-6)


@given(expression_matrices)
def test_cosine_similarity_is_symmetric(x):
    sim = cosine_similarity_matrix(x)
    npt.assert_allclose(sim, sim.T, rtol=1e-10)


@given(expression_matrices)
def test_cosine_similarity_nonneg_for_nonneg_input(x):
    sim = cosine_similarity_matrix(x)
    atol = 1e-10
    assert np.all(sim >= -atol)


@st.composite
def matrices_and_distributions(draw):
    n = draw(st.integers(min_value=1, max_value=10))
    d = draw(st.integers(min_value=1, max_value=10))
    x = draw(
        arrays(
            "d",
            (n, d),
            elements=st.floats(0, 100, allow_nan=False, allow_infinity=False),
            fill=st.nothing(),
        )
    )
    freq = draw(
        st.lists(
            st.integers(min_value=1, max_value=1000), min_size=n, max_size=n
        )
    )
    freq = np.array(freq, dtype=float)
    p = freq / freq.sum()
    return x, p


@given(matrices_and_distributions())
def test_weighted_cosine_similarities_matches_explicit(x_and_p):
    x, p = x_and_p
    x_norm = _l2_normalize_rows(x)
    s_full = x_norm @ x_norm.T
    expected = s_full @ p
    result = weighted_cosine_similarities(x_norm, p)
    npt.assert_allclose(result, expected, rtol=1e-6, atol=1e-15)
