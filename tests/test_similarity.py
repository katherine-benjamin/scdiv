import hypothesis.strategies as st
import numpy as np
import numpy.testing as npt
import scipy.sparse
from hypothesis import given
from hypothesis.extra.numpy import arrays

from scdiv.similarity import (
    cell_type_similarity,
    cosine_similarity_matrix,
    feature_transform,
    l2_normalize_rows,
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
def testl2_normalize_rows_unit_norm_or_zero(x):
    result = l2_normalize_rows(x)
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


@given(expression_matrices)
def test_feature_transform_alpha_one_is_cosine_normalization(x):
    npt.assert_allclose(feature_transform(x, 1.0), l2_normalize_rows(x), rtol=1e-10)


@given(expression_matrices)
def test_feature_transform_alpha_half_is_bhattacharyya(x):
    row_sums = x.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    expected = l2_normalize_rows(np.sqrt(x / row_sums))
    npt.assert_allclose(feature_transform(x, 0.5), expected, rtol=1e-6, atol=1e-15)


@given(expression_matrices)
def test_feature_transform_rows_unit_norm_or_zero(x):
    norms = np.linalg.norm(feature_transform(x, 0.5), axis=1)
    for norm, row_sum in zip(norms, x.sum(axis=1), strict=True):
        if row_sum == 0:
            npt.assert_allclose(norm, 0.0, atol=1e-10)
        else:
            npt.assert_allclose(norm, 1.0, rtol=1e-6)


@given(expression_matrices)
def test_l2_normalize_rows_sparse_matches_dense(x):
    dense = l2_normalize_rows(x)
    sparse = l2_normalize_rows(scipy.sparse.csr_matrix(x))
    npt.assert_allclose(sparse.toarray(), dense, rtol=1e-6, atol=1e-12)


@given(expression_matrices, st.sampled_from([1.0, 0.5, 0.25]))
def test_feature_transform_sparse_matches_dense(x, alpha):
    dense = feature_transform(x, alpha)
    sparse = feature_transform(scipy.sparse.csr_matrix(x), alpha)
    npt.assert_allclose(sparse.toarray(), dense, rtol=1e-6, atol=1e-12)


def test_feature_transform_keeps_float32_sparse():
    x = scipy.sparse.csr_matrix(np.eye(4, dtype=np.float32))
    assert feature_transform(x, 1.0).dtype == np.float32
    assert feature_transform(x, 0.5).dtype == np.float32


def test_cell_type_similarity_sparse_matches_dense():
    rng = np.random.default_rng(0)
    x = rng.poisson(1.5, size=(20, 8)).astype(float)
    labels = np.array([f"t{i % 3}" for i in range(20)])
    dense_sim, _ = cell_type_similarity(x, labels, alpha=0.5)
    sparse_sim, _ = cell_type_similarity(scipy.sparse.csr_matrix(x), labels, alpha=0.5)
    npt.assert_allclose(sparse_sim, dense_sim, rtol=1e-6, atol=1e-12)


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
    x_norm = l2_normalize_rows(x)
    s_full = x_norm @ x_norm.T
    expected = s_full @ p
    result = weighted_cosine_similarities(x_norm, p)
    npt.assert_allclose(result, expected, rtol=1e-6, atol=1e-15)
