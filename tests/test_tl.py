import hypothesis.strategies as st
import numpy as np
import pytest
from anndata import AnnData
from hypothesis import given, settings
from hypothesis.extra.numpy import arrays

import scdiv.tl

RTOL = 1e-06


def _make_adata(x, cell_types=None, samples=None):
    obs = {}
    if cell_types is not None:
        obs["cell_type"] = cell_types
    if samples is not None:
        obs["sample"] = samples
    return AnnData(X=np.array(x, dtype=float), obs=obs)


# --- Strategies ---


@st.composite
def adata_with_cell_types(draw):
    n_types = draw(st.integers(min_value=1, max_value=5))
    cells_per_type = draw(
        st.lists(
            st.integers(min_value=1, max_value=10),
            min_size=n_types,
            max_size=n_types,
        )
    )
    n_cells = sum(cells_per_type)
    n_genes = draw(st.integers(min_value=1, max_value=10))
    x = draw(
        arrays(
            "d",
            (n_cells, n_genes),
            elements=st.floats(
                min_value=0.01, max_value=100, allow_nan=False, allow_infinity=False
            ),
            fill=st.nothing(),
        )
    )
    types = [f"type_{i}" for i, n in enumerate(cells_per_type) for _ in range(n)]
    return _make_adata(x, cell_types=types), n_types


@st.composite
def adata_with_groups(draw):
    n_types = draw(st.integers(min_value=1, max_value=3))
    n_groups = draw(st.integers(min_value=2, max_value=4))
    cells_per_combo = draw(st.integers(min_value=1, max_value=5))
    n_cells = n_types * n_groups * cells_per_combo
    n_genes = draw(st.integers(min_value=1, max_value=10))
    x = draw(
        arrays(
            "d",
            (n_cells, n_genes),
            elements=st.floats(
                min_value=0.01, max_value=100, allow_nan=False, allow_infinity=False
            ),
            fill=st.nothing(),
        )
    )
    types = []
    groups = []
    for g in range(n_groups):
        for t in range(n_types):
            for _ in range(cells_per_combo):
                types.append(f"type_{t}")
                groups.append(f"group_{g}")
    return _make_adata(x, cell_types=types, samples=groups), n_types, n_groups


orders = st.floats(min_value=0, max_value=1000, allow_nan=False)


# --- Cell-type mode properties ---


@given(adata_with_cell_types(), orders)
def test_cell_type_diversity_in_range(adata_and_n, order):
    adata, n_types = adata_and_n
    scdiv.tl.diversity(adata, order, cell_type_key="cell_type", use_highly_variable=False)
    div = adata.uns["scdiv_diversity"]
    assert 1 - RTOL <= div <= n_types * (1 + RTOL)


@given(orders)
def test_single_cell_type_gives_one(order):
    x = np.random.default_rng(0).random((5, 3))
    adata = _make_adata(x, cell_types=["A"] * 5)
    scdiv.tl.diversity(adata, order, cell_type_key="cell_type", use_highly_variable=False)
    assert abs(adata.uns["scdiv_diversity"] - 1.0) < RTOL


@given(orders)
def test_identical_expression_gives_one(order):
    x = np.array([[1.0, 2.0, 3.0]] * 6)
    adata = _make_adata(x, cell_types=["A", "A", "B", "B", "C", "C"])
    scdiv.tl.diversity(adata, order, cell_type_key="cell_type", use_highly_variable=False)
    assert abs(adata.uns["scdiv_diversity"] - 1.0) < RTOL


@given(adata_with_cell_types(), orders, orders)
def test_cell_type_decreasing_in_order(adata_and_n, order1, order2):
    adata, _ = adata_and_n
    adata2 = adata.copy()
    scdiv.tl.diversity(adata, order1, cell_type_key="cell_type", use_highly_variable=False)
    scdiv.tl.diversity(adata2, order2, cell_type_key="cell_type", use_highly_variable=False)
    div1 = adata.uns["scdiv_diversity"]
    div2 = adata2.uns["scdiv_diversity"]
    if order1 <= order2:
        assert div2 <= div1 * (1 + RTOL)
    else:
        assert div1 <= div2 * (1 + RTOL)


# --- Singleton mode properties ---


@given(
    arrays(
        "d",
        st.tuples(st.integers(1, 10), st.integers(1, 10)),
        elements=st.floats(
            min_value=0.01, max_value=100, allow_nan=False, allow_infinity=False
        ),
        fill=st.nothing(),
    ),
    orders,
)
def test_singleton_diversity_in_range(x, order):
    adata = _make_adata(x)
    scdiv.tl.diversity(adata, order, use_highly_variable=False)
    div = adata.uns["scdiv_diversity"]
    n = x.shape[0]
    assert 1 - RTOL <= div <= n * (1 + RTOL)


@given(orders)
def test_singleton_identical_gives_one(order):
    x = np.array([[1.0, 2.0]] * 4)
    adata = _make_adata(x)
    scdiv.tl.diversity(adata, order, use_highly_variable=False)
    assert abs(adata.uns["scdiv_diversity"] - 1.0) < RTOL


# --- Groupby properties ---


@given(adata_with_groups(), orders)
@settings(max_examples=50)
def test_groupby_all_groups_in_range(adata_n_g, order):
    adata, n_types, n_groups = adata_n_g
    scdiv.tl.diversity(adata, order, cell_type_key="cell_type", groupby="sample", use_highly_variable=False)
    group_divs = adata.uns["scdiv_diversity"]
    assert len(group_divs) == n_groups
    for div in group_divs.values():
        assert 1 - RTOL <= div <= n_types * (1 + RTOL)


@given(adata_with_groups(), orders)
@settings(max_examples=50)
def test_groupby_obs_matches_uns(adata_n_g, order):
    adata, _, _ = adata_n_g
    scdiv.tl.diversity(adata, order, cell_type_key="cell_type", groupby="sample", use_highly_variable=False)
    group_divs = adata.uns["scdiv_diversity"]
    for _, row in adata.obs.iterrows():
        assert row["scdiv_diversity"] == group_divs[row["sample"]]


# --- Validation ---


def test_invalid_cell_type_key_raises():
    adata = _make_adata(np.ones((2, 2)))
    with pytest.raises(KeyError, match="nonexistent"):
        scdiv.tl.diversity(adata, 1, cell_type_key="nonexistent")


def test_invalid_groupby_key_raises():
    adata = _make_adata(np.ones((2, 2)))
    with pytest.raises(KeyError, match="nonexistent"):
        scdiv.tl.diversity(adata, 1, groupby="nonexistent")


# --- Sparse support ---


@given(adata_with_cell_types(), orders)
@settings(max_examples=20)
def test_sparse_matches_dense(adata_and_n, order):
    import scipy.sparse  # noqa: PLC0415

    adata_dense, _ = adata_and_n
    adata_sparse = AnnData(
        X=scipy.sparse.csr_matrix(adata_dense.X),
        obs=adata_dense.obs.copy(),
    )
    scdiv.tl.diversity(adata_dense, order, cell_type_key="cell_type", use_highly_variable=False)
    scdiv.tl.diversity(adata_sparse, order, cell_type_key="cell_type", use_highly_variable=False)
    assert abs(
        adata_dense.uns["scdiv_diversity"] - adata_sparse.uns["scdiv_diversity"]
    ) < RTOL
