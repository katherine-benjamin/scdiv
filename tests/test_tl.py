import hypothesis.strategies as st
import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from hypothesis import given, settings
from hypothesis.extra.numpy import arrays

import scdiv.tl

RTOL = 1e-06
ATOL = 1e-10


def _make_adata(x, cell_types=None, samples=None, highly_variable=None):
    x = np.array(x, dtype=float)
    obs = {}
    if cell_types is not None:
        obs["cell_type"] = cell_types
    if samples is not None:
        obs["sample"] = samples
    var = {}
    if highly_variable is not None:
        var["highly_variable"] = highly_variable
    return AnnData(
        X=x,
        obs=pd.DataFrame(obs) if obs else None,
        var=pd.DataFrame(var, index=[f"g{i}" for i in range(x.shape[1])]) if var else None,
    )


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
    scdiv.tl.diversity(
        adata, order, cell_type_key="cell_type", use_highly_variable=False
    )
    div = adata.uns["scdiv_diversity"]
    assert 1 - RTOL <= div <= n_types * (1 + RTOL)


@given(orders)
def test_single_cell_type_gives_one(order):
    x = np.random.default_rng(0).random((5, 3))
    adata = _make_adata(x, cell_types=["A"] * 5)
    scdiv.tl.diversity(
        adata, order, cell_type_key="cell_type", use_highly_variable=False
    )
    assert abs(adata.uns["scdiv_diversity"] - 1.0) < ATOL


@given(orders)
def test_identical_expression_gives_one(order):
    x = np.array([[1.0, 2.0, 3.0]] * 6)
    adata = _make_adata(x, cell_types=["A", "A", "B", "B", "C", "C"])
    scdiv.tl.diversity(
        adata, order, cell_type_key="cell_type", use_highly_variable=False
    )
    assert abs(adata.uns["scdiv_diversity"] - 1.0) < ATOL


@given(adata_with_cell_types(), orders, orders)
def test_cell_type_decreasing_in_order(adata_and_n, order1, order2):
    adata, _ = adata_and_n
    adata2 = adata.copy()
    scdiv.tl.diversity(
        adata, order1, cell_type_key="cell_type", use_highly_variable=False
    )
    scdiv.tl.diversity(
        adata2, order2, cell_type_key="cell_type", use_highly_variable=False
    )
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
    assert abs(adata.uns["scdiv_diversity"] - 1.0) < ATOL


# --- Groupby properties ---


@given(adata_with_groups(), orders)
@settings(max_examples=50)
def test_groupby_all_groups_in_range(adata_n_g, order):
    adata, n_types, n_groups = adata_n_g
    scdiv.tl.diversity(
        adata,
        order,
        cell_type_key="cell_type",
        groupby="sample",
        use_highly_variable=False,
    )
    group_divs = adata.uns["scdiv_diversity"]
    assert len(group_divs) == n_groups
    for div in group_divs.values():
        assert 1 - RTOL <= div <= n_types * (1 + RTOL)


@given(adata_with_groups(), orders)
@settings(max_examples=50)
def test_groupby_obs_matches_uns(adata_n_g, order):
    adata, _, _ = adata_n_g
    scdiv.tl.diversity(
        adata,
        order,
        cell_type_key="cell_type",
        groupby="sample",
        use_highly_variable=False,
    )
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
    scdiv.tl.diversity(
        adata_dense, order, cell_type_key="cell_type", use_highly_variable=False
    )
    scdiv.tl.diversity(
        adata_sparse, order, cell_type_key="cell_type", use_highly_variable=False
    )
    np.testing.assert_allclose(
        adata_dense.uns["scdiv_diversity"],
        adata_sparse.uns["scdiv_diversity"],
        rtol=RTOL,
    )


# --- use_highly_variable ---


def test_hvg_matches_manual_subset():
    """HVG filtering should give the same result as manually subsetting genes."""
    rng = np.random.default_rng(42)
    x = rng.random((8, 6))
    hvg = np.array([True, False, True, True, False, True])
    types = ["A", "A", "B", "B", "C", "C", "A", "B"]
    gene_names = [f"g{i}" for i in range(6)]

    adata_hvg = AnnData(
        X=x,
        obs=pd.DataFrame({"cell_type": types}),
        var=pd.DataFrame({"highly_variable": hvg}, index=gene_names),
    )
    adata_manual = AnnData(
        X=x[:, hvg],
        obs=pd.DataFrame({"cell_type": types}),
        var=pd.DataFrame(index=np.array(gene_names)[hvg]),
    )

    scdiv.tl.diversity(adata_hvg, 1, cell_type_key="cell_type")
    scdiv.tl.diversity(
        adata_manual, 1, cell_type_key="cell_type", use_highly_variable=False
    )
    np.testing.assert_allclose(
        adata_hvg.uns["scdiv_diversity"],
        adata_manual.uns["scdiv_diversity"],
        rtol=RTOL,
    )


def test_hvg_changes_result():
    """Using HVG should generally give a different result than all genes."""
    rng = np.random.default_rng(42)
    x = rng.random((6, 4))
    types = ["A", "A", "B", "B", "C", "C"]

    adata_hvg = _make_adata(
        x, cell_types=types,
        highly_variable=[True, False, True, False],
    )
    adata_all = _make_adata(x, cell_types=types)

    scdiv.tl.diversity(adata_hvg, 1, cell_type_key="cell_type")
    scdiv.tl.diversity(
        adata_all, 1, cell_type_key="cell_type", use_highly_variable=False
    )
    assert (
        adata_hvg.uns["scdiv_diversity"] != adata_all.uns["scdiv_diversity"]
    )


def test_hvg_missing_column_raises():
    adata = _make_adata(np.ones((2, 3)))
    with pytest.raises(KeyError, match="highly_variable"):
        scdiv.tl.diversity(adata, 1)


@given(adata_with_cell_types(), orders)
def test_hvg_diversity_in_range(adata_and_n, order):
    """With HVG filtering, diversity should still be in [1, n_types]."""
    adata, n_types = adata_and_n
    n_genes = adata.X.shape[1]
    adata.var = pd.DataFrame(
        {"highly_variable": [i % 2 == 0 for i in range(n_genes)]},
        index=[f"g{i}" for i in range(n_genes)],
    )
    scdiv.tl.diversity(adata, order, cell_type_key="cell_type")
    div = adata.uns["scdiv_diversity"]
    assert 1 - RTOL <= div <= n_types * (1 + RTOL)


# --- NaN label handling ---


def test_nan_labels_are_dropped():
    """Cells with NaN cell type labels should be dropped with a warning."""
    rng = np.random.default_rng(42)
    x = rng.random((6, 3))
    adata_with_nan = _make_adata(
        x, cell_types=["A", "A", "B", "B", None, None]
    )
    adata_clean = _make_adata(
        x[:4], cell_types=["A", "A", "B", "B"]
    )

    with pytest.warns(UserWarning, match="Dropping 2 cells"):
        scdiv.tl.diversity(
            adata_with_nan, 1,
            cell_type_key="cell_type", use_highly_variable=False,
        )
    scdiv.tl.diversity(
        adata_clean, 1,
        cell_type_key="cell_type", use_highly_variable=False,
    )

    np.testing.assert_allclose(
        adata_with_nan.uns["scdiv_diversity"],
        adata_clean.uns["scdiv_diversity"],
        rtol=RTOL,
    )


def test_no_warning_without_nan_labels():
    """No warning should be raised when all labels are present."""
    import warnings as _warnings  # noqa: PLC0415

    x = np.random.default_rng(0).random((4, 3))
    adata = _make_adata(x, cell_types=["A", "A", "B", "B"])
    with _warnings.catch_warnings():
        _warnings.simplefilter("error")
        scdiv.tl.diversity(
            adata, 1, cell_type_key="cell_type", use_highly_variable=False
        )


# --- layer parameter ---


def test_layer_is_used():
    """Diversity computed from a layer should match using that data as X."""
    rng = np.random.default_rng(42)
    x_main = rng.random((6, 3))
    x_raw = rng.random((6, 3))
    types = ["A", "A", "B", "B", "C", "C"]

    adata_layer = _make_adata(x_main, cell_types=types)
    adata_layer.layers["raw"] = x_raw

    adata_direct = _make_adata(x_raw, cell_types=types)

    scdiv.tl.diversity(
        adata_layer, 1,
        cell_type_key="cell_type", layer="raw", use_highly_variable=False,
    )
    scdiv.tl.diversity(
        adata_direct, 1,
        cell_type_key="cell_type", use_highly_variable=False,
    )

    np.testing.assert_allclose(
        adata_layer.uns["scdiv_diversity"],
        adata_direct.uns["scdiv_diversity"],
        rtol=RTOL,
    )


def test_layer_not_same_as_x():
    """Using a different layer should give a different result than X."""
    rng = np.random.default_rng(42)
    x = rng.random((6, 3))
    types = ["A", "A", "B", "B", "C", "C"]

    adata = _make_adata(x, cell_types=types)
    adata.layers["scaled"] = x * 100 + rng.random((6, 3))

    scdiv.tl.diversity(
        adata, 1, cell_type_key="cell_type", use_highly_variable=False,
        key_added="div_x",
    )
    scdiv.tl.diversity(
        adata, 1, cell_type_key="cell_type", layer="scaled",
        use_highly_variable=False, key_added="div_layer",
    )
    assert adata.uns["div_x"] != adata.uns["div_layer"]


# --- per_group_similarity ---


@given(adata_with_groups(), orders)
@settings(max_examples=50)
def test_per_group_similarity_in_range(adata_n_g, order):
    """With per-group similarity, each group's diversity is in [1, n_types]."""
    adata, n_types, _ = adata_n_g
    scdiv.tl.diversity(
        adata, order,
        cell_type_key="cell_type", groupby="sample",
        per_group_similarity=True, use_highly_variable=False,
    )
    for div in adata.uns["scdiv_diversity"].values():
        assert 1 - RTOL <= div <= n_types * (1 + RTOL)


# --- Groupby edge cases ---


def test_single_cell_group():
    """A group with a single cell should have diversity 1."""
    rng = np.random.default_rng(42)
    x = rng.random((4, 3))
    adata = _make_adata(
        x,
        cell_types=["A", "B", "A", "A"],
        samples=["s1", "s1", "s1", "s2"],
    )
    scdiv.tl.diversity(
        adata, 1,
        cell_type_key="cell_type", groupby="sample",
        use_highly_variable=False,
    )
    assert abs(adata.uns["scdiv_diversity"]["s2"] - 1.0) < ATOL


def test_group_with_one_cell_type():
    """A group where all cells are the same type should have diversity 1."""
    rng = np.random.default_rng(42)
    x = rng.random((6, 3))
    adata = _make_adata(
        x,
        cell_types=["A", "A", "B", "B", "A", "A"],
        samples=["s1", "s1", "s1", "s1", "s2", "s2"],
    )
    scdiv.tl.diversity(
        adata, 1,
        cell_type_key="cell_type", groupby="sample",
        use_highly_variable=False,
    )
    assert abs(adata.uns["scdiv_diversity"]["s2"] - 1.0) < ATOL
