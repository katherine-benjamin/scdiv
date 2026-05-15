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
        var=pd.DataFrame(
            var, index=[f"g{i}" for i in range(x.shape[1])]
        ) if var else None,
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


def test_obsm_matches_x_when_data_is_same():
    """obsm path gives the same result as X when the matrix is the same."""
    rng = np.random.default_rng(0)
    x = rng.random((6, 3))
    types = ["A", "A", "B", "B", "C", "C"]

    adata_obsm = _make_adata(x, cell_types=types)
    adata_obsm.obsm["X_rep"] = x.copy()

    adata_x = _make_adata(x, cell_types=types)

    scdiv.tl.diversity(
        adata_obsm, 1, cell_type_key="cell_type", obsm="X_rep",
    )
    scdiv.tl.diversity(
        adata_x, 1, cell_type_key="cell_type", use_highly_variable=False,
    )
    np.testing.assert_allclose(
        adata_obsm.uns["scdiv_diversity"], adata_x.uns["scdiv_diversity"],
        rtol=RTOL,
    )


def test_obsm_with_different_dim():
    """obsm of shape (n_obs, k != n_vars) is accepted as-is."""
    rng = np.random.default_rng(0)
    x = rng.random((6, 4))
    pcs = rng.random((6, 10))  # n_features != n_vars
    types = ["A", "A", "B", "B", "C", "C"]
    adata = _make_adata(x, cell_types=types)
    adata.obsm["X_pca"] = pcs

    scdiv.tl.diversity(adata, 1, cell_type_key="cell_type", obsm="X_pca")
    assert "scdiv_diversity" in adata.uns


def test_obsm_and_layer_raises():
    """Passing both layer and obsm is a usage error."""
    x = np.random.default_rng(0).random((4, 3))
    adata = _make_adata(x, cell_types=["A", "A", "B", "B"])
    adata.layers["raw"] = x.copy()
    adata.obsm["X_rep"] = x.copy()
    with pytest.raises(TypeError, match="at most one of"):
        scdiv.tl.diversity(
            adata, 1, cell_type_key="cell_type", layer="raw", obsm="X_rep",
        )


def test_obsm_missing_key_raises():
    """Missing obsm key surfaces as KeyError."""
    x = np.random.default_rng(0).random((4, 3))
    adata = _make_adata(x, cell_types=["A", "A", "B", "B"])
    with pytest.raises(KeyError, match="obsm key 'X_pca' not found"):
        scdiv.tl.diversity(adata, 1, cell_type_key="cell_type", obsm="X_pca")


def test_obsm_skips_hvg_check():
    """With obsm, missing 'highly_variable' should not raise."""
    rng = np.random.default_rng(0)
    x = rng.random((6, 3))
    adata = _make_adata(x, cell_types=["A", "A", "B", "B", "C", "C"])  # no HVG col
    adata.obsm["X_rep"] = x.copy()
    # Default use_highly_variable=True normally raises without the column;
    # with obsm set the flag is ignored.
    scdiv.tl.diversity(adata, 1, cell_type_key="cell_type", obsm="X_rep")
    assert "scdiv_diversity" in adata.uns


def test_obsm_negative_values_warns():
    """obsm matrices with negative entries trigger a warning about cosine
    similarity assumptions."""
    rng = np.random.default_rng(0)
    x = rng.random((6, 3))
    adata = _make_adata(x, cell_types=["A", "A", "B", "B", "C", "C"])
    # PCA-like rep with negative entries
    adata.obsm["X_pca"] = rng.standard_normal((6, 4))
    with pytest.warns(UserWarning, match="non-negative representation"):
        scdiv.tl.diversity(adata, 1, cell_type_key="cell_type", obsm="X_pca")


def test_obsm_nonneg_values_no_warn():
    """No warning for a strictly non-negative obsm matrix."""
    rng = np.random.default_rng(0)
    x = rng.random((6, 3))
    adata = _make_adata(x, cell_types=["A", "A", "B", "B", "C", "C"])
    adata.obsm["X_rep"] = rng.random((6, 4))  # all in [0, 1)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        scdiv.tl.diversity(adata, 1, cell_type_key="cell_type", obsm="X_rep")


def test_obsm_grouped():
    """obsm path works with the groupby branch."""
    rng = np.random.default_rng(1)
    x = rng.random((8, 4))
    adata = _make_adata(
        x,
        cell_types=["A", "A", "B", "B", "A", "A", "B", "B"],
        samples=["s1", "s1", "s1", "s1", "s2", "s2", "s2", "s2"],
    )
    adata.obsm["X_rep"] = rng.random((8, 5))
    scdiv.tl.diversity(
        adata, 1, cell_type_key="cell_type", groupby="sample", obsm="X_rep",
    )
    assert isinstance(adata.uns["scdiv_diversity"], dict)
    assert set(adata.uns["scdiv_diversity"].keys()) == {"s1", "s2"}
    assert adata.uns["scdiv_diversity_params"]["obsm"] == "X_rep"


# --- Reeve et al. modes (alpha / alpha_norm / gamma) ---


@given(adata_with_groups(), orders)
@settings(max_examples=50)
def test_alpha_norm_in_range(adata_n_g, order):
    """alpha_norm is in [1, n_types] per group."""
    adata, n_types, _ = adata_n_g
    scdiv.tl.diversity(
        adata, order,
        cell_type_key="cell_type", groupby="sample",
        mode="alpha_norm", use_highly_variable=False,
    )
    for div in adata.uns["scdiv_diversity"].values():
        assert 1 - RTOL <= div <= n_types * (1 + RTOL)


@given(adata_with_groups(), orders)
@settings(max_examples=50)
def test_alpha_equals_alpha_norm_over_weight(adata_n_g, order):
    """Reeve identity: raw alpha_j = alpha_norm_j / w_j."""
    adata, _, _ = adata_n_g
    adata2 = adata.copy()
    scdiv.tl.diversity(
        adata, order, cell_type_key="cell_type", groupby="sample",
        mode="alpha_norm", use_highly_variable=False,
    )
    scdiv.tl.diversity(
        adata2, order, cell_type_key="cell_type", groupby="sample",
        mode="alpha", use_highly_variable=False,
    )
    counts = adata.obs["sample"].value_counts()
    n_total = counts.sum()
    for g, alpha_norm in adata.uns["scdiv_diversity"].items():
        w_j = counts[g] / n_total
        np.testing.assert_allclose(
            adata2.uns["scdiv_diversity"][g], alpha_norm / w_j, rtol=RTOL,
        )


@given(adata_with_groups(), orders)
@settings(max_examples=50)
def test_gamma_in_range(adata_n_g, order):
    """gamma per-group values are in [1, n_types]."""
    adata, n_types, _ = adata_n_g
    scdiv.tl.diversity(
        adata, order, cell_type_key="cell_type", groupby="sample",
        mode="gamma", use_highly_variable=False,
    )
    for div in adata.uns["scdiv_diversity"].values():
        assert 1 - RTOL <= div <= n_types * (1 + RTOL)


@given(adata_with_groups(), orders)
@settings(max_examples=50)
def test_gamma_aggregate_matches_pooled(adata_n_g, order):
    """The gamma aggregate equals the diversity of the pooled metacommunity."""
    adata, _, _ = adata_n_g
    adata2 = adata.copy()
    scdiv.tl.diversity(
        adata, order, cell_type_key="cell_type", groupby="sample",
        mode="gamma", aggregate=True, use_highly_variable=False,
    )
    scdiv.tl.diversity(
        adata2, order, cell_type_key="cell_type", use_highly_variable=False,
    )
    np.testing.assert_allclose(
        adata.uns["scdiv_diversity_metacommunity"],
        adata2.uns["scdiv_diversity"],
        rtol=RTOL,
    )


def test_invalid_mode_raises():
    adata = _make_adata(
        np.ones((4, 2)),
        cell_types=["A", "B", "A", "B"],
        samples=["s1", "s1", "s2", "s2"],
    )
    with pytest.raises(ValueError, match="mode"):
        scdiv.tl.diversity(
            adata, 1, cell_type_key="cell_type", groupby="sample",
            mode="bogus", use_highly_variable=False,
        )


def test_aggregate_requires_groupby():
    adata = _make_adata(np.ones((4, 2)), cell_types=["A", "B", "A", "B"])
    with pytest.raises(ValueError, match="groupby"):
        scdiv.tl.diversity(
            adata, 1, cell_type_key="cell_type",
            aggregate=True, use_highly_variable=False,
        )


def test_singleton_grouped_modes_run():
    """All three modes work in singleton (no cell_type_key) grouped mode."""
    rng = np.random.default_rng(0)
    x = rng.random((8, 3))
    adata = _make_adata(x, samples=["s1"] * 4 + ["s2"] * 4)
    for mode in ("alpha_norm", "alpha", "gamma"):
        scdiv.tl.diversity(
            adata, 1, groupby="sample", mode=mode,
            aggregate=True, use_highly_variable=False,
        )
        assert set(adata.uns["scdiv_diversity"]) == {"s1", "s2"}
        assert "scdiv_diversity_metacommunity" in adata.uns


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


def test_nan_groupby_dropped_singleton():
    """Cells with NaN groupby labels should be dropped (singleton mode)."""
    rng = np.random.default_rng(0)
    x = rng.random((6, 3))
    adata_with_nan = _make_adata(x, samples=["s1", "s1", "s2", "s2", None, None])
    adata_clean = _make_adata(x[:4], samples=["s1", "s1", "s2", "s2"])

    with pytest.warns(UserWarning, match="Dropping 2 cells"):
        scdiv.tl.diversity(
            adata_with_nan, 1, groupby="sample", use_highly_variable=False,
        )
    scdiv.tl.diversity(
        adata_clean, 1, groupby="sample", use_highly_variable=False,
    )

    for key in ("s1", "s2"):
        np.testing.assert_allclose(
            adata_with_nan.uns["scdiv_diversity"][key],
            adata_clean.uns["scdiv_diversity"][key],
            rtol=RTOL,
        )


def test_nan_groupby_dropped_celltype():
    """Cells with NaN groupby labels should be dropped (cell-type mode).

    Cell-type mode previously silently included NaN-group cells in the
    total-cell denominator, inflating w_j and biasing the result.
    """
    rng = np.random.default_rng(0)
    x = rng.random((8, 3))
    cell_types = ["A", "B", "A", "B", "A", "B", "A", "B"]
    adata_with_nan = _make_adata(
        x,
        cell_types=cell_types,
        samples=["s1", "s1", "s2", "s2", "s2", "s2", None, None],
    )
    adata_clean = _make_adata(
        x[:6],
        cell_types=cell_types[:6],
        samples=["s1", "s1", "s2", "s2", "s2", "s2"],
    )

    with pytest.warns(UserWarning, match="Dropping 2 cells"):
        scdiv.tl.diversity(
            adata_with_nan, 1,
            cell_type_key="cell_type", groupby="sample",
            use_highly_variable=False,
        )
    scdiv.tl.diversity(
        adata_clean, 1,
        cell_type_key="cell_type", groupby="sample",
        use_highly_variable=False,
    )

    for key in ("s1", "s2"):
        np.testing.assert_allclose(
            adata_with_nan.uns["scdiv_diversity"][key],
            adata_clean.uns["scdiv_diversity"][key],
            rtol=RTOL,
        )


# --- sparsity ---


def test_sparsity_per_cell_correct():
    """Per-cell zero fraction matches a hand-computed expectation."""
    x = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],  # 3/4 = 0.75
            [1.0, 2.0, 0.0, 0.0],  # 2/4 = 0.50
            [1.0, 2.0, 3.0, 4.0],  # 0/4 = 0.00
        ]
    )
    adata = AnnData(X=x)
    scdiv.tl.sparsity(adata)
    np.testing.assert_allclose(
        adata.obs["sparsity"].to_numpy(), [0.75, 0.50, 0.0]
    )


def test_sparsity_sparse_matrix_path():
    """Sparse CSR input uses the .getnnz fast path with the same result."""
    import scipy.sparse as sp
    x_dense = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 2.0, 3.0]])
    adata = AnnData(X=sp.csr_matrix(x_dense))
    scdiv.tl.sparsity(adata)
    np.testing.assert_allclose(
        adata.obs["sparsity"].to_numpy(), [2 / 3, 1.0, 0.0]
    )


def test_sparsity_layer_path():
    """layer= selects an alternative gene-space matrix."""
    rng = np.random.default_rng(0)
    x = rng.random((5, 4))
    adata = AnnData(X=x)
    adata.layers["zeros_only"] = np.zeros((5, 4))
    scdiv.tl.sparsity(adata, layer="zeros_only", key_added="zf")
    np.testing.assert_allclose(adata.obs["zf"].to_numpy(), [1.0] * 5)


def test_sparsity_obsm_path():
    """obsm= scores any per-cell matrix; second dim need not match n_vars."""
    rng = np.random.default_rng(0)
    x = rng.random((5, 4))
    adata = AnnData(X=x)
    adata.obsm["latent"] = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )
    scdiv.tl.sparsity(adata, obsm="latent")
    np.testing.assert_allclose(
        adata.obs["sparsity"].to_numpy(), [2 / 3, 1.0, 0.0, 1 / 3, 2 / 3]
    )


def test_sparsity_layer_and_obsm_raises():
    x = np.random.default_rng(0).random((4, 3))
    adata = AnnData(X=x)
    adata.layers["raw"] = x.copy()
    adata.obsm["latent"] = x.copy()
    with pytest.raises(TypeError, match="at most one of"):
        scdiv.tl.sparsity(adata, layer="raw", obsm="latent")


def test_sparsity_missing_layer_raises():
    x = np.random.default_rng(0).random((4, 3))
    adata = AnnData(X=x)
    with pytest.raises(KeyError, match="layer key 'absent'"):
        scdiv.tl.sparsity(adata, layer="absent")


def test_sparsity_missing_obsm_raises():
    x = np.random.default_rng(0).random((4, 3))
    adata = AnnData(X=x)
    with pytest.raises(KeyError, match="obsm key 'absent'"):
        scdiv.tl.sparsity(adata, obsm="absent")


def test_sparsity_region_aggregate():
    """region_key triggers per-region mean in uns."""
    x = np.array(
        [
            [1.0, 0.0, 0.0],  # 2/3
            [1.0, 1.0, 0.0],  # 1/3
            [1.0, 1.0, 1.0],  # 0
            [0.0, 0.0, 0.0],  # 1
        ]
    )
    adata = AnnData(
        X=x,
        obs=pd.DataFrame(
            {"region": pd.Categorical(["A", "A", "B", "B"])}
        ),
    )
    scdiv.tl.sparsity(adata, region_key="region")
    assert adata.uns["sparsity"]["A"] == pytest.approx(0.5)  # mean(2/3, 1/3)
    assert adata.uns["sparsity"]["B"] == pytest.approx(0.5)  # mean(0, 1)


def test_sparsity_missing_region_key_raises():
    x = np.random.default_rng(0).random((4, 3))
    adata = AnnData(X=x)
    with pytest.raises(KeyError, match="region_key 'no_col'"):
        scdiv.tl.sparsity(adata, region_key="no_col")


def test_sparsity_no_region_key_skips_uns():
    x = np.random.default_rng(0).random((4, 3))
    adata = AnnData(X=x)
    scdiv.tl.sparsity(adata)
    assert "sparsity" in adata.obs.columns
    assert "sparsity" not in adata.uns
