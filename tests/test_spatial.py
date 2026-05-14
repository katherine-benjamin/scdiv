import matplotlib
import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.collections import PolyCollection  # noqa: E402

import scdiv.pl  # noqa: E402
import scdiv.spatial  # noqa: E402
import scdiv.tl  # noqa: E402

ATOL = 1e-10


def _make_spatial_adata(coords, expression=None, cell_types=None):
    n = coords.shape[0]
    if expression is None:
        rng = np.random.default_rng(0)
        expression = rng.random((n, 3))
    obs = {}
    if cell_types is not None:
        obs["cell_type"] = cell_types
    adata = AnnData(
        X=np.asarray(expression, dtype=float),
        obs=pd.DataFrame(obs, index=[f"c{i}" for i in range(n)])
        if obs
        else pd.DataFrame(index=[f"c{i}" for i in range(n)]),
    )
    adata.obsm["spatial"] = np.asarray(coords, dtype=float)
    return adata


# --- Square assignments ---


def test_square_assigns_points_to_correct_cell():
    coords = np.array(
        [
            [0.1, 0.1],
            [0.9, 0.1],
            [1.1, 0.1],
            [0.1, 1.1],
            [1.1, 1.1],
        ]
    )
    expression = np.ones((5, 2))
    adata = _make_spatial_adata(coords, expression=expression)
    scdiv.spatial.partition(adata, method="square", region_size=1.0, min_cells=1)
    labels = adata.obs["spatial_region"].to_numpy()
    assert labels[0] == labels[1]
    assert labels[1] != labels[2]
    assert labels[0] != labels[3]
    assert labels[2] != labels[4]


def test_square_centers_are_geometric_centers():
    coords = np.array([[0.2, 0.3], [0.8, 0.6], [2.5, 2.5]])
    adata = _make_spatial_adata(coords, expression=np.ones((3, 2)))
    scdiv.spatial.partition(adata, method="square", region_size=1.0, min_cells=1)
    centers = adata.uns["spatial_region_params"]["region_centers"]
    # x_min=0.2, y_min=0.3, region_size=1.0
    # First two points are in the (0,0) region with center (0.7, 0.8)
    label_first = adata.obs["spatial_region"].to_numpy()[0]
    cx, cy = centers[label_first]
    assert abs(cx - 0.7) < ATOL
    assert abs(cy - 0.8) < ATOL


def test_square_min_cells_drops_sparse_regions():
    coords = np.array(
        [
            [0.1, 0.1],
            [0.2, 0.2],
            [0.3, 0.3],  # region A (3 cells)
            [5.0, 5.0],  # region B (1 cell, should be dropped)
        ]
    )
    adata = _make_spatial_adata(coords, expression=np.ones((4, 2)))
    scdiv.spatial.partition(adata, method="square", region_size=1.0, min_cells=2)
    labels = adata.obs["spatial_region"]
    assert labels.iloc[0] == labels.iloc[1] == labels.iloc[2]
    assert pd.isna(labels.iloc[3])
    centers = adata.uns["spatial_region_params"]["region_centers"]
    assert len(centers) == 1


# --- Hex assignments ---


def test_hex_axial_origin_at_origin():
    """A point at (0, 0) lands in axial hex (0, 0)."""
    coords = np.array([[0.0, 0.0]])
    adata = _make_spatial_adata(coords, expression=np.ones((1, 2)))
    scdiv.spatial.partition(adata, method="hex", region_size=1.0, min_cells=1)
    assert adata.obs["spatial_region"].iloc[0] == "0,0"


def test_hex_centers_are_at_axial_positions():
    """Hex centers should match the axial -> pixel formula."""
    coords = np.array([[0.0, 0.0], [3.0, 0.0]])
    adata = _make_spatial_adata(coords, expression=np.ones((2, 2)))
    scdiv.spatial.partition(adata, method="hex", region_size=1.0, min_cells=1)
    centers = adata.uns["spatial_region_params"]["region_centers"]
    cx0, cy0 = centers["0,0"]
    assert abs(cx0 - 0.0) < ATOL
    assert abs(cy0 - 0.0) < ATOL


def test_hex_neighboring_points_share_region():
    """Points well inside a single hex share the same label."""
    coords = np.array([[0.0, 0.0], [0.1, 0.1], [-0.1, -0.1]])
    adata = _make_spatial_adata(coords, expression=np.ones((3, 2)))
    scdiv.spatial.partition(adata, method="hex", region_size=1.0, min_cells=1)
    labels = adata.obs["spatial_region"].to_numpy()
    assert labels[0] == labels[1] == labels[2] == "0,0"


# --- Validation ---


def test_invalid_method_raises():
    adata = _make_spatial_adata(np.zeros((2, 2)), expression=np.ones((2, 2)))
    with pytest.raises(ValueError, match="method"):
        scdiv.spatial.partition(
            adata,
            method="bogus",
            region_size=1.0,
        )


def test_missing_spatial_key_raises():
    rng = np.random.default_rng(0)
    adata = AnnData(X=rng.random((4, 2)))
    with pytest.raises(KeyError, match="spatial"):
        scdiv.spatial.partition(adata, method="square", region_size=1.0)


def test_non_positive_region_size_raises():
    adata = _make_spatial_adata(np.zeros((2, 2)), expression=np.ones((2, 2)))
    with pytest.raises(ValueError, match="region_size"):
        scdiv.spatial.partition(adata, method="square", region_size=0.0)


# --- Integration with tl.diversity ---


def test_partition_then_diversity_alpha():
    rng = np.random.default_rng(42)
    n = 40
    # Two clusters in (x, y) far apart
    coords = np.vstack([rng.random((n // 2, 2)), rng.random((n // 2, 2)) + 10.0])
    x = rng.random((n, 3))
    cell_types = ["A", "B"] * (n // 2)
    adata = _make_spatial_adata(coords, expression=x, cell_types=cell_types)
    scdiv.spatial.partition(adata, method="square", region_size=2.0, min_cells=2)
    scdiv.tl.diversity(
        adata,
        order=1,
        cell_type_key="cell_type",
        groupby="spatial_region",
        mode="alpha_norm",
        use_highly_variable=False,
    )
    result = adata.uns["scdiv_diversity"]
    assert isinstance(result, dict)
    assert len(result) >= 1
    for div in result.values():
        assert 1.0 - 1e-6 <= div <= 2.0 + 1e-6


def test_partition_then_diversity_gamma_aggregate():
    rng = np.random.default_rng(42)
    n = 30
    coords = rng.random((n, 2)) * 5.0
    x = rng.random((n, 3))
    cell_types = ["A", "B", "C"] * (n // 3)
    adata = _make_spatial_adata(coords, expression=x, cell_types=cell_types)
    scdiv.spatial.partition(adata, method="hex", region_size=1.0, min_cells=2)
    scdiv.tl.diversity(
        adata,
        order=1,
        cell_type_key="cell_type",
        groupby="spatial_region",
        mode="gamma",
        aggregate=True,
        use_highly_variable=False,
    )
    assert "scdiv_diversity_metacommunity" in adata.uns


# --- diversity_heatmap plot ---


def _prepared_adata(method="square"):
    rng = np.random.default_rng(0)
    n = 30
    coords = rng.random((n, 2)) * 5.0
    x = rng.random((n, 3))
    cell_types = ["A", "B"] * (n // 2)
    adata = _make_spatial_adata(coords, expression=x, cell_types=cell_types)
    scdiv.spatial.partition(adata, method=method, region_size=2.0, min_cells=2)
    scdiv.tl.diversity(
        adata,
        order=1,
        cell_type_key="cell_type",
        groupby="spatial_region",
        mode="alpha_norm",
        use_highly_variable=False,
    )
    return adata


def test_diversity_heatmap_returns_axes():
    adata = _prepared_adata()
    ax = scdiv.pl.diversity_heatmap(adata)
    assert isinstance(ax, Axes)
    plt.close(ax.figure)


def test_diversity_heatmap_square_has_four_sided_polygons():
    adata = _prepared_adata(method="square")
    ax = scdiv.pl.diversity_heatmap(adata, colorbar=False)
    colls = [c for c in ax.collections if isinstance(c, PolyCollection)]
    assert len(colls) == 1
    n_regions = len(adata.uns["scdiv_diversity"])
    paths = colls[0].get_paths()
    assert len(paths) == n_regions
    # Path has 4 unique vertices + closing vertex
    assert all(len(p.vertices) == 5 for p in paths)
    plt.close(ax.figure)


def test_diversity_heatmap_hex_has_six_sided_polygons():
    adata = _prepared_adata(method="hex")
    ax = scdiv.pl.diversity_heatmap(adata, colorbar=False)
    colls = [c for c in ax.collections if isinstance(c, PolyCollection)]
    paths = colls[0].get_paths()
    assert all(len(p.vertices) == 7 for p in paths)
    plt.close(ax.figure)


def test_diversity_heatmap_raises_without_params():
    adata = _prepared_adata()
    del adata.uns["spatial_region_params"]
    with pytest.raises(KeyError, match="region metadata"):
        scdiv.pl.diversity_heatmap(adata)


def test_diversity_wrapper_matches_two_step():
    """scdiv.spatial.diversity should match partition + tl.diversity."""
    rng = np.random.default_rng(7)
    n = 30
    coords = rng.random((n, 2)) * 5.0
    x = rng.random((n, 3))
    cell_types = ["A", "B"] * (n // 2)

    adata_two = _make_spatial_adata(coords, expression=x, cell_types=cell_types)
    scdiv.spatial.partition(adata_two, method="square", region_size=2.0, min_cells=2)
    scdiv.tl.diversity(
        adata_two,
        1,
        cell_type_key="cell_type",
        groupby="spatial_region",
        mode="alpha_norm",
        use_highly_variable=False,
    )

    adata_one = _make_spatial_adata(coords, expression=x, cell_types=cell_types)
    scdiv.spatial.diversity(
        adata_one,
        1,
        partition={"method": "square", "region_size": 2.0, "min_cells": 2},
        cell_type_key="cell_type",
        mode="alpha_norm",
        use_highly_variable=False,
    )

    two = adata_two.uns["scdiv_diversity"]
    one = adata_one.uns["scdiv_diversity"]
    assert set(one) == set(two)
    for k in one:
        np.testing.assert_allclose(one[k], two[k], rtol=1e-12)


def test_diversity_wrapper_aggregate_flag():
    rng = np.random.default_rng(7)
    n = 30
    coords = rng.random((n, 2)) * 5.0
    x = rng.random((n, 3))
    adata = _make_spatial_adata(
        coords, expression=x, cell_types=["A", "B", "C"] * (n // 3)
    )
    scdiv.spatial.diversity(
        adata,
        1,
        partition={"method": "hex", "region_size": 1.0, "min_cells": 2},
        cell_type_key="cell_type",
        mode="gamma",
        aggregate=True,
        use_highly_variable=False,
    )
    assert "scdiv_diversity_metacommunity" in adata.uns


def test_diversity_wrapper_rejects_groupby_kwarg():
    rng = np.random.default_rng(0)
    adata = _make_spatial_adata(rng.random((6, 2)), expression=np.ones((6, 2)))
    with pytest.raises(TypeError, match="groupby"):
        scdiv.spatial.diversity(
            adata,
            1,
            partition={"region_size": 2.0, "min_cells": 1},
            groupby="something",
            use_highly_variable=False,
        )


def test_diversity_heatmap_raises_on_scalar():
    rng = np.random.default_rng(0)
    adata = _make_spatial_adata(rng.random((4, 2)), expression=np.ones((4, 2)))
    scdiv.spatial.partition(adata, method="square", region_size=2.0, min_cells=1)
    scdiv.tl.diversity(adata, 1, use_highly_variable=False)
    with pytest.raises(TypeError, match="grouped result"):
        scdiv.pl.diversity_heatmap(adata)
