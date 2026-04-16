import matplotlib
import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from matplotlib.axes import Axes

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import scdiv.pl  # noqa: E402
import scdiv.tl  # noqa: E402


def _make_grouped_adata():
    rng = np.random.default_rng(42)
    x = rng.random((8, 4))
    obs = pd.DataFrame(
        {
            "cell_type": ["A", "A", "B", "B", "A", "B", "A", "B"],
            "sample": ["s1", "s1", "s1", "s1", "s2", "s2", "s2", "s2"],
        }
    )
    return AnnData(X=x, obs=obs)


def test_diversity_bar_returns_axes():
    adata = _make_grouped_adata()
    scdiv.tl.diversity(
        adata, 1,
        cell_type_key="cell_type", groupby="sample",
        use_highly_variable=False,
    )
    ax = scdiv.pl.diversity_bar(adata)
    assert isinstance(ax, Axes)
    plt.close(ax.figure)


def test_diversity_bar_xlabel_matches_groupby():
    adata = _make_grouped_adata()
    scdiv.tl.diversity(
        adata, 1,
        cell_type_key="cell_type", groupby="sample",
        use_highly_variable=False,
    )
    ax = scdiv.pl.diversity_bar(adata)
    assert ax.get_xlabel() == "sample"
    plt.close(ax.figure)


def test_diversity_bar_one_bar_per_group():
    adata = _make_grouped_adata()
    scdiv.tl.diversity(
        adata, 1,
        cell_type_key="cell_type", groupby="sample",
        use_highly_variable=False,
    )
    ax = scdiv.pl.diversity_bar(adata)
    bars = [p for p in ax.patches if isinstance(p, matplotlib.patches.Rectangle)]
    assert len(bars) == 2
    plt.close(ax.figure)


def test_diversity_bar_raises_on_scalar_result():
    adata = _make_grouped_adata()
    scdiv.tl.diversity(
        adata, 1,
        cell_type_key="cell_type",
        use_highly_variable=False,
    )
    with pytest.raises(ValueError, match="grouped result"):
        scdiv.pl.diversity_bar(adata)


def test_diversity_bar_reference_line_toggle():
    adata = _make_grouped_adata()
    scdiv.tl.diversity(
        adata, 1,
        cell_type_key="cell_type", groupby="sample",
        use_highly_variable=False,
    )
    ax_on = scdiv.pl.diversity_bar(adata, reference_line=True)
    assert any(abs(line.get_ydata()[0] - 1.0) < 1e-10 for line in ax_on.get_lines())
    plt.close(ax_on.figure)

    ax_off = scdiv.pl.diversity_bar(adata, reference_line=False)
    assert len(ax_off.get_lines()) == 0
    plt.close(ax_off.figure)


def test_diversity_bar_accepts_user_ax():
    adata = _make_grouped_adata()
    scdiv.tl.diversity(
        adata, 1,
        cell_type_key="cell_type", groupby="sample",
        use_highly_variable=False,
    )
    fig, user_ax = plt.subplots()
    returned = scdiv.pl.diversity_bar(adata, ax=user_ax)
    assert returned is user_ax
    plt.close(fig)


def test_diversity_bar_kwargs_forwarded_to_bar():
    adata = _make_grouped_adata()
    scdiv.tl.diversity(
        adata, 1,
        cell_type_key="cell_type", groupby="sample",
        use_highly_variable=False,
    )
    ax = scdiv.pl.diversity_bar(adata, color="red")
    bars = [p for p in ax.patches if isinstance(p, matplotlib.patches.Rectangle)]
    assert all(p.get_facecolor()[:3] == (1.0, 0.0, 0.0) for p in bars)
    plt.close(ax.figure)


def test_diversity_bar_numeric_group_names():
    rng = np.random.default_rng(42)
    x = rng.random((6, 3))
    obs = pd.DataFrame(
        {
            "cell_type": ["A", "A", "B", "B", "A", "B"],
            "donor": [101, 101, 101, 202, 202, 202],
        }
    )
    adata = AnnData(X=x, obs=obs)
    scdiv.tl.diversity(
        adata, 1,
        cell_type_key="cell_type", groupby="donor",
        use_highly_variable=False,
    )
    ax = scdiv.pl.diversity_bar(adata)
    tick_labels = [t.get_text() for t in ax.get_xticklabels()]
    assert tick_labels == ["101", "202"]
    plt.close(ax.figure)


# --- similarity_heatmap ---


def test_similarity_heatmap_returns_axes():
    adata = _make_grouped_adata()
    scdiv.tl.diversity(
        adata, 1,
        cell_type_key="cell_type",
        use_highly_variable=False,
    )
    ax = scdiv.pl.similarity_heatmap(adata)
    assert isinstance(ax, Axes)
    plt.close(ax.figure)


def test_similarity_heatmap_labels_match_cell_types():
    adata = _make_grouped_adata()
    scdiv.tl.diversity(
        adata, 1,
        cell_type_key="cell_type",
        use_highly_variable=False,
    )
    ax = scdiv.pl.similarity_heatmap(adata)
    xlabels = [t.get_text() for t in ax.get_xticklabels()]
    ylabels = [t.get_text() for t in ax.get_yticklabels()]
    assert xlabels == ["A", "B"]
    assert ylabels == ["A", "B"]
    plt.close(ax.figure)


def test_similarity_heatmap_raises_without_similarity():
    rng = np.random.default_rng(42)
    x = rng.random((6, 3))
    adata = AnnData(X=x)
    scdiv.tl.diversity(adata, 1, use_highly_variable=False)
    with pytest.raises(ValueError, match="similarity matrix"):
        scdiv.pl.similarity_heatmap(adata)


def test_similarity_heatmap_accepts_user_ax():
    adata = _make_grouped_adata()
    scdiv.tl.diversity(
        adata, 1,
        cell_type_key="cell_type",
        use_highly_variable=False,
    )
    fig, user_ax = plt.subplots()
    returned = scdiv.pl.similarity_heatmap(adata, ax=user_ax)
    assert returned is user_ax
    plt.close(fig)


def test_similarity_heatmap_kwargs_override_default():
    adata = _make_grouped_adata()
    scdiv.tl.diversity(
        adata, 1,
        cell_type_key="cell_type",
        use_highly_variable=False,
    )
    ax = scdiv.pl.similarity_heatmap(adata, square=False)
    assert ax.get_aspect() != 1.0
    plt.close(ax.figure)
