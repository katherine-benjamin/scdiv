"""Plotting functions for similarity-sensitive diversity measures."""

import matplotlib.pyplot as plt
import seaborn as sns
from anndata import AnnData
from matplotlib.axes import Axes


def diversity_bar(
    adata: AnnData,
    *,
    key: str = "scdiv_diversity",
    reference_line: bool = True,
    ax: Axes | None = None,
    **kwargs: object,
) -> Axes:
    """Bar chart of per-group diversity scores.

    Args:
        adata:
            AnnData object with a grouped diversity result written by
            ``scdiv.tl.diversity(..., groupby=...)``.
        key:
            Key in ``adata.uns`` holding the ``{group: diversity}`` dict.
            Companion metadata is expected at ``adata.uns[key + "_params"]``.
        reference_line:
            If True, draw a dashed horizontal line at diversity = 1.
        ax:
            Matplotlib Axes to draw on. If None, a new figure/axes is created.
        **kwargs:
            Forwarded to ``ax.bar``.

    Returns:
        The matplotlib Axes containing the bar chart.

    """
    result = adata.uns[key]
    if not isinstance(result, dict):
        msg = (
            f"{key!r} is a scalar; diversity_bar needs a grouped result "
            "from tl.diversity(..., groupby=...)."
        )
        raise ValueError(msg)

    params = adata.uns.get(f"{key}_params", {})

    if ax is None:
        _, ax = plt.subplots()

    ax.bar([str(k) for k in result], list(result.values()), **kwargs)

    if reference_line:
        ax.axhline(1.0, linestyle="--", color="gray", linewidth=1)

    ax.set_ylabel("Diversity")
    if groupby := params.get("groupby"):
        ax.set_xlabel(groupby)

    return ax


def similarity_heatmap(
    adata: AnnData,
    *,
    key: str = "scdiv_diversity",
    ax: Axes | None = None,
    **kwargs: object,
) -> Axes:
    """Heatmap of the between-cell-type similarity matrix.

    Args:
        adata:
            AnnData object with a cell-type-mode diversity result written
            by ``scdiv.tl.diversity(..., cell_type_key=...)``.
        key:
            Key whose companion ``adata.uns[key + "_params"]`` holds the
            similarity matrix and cell-type labels.
        ax:
            Matplotlib Axes to draw on. If None, a new figure/axes is created.
        **kwargs:
            Forwarded to ``seaborn.heatmap``.

    Returns:
        The matplotlib Axes containing the heatmap.

    """
    params = adata.uns.get(f"{key}_params", {})
    if "similarity" not in params:
        msg = (
            f"No similarity matrix found at {key + '_params'!r}. "
            "Run tl.diversity with cell_type_key=... first."
        )
        raise ValueError(msg)

    similarity = params["similarity"]
    cell_types = params.get("cell_types", [])

    if ax is None:
        _, ax = plt.subplots()

    kwargs.setdefault("square", True)
    sns.heatmap(
        similarity,
        xticklabels=cell_types,
        yticklabels=cell_types,
        ax=ax,
        **kwargs,
    )
    return ax
