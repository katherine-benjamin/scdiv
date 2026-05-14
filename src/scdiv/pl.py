"""Plotting functions for similarity-sensitive diversity measures."""

import math

import matplotlib.collections
import matplotlib.pyplot as plt
import numpy as np
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
        raise TypeError(msg)

    params = adata.uns.get(f"{key}_params", {})

    if ax is None:
        _, ax = plt.subplots()

    ax.bar([str(k) for k in result], list(result.values()), **kwargs)  # ty: ignore[invalid-argument-type]

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
    kwargs.setdefault("vmin", 0)
    kwargs.setdefault("vmax", 1)
    sns.heatmap(
        similarity,
        xticklabels=cell_types,
        yticklabels=cell_types,
        ax=ax,
        **kwargs,
    )
    return ax


def _region_polygon(
    method: str, center: tuple[float, float], region_size: float
) -> list[tuple[float, float]]:
    """Return polygon vertices for a single region centered at ``center``."""
    cx, cy = center
    if method == "square":
        half = region_size / 2.0
        return [
            (cx - half, cy - half),
            (cx + half, cy - half),
            (cx + half, cy + half),
            (cx - half, cy + half),
        ]
    half_w = math.sqrt(3.0) / 2.0 * region_size
    half_h = region_size / 2.0
    return [
        (cx, cy + region_size),
        (cx + half_w, cy + half_h),
        (cx + half_w, cy - half_h),
        (cx, cy - region_size),
        (cx - half_w, cy - half_h),
        (cx - half_w, cy + half_h),
    ]


def diversity_heatmap(  # noqa: PLR0913
    adata: AnnData,
    *,
    key: str = "scdiv_diversity",
    region_key: str = "spatial_region",
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    colorbar: bool = True,
    ax: Axes | None = None,
    **kwargs: object,
) -> Axes:
    """Polygon map of per-region diversity.

    Args:
        adata:
            AnnData object with a region-grouped diversity result written
            by ``scdiv.tl.diversity(..., groupby=<region_key>)`` after
            running :func:`scdiv.spatial.partition`.
        key:
            Key in ``adata.uns`` holding the ``{region: diversity}`` dict.
        region_key:
            Key whose companion ``adata.uns[region_key + "_params"]``
            holds region geometry (``method``, ``region_size``,
            ``region_centers``).
        cmap:
            Colormap name.
        vmin:
            Lower color limit. If both ``vmin`` and ``vmax`` are None,
            limits are autoscaled from the values.
        vmax:
            Upper color limit. If both ``vmin`` and ``vmax`` are None,
            limits are autoscaled from the values.
        colorbar:
            If True, attach a colorbar to ``ax``.
        ax:
            Matplotlib Axes to draw on. If None, a new figure/axes is
            created.
        **kwargs:
            Forwarded to ``matplotlib.collections.PolyCollection``.

    Returns:
        The matplotlib Axes containing the polygon map.

    """
    result = adata.uns[key]
    if not isinstance(result, dict):
        msg = (
            f"{key!r} is a scalar; diversity_heatmap needs a grouped result "
            "from tl.diversity(..., groupby=<region key>)."
        )
        raise TypeError(msg)

    params_key = f"{region_key}_params"
    if params_key not in adata.uns:
        msg = (
            f"No region metadata at adata.uns[{params_key!r}]. "
            "Run scdiv.spatial.partition first."
        )
        raise KeyError(msg)

    params = adata.uns[params_key]
    method = params["method"]
    region_size = params["region_size"]
    centers = params["region_centers"]

    labels = [lbl for lbl in result if lbl in centers]
    values = np.array([result[lbl] for lbl in labels])
    polys = [_region_polygon(method, centers[lbl], region_size) for lbl in labels]

    if ax is None:
        _, ax = plt.subplots()

    coll = matplotlib.collections.PolyCollection(
        polys,
        array=values,
        cmap=cmap,
        **kwargs,  # ty: ignore[invalid-argument-type]
    )
    if vmin is not None or vmax is not None:
        coll.set_clim(vmin, vmax)
    ax.add_collection(coll)
    ax.autoscale_view()
    ax.set_aspect("equal")

    if colorbar:
        plt.colorbar(coll, ax=ax, label="Diversity")

    return ax
