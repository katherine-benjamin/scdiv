"""Spatial region assignment for diversity analyses."""

import math
import warnings
from typing import Literal

import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy.spatial
from anndata import AnnData

import scdiv.tl

Method = Literal["square", "hex"]
_METHODS: tuple[Method, ...] = ("square", "hex")

_HEX_AREA_COEFF = 1.5 * math.sqrt(3.0)
_OVERLAP_WARN_THRESHOLD = 1.05


def _cell_area(method: Method, cell_radius: float) -> float:
    """Area of one cell. ``cell_radius`` is half-side (square) or circumradius (hex)."""
    if method == "square":
        return (2.0 * cell_radius) ** 2
    return _HEX_AREA_COEFF * cell_radius**2


def _region_area(method: Method, region_size: float) -> float:
    """Area of one region. ``region_size`` is side (square) or circumradius (hex)."""
    if method == "square":
        return region_size**2
    return _HEX_AREA_COEFF * region_size**2


def _infer_cell_radius(coords: npt.NDArray, method: Method) -> float:
    """Infer ``cell_radius`` from median nearest-neighbor distance.

    For grid-binned data this is the bin spacing. The returned radius is
    chosen so that a fully-tiled region reaches coverage 1.0 for the
    matching shape.
    """
    tree = scipy.spatial.cKDTree(coords)
    dists, _ = tree.query(coords, k=2)
    nn = float(np.median(np.asarray(dists)[:, 1]))
    if method == "square":
        return nn / 2.0
    return nn * math.sqrt(1.0 / _HEX_AREA_COEFF)


def _square_assignments(
    coords: npt.NDArray, region_size: float
) -> tuple[npt.NDArray, npt.NDArray]:
    """Assign points to square-grid regions.

    Args:
        coords: Spatial coordinates, shape (n, 2).
        region_size: Side length of each square region.

    Returns:
        (labels, centers) where labels has shape (n,) with strings of the
        form ``"ix,iy"`` and centers has shape (n, 2) giving the geometric
        center of the region each point falls in.

    """
    x_min, y_min = coords.min(axis=0)
    ix = np.floor((coords[:, 0] - x_min) / region_size).astype(int)
    iy = np.floor((coords[:, 1] - y_min) / region_size).astype(int)
    labels = np.array([f"{a},{b}" for a, b in zip(ix, iy, strict=True)])
    cx = x_min + (ix + 0.5) * region_size
    cy = y_min + (iy + 0.5) * region_size
    return labels, np.column_stack([cx, cy])


def _hex_assignments(
    coords: npt.NDArray, region_size: float
) -> tuple[npt.NDArray, npt.NDArray]:
    """Assign points to hexagonal regions.

    See https://www.redblobgames.com/grids/hexagons/

    Args:
        coords: Spatial coordinates, shape (n, 2).
        region_size: Circumradius of each hex (vertex-to-center distance).

    Returns:
        (labels, centers) where labels has shape (n,) with strings of the
        form ``"q,r"`` and centers has shape (n, 2).

    """
    sqrt3 = math.sqrt(3.0)

    # Step 1: project each pixel coord into fractional cube coords
    q_frac = (sqrt3 / 3.0 * coords[:, 0] - coords[:, 1] / 3.0) / region_size
    r_frac = (2.0 / 3.0 * coords[:, 1]) / region_size
    s_frac = -q_frac - r_frac

    # Step 2: round to the nearest integer hex.
    q = np.round(q_frac).astype(int)
    r = np.round(r_frac).astype(int)
    s = np.round(s_frac).astype(int)

    q_err = np.abs(q - q_frac)
    r_err = np.abs(r - r_frac)
    s_err = np.abs(s - s_frac)
    fix_q = (q_err > r_err) & (q_err > s_err)
    fix_r = (~fix_q) & (r_err > s_err)
    q[fix_q] = -r[fix_q] - s[fix_q]
    r[fix_r] = -q[fix_r] - s[fix_r]

    # Step 3: convert each hex's (q, r) back to a pixel center and build
    # the "q,r" string labels used downstream.
    labels = np.array([f"{a},{b}" for a, b in zip(q, r, strict=True)])
    cx = region_size * (sqrt3 * q + sqrt3 / 2.0 * r)
    cy = region_size * 1.5 * r
    return labels, np.column_stack([cx, cy])


def partition(  # noqa: PLR0913
    adata: AnnData,
    *,
    method: Method = "square",
    region_size: float,
    spatial_key: str = "spatial",
    min_cells: int = 50,
    min_density: float = 0.5,
    cell_radius: float | Literal["auto"] | None = None,
    key_added: str = "spatial_region",
) -> None:
    """Partition cells into spatial regions for downstream diversity analysis.

    Tiles the (x, y) plane into squares or hexagons and assigns
    every cell a region label. Regions with fewer than ``min_cells`` cells
    are dropped, then optionally regions where the total cell area falls
    below ``min_density`` times the region area.

    Args:
        adata:
            Annotated data matrix. Spatial coordinates are read from
            ``adata.obsm[spatial_key]`` (first two columns).
        method:
            ``"square"`` for a square grid (``region_size`` = side length)
            or ``"hex"`` for a hex grid (``region_size`` =
            circumradius / vertex-to-center distance).
        region_size:
            Characteristic length scale of each region.
        spatial_key:
            Key in ``adata.obsm`` holding spatial coordinates.
        min_cells:
            Regions with strictly fewer than this many cells are dropped.
        min_density:
            Coverage threshold in [0, 1]. After ``min_cells`` filtering,
            drop regions whose total cell area is less than ``min_density``
            of the total region area. ``0.0`` disables this filter.
        cell_radius:
            Half-side (square) or circumradius (hex) of one cell. Used
            only when ``min_density > 0``. If ``auto`` then radius is
            inferred from spatial coordinates.
        key_added:
            Where to store region labels in ``adata.obs`` and parameters
            in ``adata.uns[key_added + "_params"]``.

    """
    if method not in _METHODS:
        msg = f"method must be one of {_METHODS}, got {method!r}."
        raise ValueError(msg)
    if spatial_key not in adata.obsm:
        msg = f"spatial_key {spatial_key!r} not found in adata.obsm."
        raise KeyError(msg)
    if region_size <= 0:
        msg = f"region_size must be positive, got {region_size}."
        raise ValueError(msg)

    coords = np.asarray(adata.obsm[spatial_key])[:, :2].astype(float)

    if method == "square":
        labels, centers = _square_assignments(coords, region_size)
    else:
        labels, centers = _hex_assignments(coords, region_size)

    unique, first_idx, counts = np.unique(labels, return_index=True, return_counts=True)
    keep = counts >= min_cells

    resolved_radius: float | None = None
    if min_density > 0:
        if cell_radius is None or cell_radius == "auto":
            if cell_radius is None:
                warnings.warn(
                    "cell_radius not provided; inferring from spatial "
                    "coordinates. Pass cell_radius='auto' to suppress this "
                    "warning or a numeric value to override.",
                    stacklevel=2,
                )
            resolved_radius = _infer_cell_radius(coords, method)
        else:
            resolved_radius = float(cell_radius)

        cell_area = _cell_area(method, resolved_radius)
        region_area = _region_area(method, region_size)
        coverage = counts * cell_area / region_area
        if keep.any() and (coverage[keep] > _OVERLAP_WARN_THRESHOLD).any():
            max_cov = float(coverage[keep].max())
            warnings.warn(
                f"Total cell area exceeds region area in some regions "
                f"(max coverage = {max_cov:.2f}); cell_radius may be too "
                f"large or cells may overlap.",
                stacklevel=2,
            )
        keep = keep & (coverage >= min_density)

    kept = unique[keep].tolist()

    # `pd.Categorical` coerces values not in `categories` to NaN, which
    # naturally drops cells whose region was filtered out.
    adata.obs[key_added] = pd.Categorical(labels, categories=kept)

    region_centers = {
        str(lbl): (float(centers[i, 0]), float(centers[i, 1]))
        for lbl, i in zip(kept, first_idx[keep], strict=True)
    }

    adata.uns[f"{key_added}_params"] = {
        "partition_method": method,
        "region_size": float(region_size),
        "spatial_key": spatial_key,
        "min_cells": int(min_cells),
        "min_density": float(min_density),
        "cell_radius": resolved_radius,
        "region_centers": region_centers,
    }


def diversity(
    adata: AnnData,
    order: float,
    *,
    partition_kwargs: dict,
    **kwargs: object,
) -> None:
    """Partition cells into spatial regions and compute per-region diversity.

    Args:
        adata: Annotated data matrix with spatial coordinates.
        order: Order of the diversity.
        partition_kwargs: Keyword arguments forwarded to :func:`partition`.
        **kwargs: Forwarded to :func:`scdiv.tl.diversity`; must not include
            ``groupby`` (it is bound to the region key from the partition).

    """
    if "groupby" in kwargs:
        msg = (
            "`groupby` cannot be passed to scdiv.spatial.diversity; it is "
            "bound to the region key from partition_kwargs."
        )
        raise TypeError(msg)
    partition(adata, **partition_kwargs)
    kwargs["groupby"] = partition_kwargs.get("key_added", "spatial_region")
    scdiv.tl.diversity(adata, order, **kwargs)  # ty: ignore[invalid-argument-type]
