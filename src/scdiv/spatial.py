"""Spatial region assignment for diversity analyses."""

import math
from typing import Literal

import numpy as np
import numpy.typing as npt
import pandas as pd
from anndata import AnnData

import scdiv.tl

Method = Literal["square", "hex"]
_METHODS: tuple[Method, ...] = ("square", "hex")


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
    """Assign points to pointy-top hexagonal regions.

    Uses axial coordinates with circumradius ``region_size`` (vertex
    distance from center). Fractional axial coords are rounded via the
    standard cube-coordinate rounding so points land in the nearest hex.

    Args:
        coords: Spatial coordinates, shape (n, 2).
        region_size: Circumradius of each hex (vertex-to-center distance).

    Returns:
        (labels, centers) where labels has shape (n,) with strings of the
        form ``"q,r"`` and centers has shape (n, 2).

    """
    sqrt3 = math.sqrt(3.0)
    qf = (sqrt3 / 3.0 * coords[:, 0] - coords[:, 1] / 3.0) / region_size
    rf = (2.0 / 3.0 * coords[:, 1]) / region_size
    sf = -qf - rf

    rq = np.round(qf).astype(int)
    rr = np.round(rf).astype(int)
    rs = np.round(sf).astype(int)

    dq = np.abs(rq - qf)
    dr = np.abs(rr - rf)
    ds = np.abs(rs - sf)
    fix_q = (dq > dr) & (dq > ds)
    fix_r = (~fix_q) & (dr > ds)
    rq[fix_q] = -rr[fix_q] - rs[fix_q]
    rr[fix_r] = -rq[fix_r] - rs[fix_r]

    labels = np.array([f"{a},{b}" for a, b in zip(rq, rr, strict=True)])
    cx = region_size * (sqrt3 * rq + sqrt3 / 2.0 * rr)
    cy = region_size * 1.5 * rr
    return labels, np.column_stack([cx, cy])


def partition(  # noqa: PLR0913
    adata: AnnData,
    *,
    method: Method = "square",
    region_size: float,
    spatial_key: str = "spatial",
    min_cells: int = 10,
    key_added: str = "spatial_region",
) -> None:
    """Partition cells into spatial regions for downstream diversity analysis.

    Tiles the (x, y) plane into squares or pointy-top hexagons and assigns
    every cell a region label. Regions with fewer than ``min_cells`` cells
    are dropped.

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
    kept = unique[keep].tolist()

    # `pd.Categorical` coerces values not in `categories` to NaN, which
    # naturally drops cells whose region was filtered out by min_cells.
    adata.obs[key_added] = pd.Categorical(labels, categories=kept)

    region_centers = {
        str(lbl): (float(centers[i, 0]), float(centers[i, 1]))
        for lbl, i in zip(kept, first_idx[keep], strict=True)
    }

    adata.uns[f"{key_added}_params"] = {
        "method": method,
        "region_size": float(region_size),
        "spatial_key": spatial_key,
        "min_cells": int(min_cells),
        "region_centers": region_centers,
    }


_partition_fn = partition


def diversity(
    adata: AnnData,
    order: float,
    *,
    partition: dict,
    **kwargs: object,
) -> None:
    """Partition cells into spatial regions and compute per-region diversity.

    Args:
        adata: Annotated data matrix with spatial coordinates.
        order: Order of the diversity.
        partition: Keyword arguments for :func:`partition`.
        **kwargs: Forwarded to :func:`scdiv.tl.diversity`; must not include
            ``groupby`` (it is bound to the region key from ``partition``).

    """
    if "groupby" in kwargs:
        msg = (
            "`groupby` cannot be passed to scdiv.spatial.diversity; it is "
            "bound to the region key from `partition`."
        )
        raise TypeError(msg)
    _partition_fn(adata, **partition)
    kwargs["groupby"] = partition.get("key_added", "spatial_region")
    scdiv.tl.diversity(adata, order, **kwargs)  # ty: ignore[invalid-argument-type]
