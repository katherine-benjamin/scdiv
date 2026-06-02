"""AnnData integration for similarity-sensitive diversity measures."""

import warnings
from typing import cast

import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy.sparse
from anndata import AnnData

import scdiv.diversity
import scdiv.similarity
from scdiv._types import Matrix
from scdiv.diversity import _MODES, Mode


def _build_distribution_for_types(
    labels: npt.NDArray, cell_types: npt.NDArray
) -> npt.NDArray:
    """Build a distribution over a fixed set of cell types."""
    types_present, counts = np.unique(labels, return_counts=True)
    distribution = np.zeros(len(cell_types))
    for i, ct in enumerate(cell_types):
        idx = np.searchsorted(types_present, ct)
        if idx < len(types_present) and types_present[idx] == ct:
            distribution[i] = counts[idx]
    return distribution / distribution.sum()


def _compute_cell_type_diversity(  # noqa: PLR0913
    x: Matrix,
    labels: npt.NDArray,
    order: float,
    *,
    similarity: npt.NDArray | None = None,
    cell_types: npt.NDArray | None = None,
    alpha: float = 1.0,
) -> tuple[float, npt.NDArray, npt.NDArray, npt.NDArray]:
    """Compute diversity in cell-type mode.

    If similarity and cell_types are provided (global similarity mode),
    uses them and builds a distribution over the given types. Otherwise
    computes both from the data.

    Returns:
        (diversity_value, similarity_matrix, cell_types, distribution)

    """
    if similarity is None:
        similarity, cell_types = scdiv.similarity.cell_type_similarity(x, labels, alpha)
        dist, _ = scdiv.diversity.distribution_from_labels(labels)
    else:
        if cell_types is None:
            cell_types = np.unique(labels)
        dist = _build_distribution_for_types(labels, cell_types)

    div = scdiv.diversity.diversity(similarity, order, dist)
    return div, similarity, cell_types, dist


def _compute_singleton_diversity(
    x: Matrix, order: float, alpha: float = 1.0
) -> float:
    """Compute diversity treating each cell as its own type.

    Uses factored O(n*d) computation to avoid materialising the
    n_cells x n_cells similarity matrix.
    """
    x_norm = scdiv.similarity.feature_transform(x, alpha)
    n = cast("npt.NDArray", x_norm).shape[0]
    distribution = np.ones(n) / n
    w_sims = scdiv.similarity.weighted_cosine_similarities(x_norm, distribution)
    return scdiv.diversity.diversity_from_weighted_similarities(
        w_sims, order, distribution
    )


def _get_expression_matrix(
    adata: AnnData,
    layer: str | None,
    obsm: str | None = None,
    *,
    use_highly_variable: bool = True,
) -> Matrix:
    """Extract a per-cell vector representation.

    The source is picked from at most one of:

    - ``obsm``: a key in ``adata.obsm`` (``use_highly_variable`` ignored)
    - ``layer``: a key in ``adata.layers``. Same shape as ``adata.X``;
    - neither: ``adata.X``.

    Args:
        adata: Annotated data matrix.
        layer: Key in ``adata.layers``, or None.
        obsm: Key in ``adata.obsm``, or None. Mutually exclusive with
            ``layer``.
        use_highly_variable: If True, subset to highly variable genes.
            Ignored when ``obsm`` is set.

    Returns:
        Matrix of shape ``(n_cells, n_features)``.

    """
    if obsm is not None:
        if obsm not in adata.obsm:
            msg = f"obsm key {obsm!r} not found in adata.obsm."
            raise KeyError(msg)
        x_obsm = np.asarray(adata.obsm[obsm], dtype=float)
        if (x_obsm < 0).any():
            warnings.warn(
                f"adata.obsm[{obsm!r}] contains negative values. "
                "Similarity-sensitive diversity uses cosine similarity, "
                "which assumes a non-negative representation; "
                "mean-centered embeddings (eg. PCA) can break the "
                "power-mean inside the diversity formula. Pass a "
                "non-negative representation instead.",
                stacklevel=3,
            )
        return x_obsm

    x = adata.layers[layer] if layer is not None else adata.X

    if use_highly_variable:
        if "highly_variable" not in adata.var.columns:
            msg = (
                "use_highly_variable=True but 'highly_variable' not found "
                "in adata.var. Run sc.pp.highly_variable_genes first."
            )
            raise KeyError(msg)
        hvg_mask = adata.var["highly_variable"].to_numpy()
        x = x[:, hvg_mask]  # ty: ignore[not-subscriptable]

    if scipy.sparse.issparse(x):
        return cast("scipy.sparse.csr_matrix", x).tocsr()
    return np.asarray(x)


def _is_spatial_partition(adata: AnnData, groupby: str) -> bool:
    """Return True if ``groupby`` was set by :func:`scdiv.spatial.partition`."""
    params = adata.uns.get(f"{groupby}_params", None)
    return isinstance(params, dict) and "partition_method" in params


def _get_labels_and_mask(
    adata: AnnData, cell_type_key: str | None
) -> tuple[npt.NDArray | None, npt.NDArray]:
    """Extract cell type labels and a mask for non-NaN entries.

    Args:
        adata: Annotated data matrix.
        cell_type_key: Column in adata.obs, or None.

    Returns:
        (labels, mask) where labels is a numpy array of cell type
        labels, shape (n_cells,), or None if cell_type_key is None.
        mask is a boolean array, shape (n_cells,), marking cells
        with valid (non-NaN) labels.

    """
    if cell_type_key is None:
        return None, np.ones(adata.n_obs, dtype=bool)

    labels = adata.obs[cell_type_key].to_numpy()
    mask = pd.notna(labels)
    if not mask.all():
        n_dropped = (~mask).sum()
        warnings.warn(
            f"Dropping {n_dropped} cells with missing {cell_type_key!r} labels.",
            stacklevel=3,
        )
    return labels, mask


def _compute_global(
    x: Matrix,
    mask: npt.NDArray,
    labels: npt.NDArray | None,
    order: float,
    alpha: float = 1.0,
) -> tuple[float, dict]:
    """Compute a single diversity value across all (masked) cells.

    Args:
        x: Expression matrix, shape (n_cells, n_genes).
        mask: Boolean array, shape (n_cells,). Cells to include.
        labels: Cell type labels, shape (n_cells,), or None for
            singleton mode.
        order: Order of the power mean.
        alpha: Exponent of the probability-geometric similarity family.

    Returns:
        (diversity_value, extras) where extras is a dict of computed
        quantities (similarity matrix, cell types, distribution) or
        empty in singleton mode.

    """
    x_masked = cast("npt.NDArray", x)[mask]

    if labels is None:
        return _compute_singleton_diversity(x_masked, order, alpha), {}

    labels_masked = labels[mask]
    div, sim, cell_types, dist = _compute_cell_type_diversity(
        x_masked, labels_masked, order, alpha=alpha
    )
    extras = {
        "similarity": sim,
        "cell_types": list(cell_types),
        "distribution": dist,
    }
    return div, extras


def _compute_grouped_celltype(  # noqa: PLR0913
    x: Matrix,
    mask: npt.NDArray,
    labels: npt.NDArray,
    order: float,
    groups: pd.Series,
    *,
    mode: Mode,
    aggregate: bool,
    alpha: float = 1.0,
) -> tuple[dict, float | None, dict]:
    """Per-group diversity in cell-type mode via Reeve partition diversity."""
    x_masked = cast("npt.NDArray", x)[mask]
    labels_masked = labels[mask]
    sim, cell_types = scdiv.similarity.cell_type_similarity(
        x_masked, labels_masked, alpha
    )
    n_total = int(mask.sum())

    group_keys: list = []
    cols: list[npt.NDArray] = []
    weights: list[float] = []
    for g in groups.unique():
        group_mask = (groups == g).to_numpy() & mask
        n_group = int(group_mask.sum())
        if n_group == 0:
            continue
        group_keys.append(g)
        cols.append(_build_distribution_for_types(labels[group_mask], cell_types))
        weights.append(n_group / n_total)

    if not group_keys:
        warnings.warn(
            "No non-empty groups for diversity computation; "
            "returning empty per-group result and NaN aggregate.",
            stacklevel=3,
        )
        return {}, (float("nan") if aggregate else None), {
            "similarity": sim,
            "cell_types": list(cell_types),
        }

    distributions = np.column_stack(cols)
    per_group, meta = scdiv.diversity.partition_diversity(
        sim,
        distributions,
        np.array(weights),
        order,
        mode=mode,
        aggregate=aggregate,
    )
    group_diversities = dict(zip(group_keys, per_group, strict=True))
    extras = {"similarity": sim, "cell_types": list(cell_types)}
    return group_diversities, meta, extras


def _compute_grouped_singleton(  # noqa: PLR0913
    x: Matrix,
    mask: npt.NDArray,
    order: float,
    groups: pd.Series,
    *,
    mode: Mode,
    aggregate: bool,
    alpha: float = 1.0,
) -> tuple[dict, float | None]:
    """Per-group diversity in singleton mode (each cell its own type)."""
    x_norm = scdiv.similarity.feature_transform(cast("npt.NDArray", x)[mask], alpha)
    group_keys, group_idx = np.unique(groups[mask].to_numpy(), return_inverse=True)
    if len(group_keys) == 0:
        warnings.warn(
            "No non-empty groups for diversity computation; "
            "returning empty per-group result and NaN aggregate.",
            stacklevel=3,
        )
        return {}, (float("nan") if aggregate else None)
    per_group, meta = scdiv.diversity.partition_diversity_singleton(
        x_norm,
        group_idx,
        order,
        mode=mode,
        aggregate=aggregate,
    )
    return dict(zip(group_keys, per_group, strict=True)), meta


def diversity(  # noqa: PLR0913
    adata: AnnData,
    order: float,
    *,
    cell_type_key: str | None = None,
    groupby: str | None = None,
    layer: str | None = None,
    obsm: str | None = None,
    use_highly_variable: bool = True,
    alpha: float = 1.0,
    mode: Mode = "alpha_norm",
    aggregate: bool = False,
    key_added: str = "scdiv_diversity",
) -> None:
    """Compute similarity-sensitive diversity on an AnnData object.

    Two modes:
        - Singleton (cell_type_key=None): each cell is its own type with
          uniform distribution.
        - Cell type (cell_type_key given): aggregates to cell types.
          Similarity = cosine similarity of mean expression per type.
          Distribution = type proportions.

    Args:
        adata:
            Annotated data matrix.
        order:
            The order of the power mean used to average diversity.
        cell_type_key:
            Column in adata.obs containing cell type labels. If None,
            each cell is treated as its own type.
        groupby:
            Column in adata.obs to group by (e.g. 'sample'). Computes
            diversity per group.
        layer:
            Key in adata.layers to use. If None and ``obsm`` is None,
            uses adata.X. Mutually exclusive with ``obsm``.
        obsm:
            Key in adata.obsm holding the per-cell vector representation.
            Mutually exclusive with ``layer``.
            When set, ``use_highly_variable`` is ignored.
        use_highly_variable:
            If True, restrict to genes marked as highly variable in
            adata.var['highly_variable']. If False, use all genes.
            Ignored when ``obsm`` is set.
        alpha:
            Exponent of the probability-geometric similarity family. Each
            cell is treated as a distribution over genes, raised to the
            power ``alpha`` before L2 normalisation. ``alpha=1`` (default)
            is cosine similarity; ``alpha=0.5`` is Bhattacharyya; smaller
            values down-weight highly expressed genes.
        mode:
            Partition diversity mode in the style of Reeve et al. (2016),
            relevant when ``groupby`` is set. One of:

            - ``"alpha_norm"`` (default): standalone diversity of each
              subcommunity; in [1, n_types].
            - ``"alpha"``: alpha_norm divided by the subcommunity weight
              w_j; a "diversity share" that can exceed n_types.
            - ``"gamma"``: each subcommunity's contribution to the pooled
              metacommunity diversity (ordinariness against the pool).
        aggregate:
            If True and ``groupby`` is set, also store a single
            metacommunity-level scalar (the w_j-weighted power mean of
            order ``1 - order`` of the per-group values; for ``gamma``,
            this equals the diversity of the pooled distribution) at
            ``adata.uns[f"{key_added}_metacommunity"]``.
        key_added:
            Key for storing results in adata.uns and adata.obs.

    """
    if mode not in _MODES:
        msg = f"mode must be one of {_MODES}, got {mode!r}."
        raise ValueError(msg)
    if alpha <= 0:
        msg = f"alpha must be positive, got {alpha!r}."
        raise ValueError(msg)
    if groupby is None and aggregate:
        msg = "aggregate=True requires groupby to be set."
        raise ValueError(msg)
    if layer is not None and obsm is not None:
        msg = "Pass at most one of `layer` and `obsm`."
        raise TypeError(msg)
    _validate_keys(adata, cell_type_key, groupby)
    x = _get_expression_matrix(
        adata, layer, obsm=obsm, use_highly_variable=use_highly_variable
    )
    labels, mask = _get_labels_and_mask(adata, cell_type_key)

    base_params = {
        "order": order,
        "cell_type_key": cell_type_key,
        "groupby": groupby,
        "layer": layer,
        "obsm": obsm,
        "use_highly_variable": use_highly_variable,
        "alpha": alpha,
        "mode": mode,
    }

    if groupby is None:
        div, extras = _compute_global(x, mask, labels, order, alpha)
        adata.uns[key_added] = div
        adata.uns[f"{key_added}_params"] = {**base_params, **extras}
        return

    groups = pd.Series(adata.obs[groupby])
    groupby_mask = pd.notna(groups).to_numpy()
    if not groupby_mask.all() and not _is_spatial_partition(adata, groupby):
        n_dropped = int((~groupby_mask).sum())
        warnings.warn(
            f"Dropping {n_dropped} cells with missing {groupby!r} labels.",
            stacklevel=2,
        )
    mask = mask & groupby_mask
    if labels is None:
        group_divs, meta = _compute_grouped_singleton(
            x,
            mask,
            order,
            groups,
            mode=mode,
            aggregate=aggregate,
            alpha=alpha,
        )
        extras: dict = {}
    else:
        group_divs, meta, extras = _compute_grouped_celltype(
            x,
            mask,
            labels,
            order,
            groups,
            mode=mode,
            aggregate=aggregate,
            alpha=alpha,
        )
    adata.uns[key_added] = group_divs
    adata.obs[key_added] = groups.map(group_divs).to_numpy()
    adata.uns[f"{key_added}_params"] = {**base_params, **extras}
    if meta is not None:
        adata.uns[f"{key_added}_metacommunity"] = meta


def _validate_keys(
    adata: AnnData,
    cell_type_key: str | None,
    groupby: str | None,
) -> None:
    """Raise KeyError if obs columns are missing.

    Args:
        adata: Annotated data matrix.
        cell_type_key: Column name to check, or None.
        groupby: Column name to check, or None.

    """
    if cell_type_key is not None and cell_type_key not in adata.obs.columns:
        msg = f"cell_type_key {cell_type_key!r} not found in adata.obs."
        raise KeyError(msg)
    if groupby is not None and groupby not in adata.obs.columns:
        msg = f"groupby key {groupby!r} not found in adata.obs."
        raise KeyError(msg)


def sparsity(
    adata: AnnData,
    *,
    layer: str | None = None,
    obsm: str | None = None,
    region_key: str | None = None,
    key_added: str = "sparsity",
) -> None:
    """Per-cell fraction of zero entries.

    Writes the per-cell zero fraction to ``adata.obs[key_added]``. When
    ``region_key`` is set, also writes the per-region mean to
    ``adata.uns[key_added]`` as ``{region: float}``; this is directly
    consumable by :func:`scdiv.pl.diversity_heatmap` (after running
    :func:`scdiv.spatial.partition`) and :func:`scdiv.pl.diversity_vs_metric`.

    Args:
        adata:
            Annotated data matrix.
        layer:
            Key in ``adata.layers`` to score. If None and ``obsm`` is
            None, uses ``adata.X``. Mutually exclusive with ``obsm``.
        obsm:
            Key in ``adata.obsm`` to score (any per-cell matrix).
            Mutually exclusive with ``layer``.
        region_key:
            Column in ``adata.obs`` holding region labels. When set,
            the per-region mean is written to ``adata.uns[key_added]``.
        key_added:
            Key for storing the per-cell value in ``adata.obs`` and
            (if ``region_key`` is set) the per-region mean in
            ``adata.uns``.

    """
    if layer is not None and obsm is not None:
        msg = "Pass at most one of `layer` and `obsm`."
        raise TypeError(msg)

    if obsm is not None:
        if obsm not in adata.obsm:
            msg = f"obsm key {obsm!r} not found in adata.obsm."
            raise KeyError(msg)
        x = adata.obsm[obsm]
    elif layer is not None:
        if layer not in adata.layers:
            msg = f"layer key {layer!r} not found in adata.layers."
            raise KeyError(msg)
        x = adata.layers[layer]
    else:
        x = adata.X

    n_features = x.shape[1]  # ty:ignore[unresolved-attribute]
    if hasattr(x, "getnnz"):
        nnz = np.asarray(x.getnnz(axis=1)).ravel()  # ty: ignore[call-non-callable]
    else:
        nnz = (np.asarray(x) != 0).sum(axis=1)
    adata.obs[key_added] = 1.0 - nnz / n_features

    if region_key is not None:
        if region_key not in adata.obs.columns:
            msg = f"region_key {region_key!r} not found in adata.obs."
            raise KeyError(msg)
        means = adata.obs.groupby(region_key, observed=True)[key_added].mean()  # ty:ignore[unresolved-attribute]
        adata.uns[key_added] = means.to_dict()
