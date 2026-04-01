"""AnnData integration for similarity-sensitive diversity measures."""

import warnings

import numpy as np
import numpy.typing as npt
import pandas as pd
from anndata import AnnData

from scdiv.diversity import diversity as _core_diversity
from scdiv.diversity import diversity_from_weighted_similarities
from scdiv.similarity import (
    _l2_normalize_rows,
    cosine_similarity_matrix,
    normalize_columns,
    weighted_cosine_similarities,
)


def _mean_expression_per_type(
    x: npt.NDArray, labels: npt.NDArray, cell_types: npt.NDArray
) -> npt.NDArray:
    """Compute mean expression vector for each cell type.

    Args:
        x: Expression matrix, shape (n_cells, n_genes).
        labels: Cell type label for each cell, shape (n_cells,).
        cell_types: Unique cell types to compute means for, shape (n_types,).

    Returns:
        Mean expression per type, shape (n_types, n_genes).

    """
    means = np.empty((len(cell_types), x.shape[1]))
    for i, ct in enumerate(cell_types):
        means[i] = x[labels == ct].mean(axis=0)
    return means


def _build_distribution(
    labels: npt.NDArray, cell_types: npt.NDArray
) -> npt.NDArray:
    """Build a distribution over cell_types from observed labels.

    Args:
        labels: Cell type label for each cell, shape (n_cells,).
        cell_types: The full set of cell types, shape (n_types,).

    Returns:
        Normalized abundance vector, shape (n_types,). Sums to 1.

    """
    types_present, counts = np.unique(labels, return_counts=True)
    distribution = np.zeros(len(cell_types))
    for i, ct in enumerate(cell_types):
        idx = np.searchsorted(types_present, ct)
        if idx < len(types_present) and types_present[idx] == ct:
            distribution[i] = counts[idx]
    return distribution / distribution.sum()


def _compute_cell_type_diversity(
    x: npt.NDArray,
    labels: npt.NDArray,
    order: float,
    *,
    similarity: npt.NDArray | None = None,
    cell_types: npt.NDArray | None = None,
) -> tuple[float, npt.NDArray, npt.NDArray, npt.NDArray]:
    """Compute diversity in cell-type mode.

    Returns:
        (diversity_value, similarity_matrix, cell_types, distribution)

    """
    if cell_types is None:
        cell_types = np.unique(labels)

    distribution = _build_distribution(labels, cell_types)

    if similarity is None:
        x_col_norm = normalize_columns(x)
        means = _mean_expression_per_type(x_col_norm, labels, cell_types)
        similarity = cosine_similarity_matrix(means)

    div = _core_diversity(similarity, order, distribution)
    return div, similarity, cell_types, distribution


def _compute_singleton_diversity(x: npt.NDArray, order: float) -> float:
    """Compute diversity treating each cell as its own type.

    Uses factored O(n*d) computation to avoid materialising the
    n_cells x n_cells similarity matrix.
    """
    x_col_norm = normalize_columns(x)
    x_norm = _l2_normalize_rows(x_col_norm)
    n = x_norm.shape[0]
    distribution = np.ones(n) / n
    w_sims = weighted_cosine_similarities(x_norm, distribution)
    return diversity_from_weighted_similarities(w_sims, order, distribution)


def _get_expression_matrix(
    adata: AnnData,
    layer: str | None,
    *,
    use_highly_variable: bool = True,
) -> npt.NDArray:
    """Extract expression matrix as a dense numpy array.

    Args:
        adata: Annotated data matrix.
        layer: Key in adata.layers, or None to use adata.X.
        use_highly_variable: If True, subset to highly variable genes.

    Returns:
        Dense expression matrix, shape (n_cells, n_genes).

    """
    x = adata.layers[layer] if layer is not None else adata.X

    if use_highly_variable:
        if "highly_variable" not in adata.var.columns:
            msg = (
                "use_highly_variable=True but 'highly_variable' not found "
                "in adata.var. Run sc.pp.highly_variable_genes first."
            )
            raise KeyError(msg)
        hvg_mask = adata.var["highly_variable"].to_numpy()
        x = x[:, hvg_mask]  # pyright: ignore[reportOptionalSubscript]

    if hasattr(x, "todense"):
        return np.asarray(x.todense())  # pyright: ignore[reportAttributeAccessIssue, reportOptionalMemberAccess]
    return np.asarray(x, dtype=float)


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
            f"Dropping {n_dropped} cells with missing "
            f"{cell_type_key!r} labels.",
            stacklevel=3,
        )
    return labels, mask


def _compute_global(
    x: npt.NDArray,
    mask: npt.NDArray,
    labels: npt.NDArray | None,
    order: float,
) -> tuple[float, dict]:
    """Compute a single diversity value across all (masked) cells.

    Args:
        x: Expression matrix, shape (n_cells, n_genes).
        mask: Boolean array, shape (n_cells,). Cells to include.
        labels: Cell type labels, shape (n_cells,), or None for
            singleton mode.
        order: Order of the power mean.

    Returns:
        (diversity_value, params) where params is a dict containing
        similarity matrix, cell types, and distribution (empty dict
        in singleton mode).

    """
    x_masked = x[mask]

    if labels is None:
        return _compute_singleton_diversity(x_masked, order), {}

    labels_masked = labels[mask]
    div, sim, cell_types, dist = _compute_cell_type_diversity(
        x_masked, labels_masked, order
    )
    params = {
        "similarity": sim,
        "cell_types": list(cell_types),
        "distribution": dist,
    }
    return div, params


def _compute_grouped(  # noqa: PLR0913
    x: npt.NDArray,
    mask: npt.NDArray,
    labels: npt.NDArray | None,
    order: float,
    groups: pd.Series,
    *,
    per_group_similarity: bool,
) -> tuple[dict, dict]:
    """Compute diversity per group.

    Args:
        x: Expression matrix, shape (n_cells, n_genes).
        mask: Boolean array, shape (n_cells,). Cells to include.
        labels: Cell type labels, shape (n_cells,), or None for
            singleton mode.
        order: Order of the power mean.
        groups: Group assignment for each cell.
        per_group_similarity: If True, recompute similarity within
            each group. If False, use a global similarity matrix.

    Returns:
        (group_divs, params) where group_divs maps group name to
        diversity value, and params contains the global similarity
        matrix and cell types if applicable.

    """
    global_sim = None
    global_cell_types = None

    if labels is not None and not per_group_similarity:
        x_masked = x[mask]
        labels_masked = labels[mask]
        global_cell_types = np.unique(labels_masked)
        x_col_norm = normalize_columns(x_masked)
        means = _mean_expression_per_type(
            x_col_norm, labels_masked, global_cell_types
        )
        global_sim = cosine_similarity_matrix(means)

    group_diversities: dict = {}
    for g in groups.unique():
        group_mask = (groups == g).to_numpy() & mask
        x_group = x[group_mask]
        if x_group.shape[0] == 0:
            continue

        if labels is None:
            group_diversities[g] = _compute_singleton_diversity(
                x_group, order
            )
        else:
            div, *_ = _compute_cell_type_diversity(
                x_group,
                labels[group_mask],
                order,
                similarity=global_sim,
                cell_types=global_cell_types,
            )
            group_diversities[g] = div

    params: dict = {}
    if global_cell_types is not None:
        params["similarity"] = global_sim
        params["cell_types"] = list(global_cell_types)
    return group_diversities, params


def diversity(  # noqa: PLR0913
    adata: AnnData,
    order: float,
    *,
    cell_type_key: str | None = None,
    groupby: str | None = None,
    layer: str | None = None,
    use_highly_variable: bool = True,
    per_group_similarity: bool = False,
    key_added: str = "scdiv_diversity",
) -> None:
    """Compute similarity-sensitive diversity on an AnnData object.

    Two modes:
        - Singleton (cell_type_key=None): each cell is its own type with
          uniform distribution.
        - Cell type (cell_type_key given): aggregates to cell types.
          Similarity = cosine similarity of mean expression per type
          (after per-gene L1 normalization). Distribution = type
          proportions.

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
            Key in adata.layers to use. If None, uses adata.X.
        use_highly_variable:
            If True, restrict to genes marked as highly variable in
            adata.var['highly_variable']. If False, use all genes.
        per_group_similarity:
            If True and groupby is set, recompute the similarity matrix
            within each group. Only relevant in cell-type mode.
            If False, a global similarity matrix is shared across groups.
        key_added:
            Key for storing results in adata.uns and adata.obs.

    """
    _validate_keys(adata, cell_type_key, groupby)
    x = _get_expression_matrix(adata, layer, use_highly_variable=use_highly_variable)
    labels, mask = _get_labels_and_mask(adata, cell_type_key)

    if groupby is None:
        div, params = _compute_global(x, mask, labels, order)
        adata.uns[key_added] = div
        if params:
            adata.uns[f"{key_added}_params"] = {
                "order": order,
                "cell_type_key": cell_type_key,
                "layer": layer,
                **params,
            }
    else:
        groups = pd.Series(adata.obs[groupby])
        group_divs, params = _compute_grouped(
            x, mask, labels, order, groups,
            per_group_similarity=per_group_similarity,
        )
        adata.uns[key_added] = group_divs
        adata.obs[key_added] = groups.map(group_divs).to_numpy()
        if cell_type_key is not None:
            adata.uns[f"{key_added}_params"] = {
                "order": order,
                "cell_type_key": cell_type_key,
                "groupby": groupby,
                "layer": layer,
                "per_group_similarity": per_group_similarity,
                **params,
            }


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
