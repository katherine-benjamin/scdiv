import marimo

__generated_with = "0.23.5"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Spatial diversity in mouse hippocampus (Slide-seqV2)

    Welcome to the `scdiv` spatial tutorial! We'll be working through a [Slide-seqV2][1] data set on the mouse hippocampus.

    Pipeline:

    1. Aggregate beads into pseudo-cells with `scdiv.spatial.pseudo_cells` then denoise.
    3. Partition the slide into hex regions with `scdiv.spatial.partition`.
    4. Compute per-region diversity with `scdiv.tl.diversity`.
    5. Visualise diversity *in situ*.
    6. Check some health diagnostics.

    Install: `pip install "scdiv[examples]"`.

    This notebook is a [marimo][2] notebook. You can run it with `marimo edit examples/slideseq_hippocampus.py` to get interactive features.

    [1]: https://www.nature.com/articles/s41587-020-0739-1
    [2]: https://marimo.io
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Imports and helper functions

    No need to read these unless you're curious :)
    """)
    return


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import scanpy as sc
    import squidpy as sq
    from scipy import sparse
    from sklearn.neighbors import NearestNeighbors

    import scdiv

    return NearestNeighbors, mo, np, plt, sc, scdiv, sparse, sq


@app.cell(hide_code=True)
def _(NearestNeighbors, np, plt, sc, sparse):
    def cluster_scatter(
        ax, adata, cluster_key="cluster", *, cmap="tab20", **kwargs
    ):
        """Categorical scatter coloured by a cluster column."""
        xy = adata.obsm["spatial"]
        cats = adata.obs[cluster_key].cat.categories
        for i, cat in enumerate(cats):
            mask = (adata.obs[cluster_key] == cat).to_numpy()
            ax.scatter(
                xy[mask, 0],
                xy[mask, 1],
                color=plt.get_cmap(cmap)(i / max(len(cats) - 1, 1)),
                label=cat,
                linewidths=0,
                **kwargs,
            )


    def knn_smooth(adata, *, n_neighbors=15, n_pcs=50, key_added="smoothed"):
        """Average each cell with its k nearest neighbours in PC space.

        Writes the smoothed expression matrix to ``adata.layers[key_added]``.
        """
        sc.pp.pca(adata, n_comps=n_pcs)
        _, idx = (
            NearestNeighbors(n_neighbors=n_neighbors)
            .fit(adata.obsm["X_pca"])
            .kneighbors(adata.obsm["X_pca"])
        )
        n = adata.n_obs
        avg = sparse.csr_matrix(
            (
                np.full(n * n_neighbors, 1.0 / n_neighbors),
                (np.repeat(np.arange(n), n_neighbors), idx.ravel()),
            ),
            shape=(n, n),
        )
        adata.layers[key_added] = sparse.csr_matrix(avg @ adata.X.toarray())

    return cluster_scatter, knn_smooth


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Load the slide

    We'll use `squidpy` to load in our data set.
    """)
    return


@app.cell
def _(sq):
    adata = sq.datasets.slideseqv2()
    adata
    return (adata,)


@app.cell(hide_code=True)
def _(adata, cluster_scatter, plt):
    def plot_clusters():
        fig, ax = plt.subplots(figsize=(7, 7))
        cluster_scatter(ax, adata, s=1.2)
        ax.set_aspect("equal")
        ax.set_axis_off()
        ax.set_title("Slide-seqV2 mouse hippocampus — published clusters")
        ax.legend(
            loc="center left",
            bbox_to_anchor=(1.0, 0.5),
            fontsize=8,
            frameon=False,
            markerscale=8,
        )
        fig.tight_layout()
        return fig


    plot_clusters()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Build pseudo-cells and denoise

    Slide-seqV2 beads are sparse, so here we bulk to pseudo-cells. Even then it's still often necessary to denoise (see the **Diagnostic** section below), so here we use a simple smoothing on nearest neighbours in PCA space.
    """)
    return


@app.cell
def _(adata, knn_smooth, scdiv):
    PSEUDO_CELL_SIZE = 40.0  # μm — circumradius of fine hex bins

    ad_pc = scdiv.spatial.pseudo_cells(
        adata,
        method="hex",
        region_size=PSEUDO_CELL_SIZE,
        min_cells=5,
    )

    knn_smooth(ad_pc, n_neighbors=15, n_pcs=50)

    f"{ad_pc.n_obs} pseudo-cells | median {int(ad_pc.obs['n_cells'].median())} beads each"
    return PSEUDO_CELL_SIZE, ad_pc


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Partition pseudo-cells into regions

    `scdiv.spatial` offers a `partition` function that automatically tiles your slide into regions. Low cell pseudo-cell count regions (<50 pseudo cells or <25% total pseudo-cell coverage by area) are dropped.

    If you're running this notebook interactively then you can move the slider below to adjust the region size and see how things change!
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    region_size_slider = mo.ui.slider(
        start=150,
        stop=500.0,
        step=25.0,
        value=250.0,
        label="Region radius (μm)",
        show_value=True,
    )
    region_size_slider
    return (region_size_slider,)


@app.cell
def _(ad_pc, region_size_slider, scdiv):
    scdiv.spatial.partition(
        ad_pc,
        method="hex",
        region_size=float(region_size_slider.value),
        min_cells=20,
        cell_radius="auto",
        min_density=0.25,
    )
    n_regions = int(ad_pc.obs["spatial_region"].cat.categories.size)
    n_regions
    return (n_regions,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Per-region diversity

    Once you have your regions from `partition`, each diversity computation is a single call to `scdiv.tl.diversity`.

    Per region, we compute two flavours of diversity:

    - **alpha**: standalone diversity of the region: how much
      transcriptomic heterogeneity it carries internally.
    - **gamma**: the region's contribution to the slide's overall diversity, i.e. how much "new" diversity it adds beyond what the rest of the slide already covers.

    We're computing diversity in the default singleton mode here, by not supplying any cell type annotation.
    """)
    return


@app.cell
def _(ad_pc, n_regions, scdiv):
    _ = n_regions  # Notebook dependency boilerplate


    # Compute alpha diversity
    scdiv.tl.diversity(
        ad_pc,
        order=2,
        groupby="spatial_region",
        cell_type_key=None,
        mode="alpha_norm",
        layer="smoothed",
        use_highly_variable=False,
        key_added="scdiv_alpha",
    )

    # Compute gamma diversity
    scdiv.tl.diversity(
        ad_pc,
        order=2,
        groupby="spatial_region",
        cell_type_key=None,
        mode="gamma",
        layer="smoothed",
        use_highly_variable=False,
        key_added="scdiv_gamma",
    )

    diversity_ranges = {
        k: (min(ad_pc.uns[k].values()), max(ad_pc.uns[k].values()))
        for k in ("scdiv_alpha", "scdiv_gamma")
    }
    diversity_ranges
    return (diversity_ranges,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Plotting

    We can use `scdiv.pl.diversity_heatmap()` to quickly plot the diversity scores by region.
    """)
    return


@app.cell(hide_code=True)
def _(ad_pc, diversity_ranges, plt, scdiv):
    _keys = ["scdiv_alpha", "scdiv_gamma"]
    _titles = ["Alpha", "Gamma"]


    def plot_diversity_side_by_side():

        fig, axes = plt.subplots(1, 2, figsize=(14, 7))

        for i, ax in enumerate(axes):
            vmin, vmax = diversity_ranges[_keys[i]]
            scdiv.pl.diversity_heatmap(
                ad_pc, key=_keys[i], ax=ax, vmin=vmin, vmax=vmax
            )
            ax.set_axis_off()
            ax.set_title(_titles[i])

        return fig


    plot_diversity_side_by_side()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We can also overlay the diversity plot on top of the cluster map.

    If you're running this notebook interactively then you can use the dropdown below to change the overlay.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    panel_pick = mo.ui.dropdown(
        options={"Alpha": "scdiv_alpha", "Gamma": "scdiv_gamma"},
        value="Alpha",
        label="Overlay",
    )
    panel_pick
    return (panel_pick,)


@app.cell(hide_code=True)
def _(
    ad_pc,
    adata,
    cluster_scatter,
    diversity_ranges,
    panel_pick,
    plt,
    region_size_slider,
    scdiv,
):
    _ = diversity_ranges

    _fig, (_ax_ann, _ax_div) = plt.subplots(
        1,
        2,
        figsize=(16, 8),
        sharex=True,
        sharey=True,
    )
    cluster_scatter(_ax_ann, adata, s=5)
    _ax_ann.set_title("Cluster annotation")

    cluster_scatter(_ax_div, adata, s=1.5, alpha=1.0)
    scdiv.pl.diversity_heatmap(
        ad_pc,
        key=panel_pick.value,
        ax=_ax_div,
        edgecolors="white",
        linewidths=0.4,
        alpha=0.45,
    )
    _ax_div.set_title(
        f"{panel_pick.selected_key}  |  region_size={region_size_slider.value} μm"
    )

    for _ax in (_ax_ann, _ax_div):
        _ax.set_aspect("equal")
        _ax.set_axis_off()

    _fig.legend(
        *_ax_ann.get_legend_handles_labels(),
        loc="lower center",
        ncol=7,
        bbox_to_anchor=(0.5, -0.02),
        fontsize=8,
        frameon=False,
        markerscale=4,
    )
    _fig.tight_layout(rect=[0, 0.06, 1, 1])
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Diagnostic: pseudo-cell sparsity

    If data are too noisy (pseudo-cells too small, regions too small, not enough smoothing, etc) then a key diagnostic is the alpha-diversity vs sparsity graph. If correlation between alpha-diversity and sparsity is unusually high then consider increasing pseudo-cell size, region size, or denoising strength. Some correlation is fine.
    """)
    return


@app.cell
def _(ad_pc, diversity_ranges, n_regions, plt, scdiv):
    _ = n_regions
    _ = diversity_ranges

    # Compute sparsity data
    scdiv.tl.sparsity(ad_pc, region_key="spatial_region", key_added="zero_frac")

    _fig, (_map, _scatter) = plt.subplots(1, 2, figsize=(14, 6))

    # Plot sparsity heatmap
    scdiv.pl.diversity_heatmap(
        ad_pc,
        key="zero_frac",
        ax=_map,
        cmap="cividis",
        colorbar_label="Mean zero fraction",
        edgecolors="none",
    )
    _map.set_axis_off()
    _map.set_title(f"Pseudo-cell sparsity")


    # Plot sparsity vs alpha diversity
    scdiv.pl.diversity_vs_metric(
        ad_pc, x_key="zero_frac", key="scdiv_alpha", ax=_scatter
    )
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(ad_pc, adata, cluster_scatter, diversity_ranges, plt, scdiv):
    import seaborn as sns
    import matplotlib as mpl

    sns.set_style("whitegrid")
    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.titlesize": 12,
            "axes.labelsize": 12,
            "legend.fontsize": 9,
            "text.usetex": True,
            "text.latex.preamble": r"\usepackage{underscore}",
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman"],
        }
    )

    mm = 1 / 25.4
    _fig = plt.figure(figsize=(179 * mm, 195 * mm))
    _gs = _fig.add_gridspec(
        3,
        2,
        width_ratios=[1.7, 1],
        height_ratios=[1.15, 1.0, 0.06],
        hspace=0.15,
        wspace=0.02,
    )

    _ax_clust = _fig.add_subplot(_gs[0, 0])
    _ax_legend = _fig.add_subplot(_gs[0, 1])
    _ax_legend.set_axis_off()
    cluster_scatter(_ax_clust, adata, s=1.6)
    _ax_clust.set_aspect("equal")
    _ax_clust.set_anchor("E")
    _ax_clust.set_axis_off()
    _ax_clust.set_title("Cluster annotation")

    _handles, _labels = _ax_clust.get_legend_handles_labels()
    _ax_legend.legend(
        _handles,
        _labels,
        loc="center left",
        ncol=1,
        fontsize=9,
        frameon=False,
        markerscale=5,
    )

    _mid = _gs[1, :].subgridspec(1, 2, wspace=0.15)
    _bot = _gs[2, :].subgridspec(1, 2, wspace=0.15)
    _ax_a = _fig.add_subplot(_mid[0])
    _ax_g = _fig.add_subplot(_mid[1])
    _cax_a = _fig.add_subplot(_bot[0])
    _cax_g = _fig.add_subplot(_bot[1])

    _vmin_a, _vmax_a = diversity_ranges["scdiv_alpha"]
    _vmin_g, _vmax_g = diversity_ranges["scdiv_gamma"]

    scdiv.pl.diversity_heatmap(
        ad_pc,
        key="scdiv_alpha",
        ax=_ax_a,
        vmin=_vmin_a,
        vmax=_vmax_a,
        colorbar=False,
    )
    scdiv.pl.diversity_heatmap(
        ad_pc,
        key="scdiv_gamma",
        ax=_ax_g,
        vmin=_vmin_g,
        vmax=_vmax_g,
        colorbar=False,
    )
    _ax_a.set_aspect("equal")
    _ax_a.set_axis_off()
    _ax_a.set_title(r"$\alpha$-diversity")
    _ax_g.set_aspect("equal")
    _ax_g.set_axis_off()
    _ax_g.set_title(r"$\gamma$-diversity")

    _fig.colorbar(
        mpl.cm.ScalarMappable(
            norm=mpl.colors.Normalize(vmin=_vmin_a, vmax=_vmax_a), cmap="viridis"
        ),
        cax=_cax_a,
        orientation="horizontal",
        label=r"$\alpha$-diversity",
    )
    _fig.colorbar(
        mpl.cm.ScalarMappable(
            norm=mpl.colors.Normalize(vmin=_vmin_g, vmax=_vmax_g), cmap="viridis"
        ),
        cax=_cax_g,
        orientation="horizontal",
        label=r"$\gamma$-diversity",
    )

    for _ax, _tag in [(_ax_clust, "(a)"), (_ax_a, "(b)"), (_ax_g, "(c)")]:
        _ax.text(
            -0.02,
            1.02,
            _tag,
            transform=_ax.transAxes,
            fontsize=14,
            fontweight="bold",
            ha="right",
            va="bottom",
        )

    _fig.savefig("hippocampus_diversity.pdf", bbox_inches="tight")
    _fig
    return


@app.cell
def _(adata, np, plt):
    _genes = ["Ttr", "Folr1", "Enpp2"]
    _xy = adata.obsm["spatial"]
    _X = adata[:, _genes].X
    if hasattr(_X, "toarray"):
        _X = _X.toarray()

    _fig, _axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)
    for _ax, _g, _col in zip(_axes, _genes, _X.T):
        _order = np.argsort(_col)
        _sc = _ax.scatter(
            _xy[_order, 0],
            _xy[_order, 1],
            c=_col[_order],
            s=1.0,
            cmap="magma",
            linewidths=0,
        )
        _ax.set_aspect("equal")
        _ax.set_axis_off()
        _ax.set_title(_g)
        _fig.colorbar(_sc, ax=_ax, fraction=0.04, pad=0.02)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Summary statistics

    A quick recap of the key numbers for this dataset.
    """)
    return


@app.cell(hide_code=True)
def _(PSEUDO_CELL_SIZE, ad_pc, adata, n_regions, np, sparse):
    from scipy.spatial import cKDTree

    # Beads
    _n_beads = adata.n_obs
    _n_genes = adata.n_vars
    _n_clusters = len(adata.obs["cluster"].cat.categories)

    _X = adata.X.toarray() if sparse.issparse(adata.X) else adata.X
    _umis_per_bead = _X.sum(axis=1)
    _total_umis = int(_umis_per_bead.sum())
    _median_umis = float(np.median(_umis_per_bead))
    _mean_umis = float(_umis_per_bead.mean())

    _xy = adata.obsm["spatial"]
    _extent_x = _xy[:, 0].max() - _xy[:, 0].min()
    _extent_y = _xy[:, 1].max() - _xy[:, 1].min()

    # Bead pitch from nearest-neighbour distance
    _d, _ = cKDTree(_xy).query(_xy, k=2)
    _bead_pitch = float(np.median(_d[:, 1]))

    # Pseudo-cells
    _n_pseudo = ad_pc.n_obs
    _beads_per_pc = ad_pc.obs["n_cells"]

    # Regions
    _pc_per_region = ad_pc.obs["spatial_region"].value_counts()
    _region_size = float(ad_pc.uns["spatial_region_params"]["region_size"])

    print("Slide-seqV2 puck (raw beads)")
    print("-" * 40)
    print(f"  number of beads        : {_n_beads:>8,}")
    print(f"  number of genes        : {_n_genes:>8,}")
    print(f"  number of clusters     : {_n_clusters:>8}")
    print(f"  total UMIs             : {_total_umis:>8,}")
    print(f"  UMIs per bead (median) : {_median_umis:>8.0f}")
    print(f"  UMIs per bead (mean)   : {_mean_umis:>8.1f}")
    print(f"  bead pitch (median NN) : {_bead_pitch:>8.2f} um")
    print(f"  spatial extent         : {_extent_x:.0f} x {_extent_y:.0f} um")
    print()
    print("Pseudo-cells (after scdiv.spatial.pseudo_cells + smoothing)")
    print("-" * 40)
    print(f"  number of pseudo-cells : {_n_pseudo:>8,}")
    print(
        f"  pseudo-cell radius     : {PSEUDO_CELL_SIZE:>8.0f} um  (hex circumradius)"
    )
    print(
        f"  beads per pseudo-cell  :   median {int(_beads_per_pc.median())}, "
        f"mean {_beads_per_pc.mean():.1f}, range {_beads_per_pc.min()}-{_beads_per_pc.max()}"
    )
    print(f"  HVGs used              :     none  (all {_n_genes} genes)")
    print(f"  KNN smoothing          :     k=15 neighbours in 50 PC space")
    print()
    print("Regions (after scdiv.spatial.partition)")
    print("-" * 40)
    print(f"  number of regions      : {n_regions:>8}")
    print(
        f"  region radius          : {_region_size:>8.0f} um  (hex circumradius)"
    )
    print(
        f"  pseudo-cells / region  :   median {int(_pc_per_region.median())}, "
        f"mean {_pc_per_region.mean():.1f}, range {_pc_per_region.min()}-{_pc_per_region.max()}"
    )
    return


if __name__ == "__main__":
    app.run()
