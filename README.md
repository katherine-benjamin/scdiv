# scdiv

Similarity-sensitive diversity measures for transcriptomics data.

`scdiv` computes diversity of cell populations, accounting for the similarity
between cell types. It works with raw numpy arrays or plugs directly into
[scanpy](https://scanpy.readthedocs.io/) via AnnData objects.

## Installation

```bash
pip install git+https://github.com/katherine-benjamin/scdiv.git
```

## Quick start with scanpy

If you have an AnnData object with cell type annotations and highly variable
genes already computed:

```python
import scdiv

scdiv.tl.diversity(adata, order=1, cell_type_key="cell_type")
adata.uns["scdiv_diversity"]  # the diversity score
```

### Per-sample diversity

Compute diversity separately for each sample (or batch, condition, etc.):

```python
scdiv.tl.diversity(adata, order=1, cell_type_key="cell_type", groupby="sample")

adata.uns["scdiv_diversity"]   # dict: {sample_name: diversity}
adata.obs["scdiv_diversity"]   # each cell gets its sample's diversity score
```

### Singleton mode

Treat each cell as its own type (no cell type annotations needed):

```python
scdiv.tl.diversity(adata, order=1)
```

### Options

- `layer="raw"`: use a specific layer instead of `adata.X`
- `obsm="X_smoothed"`: use a per-cell vector representation from
  `adata.obsm` instead of gene-space expression.
- `use_highly_variable=False`: use all genes (default is `True`, which
  requires `sc.pp.highly_variable_genes` to have been run). Ignored
  when `obsm` is set.
- `alpha=1.0` (default): sensitivity of the similarity measure to
  highly expressed genes. `1.0`: most sensitive to highly expressed genes,
  `0.0: least sensitive to highly expressed genes.
- `mode="alpha_norm"` (default), `"alpha"`, or `"gamma"`: partition
  diversity mode in the style of Reeve et al. (2016), used with `groupby`.
  `alpha_norm` is each subcommunity's standalone Leinster–Cobbold
  diversity; `alpha` is `alpha_norm / w_j` (a "diversity share" that can
  exceed `n_types`); `gamma` measures each subcommunity's ordinariness
  against the pooled metacommunity.
- `aggregate=True`: also store a single metacommunity-level scalar at
  `adata.uns[f"{key_added}_metacommunity"]` (the `w_j`-weighted power
  mean of order `1 - order` of the per-group values; for `gamma`, the
  diversity of the pooled distribution).
- `key_added="my_key"`: customise the storage key

### Spatial diversity

For spatial transcriptomics data, tile cells into square or hexagonal
regions and compute per-region diversity:

```python
scdiv.spatial.diversity(
    adata, order=1,
    partition_kwargs={"method": "hex", "region_size": 100},
    cell_type_key="cell_type", mode="alpha",
)

scdiv.pl.diversity_heatmap(adata)  # polygon map colored by diversity
```

Spatial coordinates are read from `adata.obsm["spatial"]`.
`partition_kwargs` forwards to `scdiv.spatial.partition` (knobs:
`method`, `region_size`, `min_cells`, `spatial_key`); remaining kwargs
forward to `scdiv.tl.diversity`.

## Numpy interface

For users who prefer to work with raw arrays:

```python
from scdiv.diversity import diversity_from_counts

# One-shot: count matrix + labels -> diversity
div = diversity_from_counts(X, labels, order=1)
```

Or step by step, if you want to inspect the intermediate results:

```python
from scdiv.similarity import cell_type_similarity
from scdiv.diversity import diversity, distribution_from_labels

sim, cell_types = cell_type_similarity(X, labels)
dist, cell_types = distribution_from_labels(labels)
div = diversity(sim, order=1, distribution=dist)
```
