# scdiv

Similarity-sensitive diversity measures for transcriptomics data.

`scdiv` computes diversity of cell populations, accounting for the similarity
between cell types. It works with raw numpy arrays or plugs directly into
[scanpy](https://scanpy.readthedocs.io/) via AnnData objects.

## Installation

```bash
pip install git+https://github.com/katherinebenjamin/scdiv.git
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

- `layer="raw"` — use a specific layer instead of `adata.X`
- `use_highly_variable=False` — use all genes (default is `True`, which
  requires `sc.pp.highly_variable_genes` to have been run)
- `per_group_similarity=True` — recompute the similarity matrix within each
  group rather than using a global one
- `key_added="my_key"` — customise the storage key (useful when computing
  diversity at multiple orders)

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
