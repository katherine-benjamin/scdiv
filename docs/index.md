# scdiv

Similarity-sensitive diversity measures for transcriptomics data.

`scdiv` computes diversity of cell populations, accounting for the similarity
between cell types. It works with raw numpy arrays or plugs directly into
[scanpy](https://scanpy.readthedocs.io/) via AnnData objects.

## Installation

```bash
pip install git+https://github.com/katherinebenjamin/scdiv.git
```

## Quick start

```python
import scdiv

scdiv.tl.diversity(adata, order=1, cell_type_key="cell_type")
adata.uns["scdiv_diversity"]  # the diversity score
```

```{toctree}
:maxdepth: 2
:hidden:

tutorials/index
api
```
