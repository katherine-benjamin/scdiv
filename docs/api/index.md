# API reference

`scdiv` follows the scanpy-style module layout:

- **`scdiv.tl`**: tools that compute on AnnData and write results into
  `adata.uns` / `adata.obs`. Start here for typical scanpy workflows.
- **`scdiv.pl`**: plotting functions that visualize results from `scdiv.tl`.
- **`scdiv.spatial`**: spatial-transcriptomics helpers. Partition cells
  into regions, compute per-region diversity, build pseudo-cells.
- **`scdiv.diversity`** and **`scdiv.similarity`**: lower-level primitives
  that operate on numpy arrays directly. Use these when working outside
  an AnnData workflow.

```{toctree}
:maxdepth: 1

tl
pl
spatial
diversity
similarity
```
