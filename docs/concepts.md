# Concepts

<!--
This page is a scaffold. Fill in the prose; the section structure and
references below are placeholders to be expanded.

`dollarmath` is enabled, so inline math like $D_q^Z(p)$ and display
math via $$...$$ blocks both work.
-->

## Similarity-sensitive diversity

<!--
TODO: motivate the problem. Why naive Shannon / Simpson / richness
under-count diversity when types are similar; what Leinster-Cobbold
adds. One short example (e.g. two near-identical cell types vs. two
distant ones) goes a long way.
-->

## The diversity order $q$

<!--
TODO: explain the role of q. q=0 (richness), q=1 (Shannon-like),
q=2 (Simpson-like), q -> infinity (Berger-Parker-like).
What does q mean for ordering by "common-ness sensitivity"?
-->

## The similarity matrix $Z$

<!--
TODO: explain how Z is constructed in scdiv:
- Cell-type mode: cosine similarity of mean expression per cell type.
- Singleton mode: each cell its own type, factored cosine identity
  used to avoid materializing the n x n matrix.
Cross-link to `scdiv.similarity.cell_type_similarity` and friends.
-->

## Partition diversity

<!--
TODO: introduce Reeve et al. partition framework. Define
subcommunities, weights w_j, the three modes:

- `alpha_norm`: standalone Leinster-Cobbold diversity of each
  subcommunity.
- `alpha`: alpha_norm divided by w_j (a "diversity share").
- `gamma`: each subcommunity's contribution to the pooled
  metacommunity diversity.

Refer back to the per-mode behavior documented in
`scdiv.tl.diversity`.
-->

## Spatial diversity

<!--
TODO: explain hex/square partitioning of spatial transcriptomics
data and how per-region diversity is computed. Mention pseudo-cells.
Cross-link to `scdiv.spatial.partition`,
`scdiv.spatial.pseudo_cells`, `scdiv.spatial.diversity`.
-->

## References

<!--
TODO: full citations. Placeholders:
-->

- Leinster, T., & Cobbold, C. A. (2012). [Measuring diversity: the
  importance of species similarity](https://doi.org/10.1890/10-2402.1).
  *Ecology*, 93(3), 477–489.
- Reeve, R., Leinster, T., Cobbold, C. A., Thompson, J., Brummitt, N.,
  Mitchell, S. N., & Matthews, L. (2016). [How to partition
  diversity](https://arxiv.org/abs/1404.6520). *arXiv:1404.6520*.
