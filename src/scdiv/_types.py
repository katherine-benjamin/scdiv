"""Shared type aliases for scdiv."""

import numpy.typing as npt
import scipy.sparse

# Dense array or scipy sparse. ``spmatrix`` is the legacy class read_h5ad
# still returns; drop it once the ecosystem defaults to ``sparray``.
Matrix = npt.NDArray | scipy.sparse.sparray | scipy.sparse.spmatrix
