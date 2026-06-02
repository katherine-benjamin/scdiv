"""Shared type aliases for scdiv."""

import numpy.typing as npt
import scipy.sparse

Matrix = npt.NDArray | scipy.sparse.sparray | scipy.sparse.spmatrix
