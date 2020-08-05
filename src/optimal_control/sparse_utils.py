"""Sparse matrix utility functions. Many related to/from scipy.sparse funcs."""

from scipy.sparse import coo_matrix, issparse, bmat, eye, dok_matrix, vstack, csr_matrix  # pylint: disable=unused-import
import numpy as np

# used as exports
assert bmat
assert eye
assert dok_matrix
assert vstack
assert csr_matrix


def block_diag(mats, format='coo', dtype=None):  # pylint: disable=redefined-builtin
    """Much faster(O(n)) version of scipy.sparse.block_diag. Maintains API."""
    # dtype = np.dtype(dtype)
    # row = []
    # col = []
    # data = []
    # r_idx = 0
    # c_idx = 0
    # for mat in mats:
    #     if issparse(mat):
    #         mat = mat.tocsr()
    #     else:
    #         mat = coo_matrix(mat).tocsr()
    #     nrows, ncols = mat.shape
    #     for r in range(nrows):
    #         for c in range(ncols):
    #             if mat[r, c] is not None:
    #                 row.append(r + r_idx)
    #                 col.append(c + c_idx)
    #                 data.append(mat[r, c])
    #     r_idx = r_idx + nrows
    #     c_idx = c_idx + ncols
    # return coo_matrix((data, (row, col)), dtype=dtype).asformat(format)
    row = []
    col = []
    data = []
    r_idx = 0
    c_idx = 0
    for _, mat in enumerate(mats):
        if issparse(mat):
            mat = mat.tocoo()
        # if isinstance(mat, (list, numbers.Number)):
        #     mat = coo_matrix(mat)
        else:
            mat = coo_matrix(mat)
        nrows, ncols = mat.shape
        # for r, c in zip(*mat.nonzero()):
        #     row.append(r + r_idx)
        #     col.append(c + c_idx)
        #     data.append(mat[r, c])
        row.append(mat.row + r_idx)
        col.append(mat.col + c_idx)
        data.append(mat.data)
        r_idx = r_idx + nrows
        c_idx = c_idx + ncols
    row = np.concatenate(row)
    col = np.concatenate(col)
    data = np.concatenate(data)
    return coo_matrix((data, (row, col)),
                      shape=(r_idx, c_idx),
                      dtype=dtype).asformat(format)
