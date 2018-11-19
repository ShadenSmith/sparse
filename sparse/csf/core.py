import copy as _copy

import numpy as np


class CSF():
    """
    A sparse multidimensional array in the compressed sparse fiber format.
    """

    def __init__(self, coo):
        self.ndim = coo.ndim
        self.shape = coo.shape
        self.nnz = coo.nnz
        self._build_csf_tree(coo)

    def _build_csf_tree(self, coo):
        # The values can be copied directly.
        self.data = np.array(coo.data)

        # Defines the sparsity structure of the sparse tensor. `indptrs` is not
        # needed for the last mode.
        self.indptrs = [0] * (self.ndim - 1)
        self.indices = [0] * self.ndim

        # The length of indptrs.
        self.indlen = np.zeros(self.ndim, dtype=np.intp)

        # The indices of the last mode can simply be copied.
        self.indices[-1] = _copy.deepcopy(coo.coords[self.ndim-1])

        # The first mode is easier than internal ones.
        self.indlen[0] = self.shape[0]
        self.indptrs[0] = np.zeros(self.shape[0] + 1, dtype=np.intp)
        self.indices[0] = np.zeros(self.shape[0], dtype=np.intp)

        # Scan through the indices and find the bounds of each slice.
        nnz_ptr = 0
        for slice_id in range(self.shape[0]):
            while (nnz_ptr < self.nnz) and (coo.coords[0][nnz_ptr] == slice_id):
                nnz_ptr += 1
            self.indptrs[0][slice_id+1] = nnz_ptr
            self.indices[0][slice_id] = slice_id

        for mode in range(1, self.ndim - 1):
            # We will edit this to point to the indptr that we are constructing now.
            indprev = self.indptrs[mode-1]
            indcurr = coo.coords[mode]
            num_slices = self.indlen[mode-1]

            num_fibs = 0
            # Count the size of this level in the CSF tree.
            # foreach 'slice' in previous mode (actually a hyperplane)
            for slice_id in range(num_slices):
                num_fibs += 1
                for nnz_idx in range(indprev[slice_id]+1, indprev[slice_id+1]):
                    if indcurr[nnz_idx] != indcurr[nnz_idx-1]:
                        num_fibs += 1

            # Allocate storage
            self.indlen[mode] = num_fibs
            self.indptrs[mode] = np.zeros((num_fibs+1), dtype=np.intp)
            self.indices[mode] = np.zeros(num_fibs, dtype=np.intp)

            # save typing
            indptr = self.indptrs[mode]
            inds = self.indices[mode]

            num_fibs = 0
            # foreach 'slice' in previous mode
            for slice_id in range(num_slices):
                start = indprev[slice_id] + 1
                end = indprev[slice_id + 1]

                # Fill in start of subtree
                indprev[slice_id] = num_fibs
                indptr[num_fibs] = start - 1
                inds[num_fibs] = indcurr[start - 1]
                num_fibs += 1

                # Fill in the rest of the subtree
                for nnz_idx in range(start, end):
                    if indcurr[nnz_idx] != indcurr[nnz_idx-1]:
                        indptr[num_fibs] = nnz_idx
                        inds[num_fibs] = indcurr[nnz_idx]
                        num_fibs += 1

                # Mark end of last hyperplane
                indprev[num_slices] = num_fibs
                indptr[num_fibs] = self.nnz

    def __str__(self):
        return '<CSF: shape={!s}, nnz={:d}>'.format(
            self.shape, self.nnz
        )
