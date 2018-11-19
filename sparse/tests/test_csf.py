import numpy as np
from sparse import COO
from sparse import CSF
from sparse.utils import assert_eq


def test_coo_to_csf():
    # Test data from Fig. 2 IPDPS'17 paper
    # http://shaden.io/pub-files/smith2017knl.pdf
    inds = [[0, 0, 0, 0, 0, 1, 1, 1],
            [0, 0, 1, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 1, 1, 1, 1],
            [1, 2, 0, 2, 0, 0, 1, 2]]
    vals = [1., 2., 3., 4., 5., 6., 7., 8.]
    coo = COO(coords=inds, data=vals)
    csf = CSF(coo)

    assert(coo.ndim == csf.ndim)
    assert_eq(csf.indptrs[0], np.array([0, 2, 3]))
    assert_eq(csf.indices[0], np.array([0, 1]))

    assert_eq(csf.indptrs[1], np.array([0, 1, 3, 4]))
    assert_eq(csf.indices[1], np.array([0, 1, 1]))

    assert_eq(csf.indptrs[2], np.array([0, 2, 4, 5, 8]))
    assert_eq(csf.indices[2], np.array([0, 0, 1, 1]))

    assert_eq(coo.data, csf.data)
    assert_eq(coo.coords[coo.ndim-1], csf.indices[csf.ndim-1])
