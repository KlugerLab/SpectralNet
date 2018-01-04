from __future__ import division
import numpy as np
cimport numpy as np
DTYPE = np.int32
ctypedef np.int32_t DTYPE_t

# @cython.boundscheck(False) 
# @cython.wraparound(False)
def convert_idxs(np.ndarray[DTYPE_t, ndim=2] orig_kn_idxs, np.ndarray[DTYPE_t] orig_p, int k, int n):
    print("in convert_idxs!")
    assert orig_kn_idxs.dtype == DTYPE and orig_p.dtype == DTYPE
    # make a copy of the first n elements of p
    cdef np.ndarray[DTYPE_t] p = orig_p[:n]
    cdef np.ndarray[DTYPE_t, ndim=2] kn_idxs = orig_kn_idxs[p]
    print('created arrays')

    cdef int len_idxs = orig_kn_idxs.shape[0]
    cdef int wid_idxs = orig_kn_idxs.shape[1]
    # cdef int j = 0
    cdef int i, j, num_neighbs
    cdef DTYPE_t kn_idx
    cdef np.ndarray[DTYPE_t, ndim=2] new_kn_idxs = np.empty((n, k + 1), dtype=DTYPE)
    cdef np.ndarray[np.int8_t] p_set = np.zeros(len_idxs, dtype=np.int8)
    cdef np.ndarray[DTYPE_t] convert = np.full(len_idxs, -1, dtype=np.int32)
    print('created vars')

    # build convert
    for i in range(n):
        convert[p[i]] = i

    # build p_set
    for i in range(n):
        p_set[p[i]] = 1

    for i in range(n):
        j = 0
        num_neighbs = 0
        while num_neighbs <= k and j < wid_idxs:
            kn_idx = kn_idxs[i, j]
            if p_set[kn_idx] == 1:
                if convert[kn_idx] == -1:
                    print("ERROR")
                if kn_idx < 0 or convert[kn_idx] < 0:
                    print(kn_idx, convert[kn_idx])
                new_kn_idxs[i, num_neighbs] = convert[kn_idx]
                num_neighbs = num_neighbs + 1
            j = j + 1
        if num_neighbs < k:
            print("num_neighbs only {}/{} of k! had {} elems in kn_idxs[i]".format(num_neighbs, k, len(kn_idxs)))

        if i % 10000 == 0:
            print("converted {}/{} indices".format(i, n))

    return new_kn_idxs[:n]

