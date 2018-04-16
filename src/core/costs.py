'''
costs.py: contains all cost functions (and related helper functions) for spectral and siamese nets
'''

from keras import backend as K
import numpy as np
from keras.backend.tensorflow_backend import expand_dims
from sklearn.neighbors import NearestNeighbors
import tensorflow as tf

def squared_distance(X, Y=None, W=None):
    '''
    Calculates the pairwise distance between points in X and Y

    X:          n x d matrix
    Y:          m x d matrix
    W:          affinity -- if provided, we normalize the distance

    returns:    n x m matrix of all pairwise squared Euclidean distances
    '''
    if Y is None:
        Y = X
    # distance = squaredDistance(X, Y)
    sum_dimensions = list(range(2, K.ndim(X) + 1))
    X = K.expand_dims(X, axis=1)
    if W is not None:
        # if W provided, we normalize X and Y by W
        D_diag = K.expand_dims(K.sqrt(K.sum(W, axis=1)), axis=1)
        X /= D_diag
        Y /= D_diag
    squared_difference = K.square(X - Y)
    distance = K.sum(squared_difference, axis=sum_dimensions)
    return distance

def knn_affinity(X, n_nbrs, scale=None, scale_nbr=None, local_scale=None, verbose=False):
    '''
    Calculates the symmetrized Gaussian affinity matrix with k1 nonzero
    affinities for each point, scaled by
    1) a provided scale,
    2) the median distance of the k2th neighbor of each point in X, or
    3) a covariance matrix S where S_ii is the distance of the k2th
    neighbor of each point i, and S_ij = 0 for all i != j
    Here, k1 = n_nbrs, k2=scale_nbr

    X:              input dataset of size n
    n_nbrs:         k1
    scale:          provided scale
    scale_nbr:      k2, used if scale not provided
    local_scale:    if True, then we use the aforementioned option 3),
                    else we use option 2)
    verbose:        extra printouts

    returns:        n x n affinity matrix
    '''
    if isinstance(n_nbrs, np.float):
        n_nbrs = int(n_nbrs)
    elif isinstance(n_nbrs, tf.Variable) and n_nbrs.dtype.as_numpy_dtype != np.int32:
        n_nbrs = tf.cast(n_nbrs, np.int32)
    #get squared distance
    Dx = squared_distance(X)
    #calculate the top k neighbors of minus the distance (so the k closest neighbors)
    nn = tf.nn.top_k(-Dx, n_nbrs, sorted=True)

    vals = nn[0]
    # apply scale
    if scale is None:
        # if scale not provided, use local scale
        if scale_nbr is None:
            scale_nbr = 0
        else:
            print("getAffinity scale_nbr, n_nbrs:", scale_nbr, n_nbrs)
            assert scale_nbr > 0 and scale_nbr <= n_nbrs
        if local_scale:
            scale = -nn[0][:, scale_nbr - 1]
            scale = tf.reshape(scale, [-1, 1])
            scale = tf.tile(scale, [1, n_nbrs])
            scale = tf.reshape(scale, [-1, 1])
            vals = tf.reshape(vals, [-1, 1])
            if verbose:
                vals = tf.Print(vals, [tf.shape(vals), tf.shape(scale)], "vals, scale shape")
            vals = vals / (2*scale)
            vals = tf.reshape(vals, [-1, n_nbrs])
        else:
            def get_median(scales, m):
                with tf.device('/cpu:0'):
                    scales = tf.nn.top_k(scales, m)[0]
                scale = scales[m - 1]
                return scale, scales
            scales = -vals[:, scale_nbr - 1]
            const = tf.shape(X)[0] // 2
            scale, scales = get_median(scales, const)
            vals = vals / (2 * scale)
    else:
        # otherwise, use provided value for global scale
        vals = vals / (2*scale**2)

    #get the affinity
    affVals = tf.exp(vals)
    #flatten this into a single vector of values to shove in a spare matrix
    affVals = tf.reshape(affVals, [-1])
    #get the matrix of indexes corresponding to each rank with 1 in the first column and k in the kth column
    nnInd = nn[1]
    #get the J index for the sparse matrix
    jj = tf.reshape(nnInd, [-1, 1])
    #the i index is just sequential to the j matrix
    ii = tf.range(tf.shape(nnInd)[0])
    ii = tf.reshape(ii, [-1, 1])
    ii = tf.tile(ii, [1, tf.shape(nnInd)[1]])
    ii = tf.reshape(ii, [-1, 1])
    #concatenate the indices to build the sparse matrix
    indices = tf.concat((ii,jj),axis=1)
    #assemble the sparse Weight matrix
    W = tf.SparseTensor(indices=tf.cast(indices, dtype='int64'), values=affVals, dense_shape=tf.cast(tf.shape(Dx), dtype='int64'))
    #fix the ordering of the indices
    W = tf.sparse_reorder(W)
    #convert to dense tensor
    W = tf.sparse_tensor_to_dense(W)
    #symmetrize
    W = (W+tf.transpose(W))/2.0;

    return W

def full_affinity(X, scale):
    '''
    Calculates the symmetrized full Gaussian affinity matrix, scaled
    by a provided scale

    X:              input dataset of size n
    scale:          provided scale

    returns:        n x n affinity matrix
    '''
    sigma = K.variable(scale)
    Dx = squared_distance(X)
    sigma_squared = K.pow(sigma, 2)
    sigma_squared = K.expand_dims(sigma_squared, -1)
    Dx_scaled = Dx / (2 * sigma_squared)
    W = K.exp(-Dx_scaled)
    return W

def get_contrastive_loss(m_neg=1, m_pos=.2):
    '''
    Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    def contrastive_loss(y_true, y_pred):
        return K.mean(y_true * K.square(K.maximum(y_pred - m_pos, 0)) +
                    (1 - y_true) * K.square(K.maximum(m_neg - y_pred, 0)))

    return contrastive_loss

def get_triplet_loss(m=1):
    '''
    Triplet loss is defined as:
        L(A, P, N) = max(d(A, N) - d(A, P) + m, 0)
    where A is the anchor, and P, N are the positive and negative
    examples w.r.t. the anchor. To adapt this loss to the keras
    paradigm, we pre-compute y_diff = d(A, N) - d(A, P).
    NOTE: since each example includes a positive and a negative,
    we no longer use y_true
    '''
    def triplet_loss(_, y_diff):
        return K.mean(K.maximum(y_diff + m, 0))

    return triplet_loss

def euclidean_distance(vects):
    '''
    Computes the euclidean distances between vects[0] and vects[1]
    '''
    x, y = vects
    return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))

def eucl_dist_output_shape(shapes):
    '''
    Provides the output shape of the above computation
    '''
    s_1, _ = shapes
    return (s_1[0], 1)

