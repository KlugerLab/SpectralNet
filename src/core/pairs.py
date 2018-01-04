'''
pairs.py: contains functions used for creating pairs from labeled and unlabeled data (currently used only for the siamese network)
'''

from __future__ import absolute_import
from __future__ import print_function
import numpy as np
import math
import random
import pickle
import h5py

from random import randint
from collections import defaultdict
from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input, Lambda, Flatten
from keras.optimizers import RMSprop
from keras import backend as K
from sklearn.neighbors import NearestNeighbors, LSHForest
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import LearningRateScheduler

from sklearn import metrics
from annoy import AnnoyIndex

##### Helper functions #####
def get_choices(arr, num_choices, valid_range=[-1, np.inf], not_arr=None, replace=False):
    '''
    Select n=num_choices choices from arr, with the following constraints for
    each choice:
        choice > valid_range[0],
        choice < valid_range[1],
        choice not in not_arr
    if replace == True, draw choices with replacement
    if arr is an integer, the pool of choices is interpreted as [0, arr]
    (inclusive)
        * in the implementation, we use an identity function to create the
        identity map arr[i] = i
    '''
    if not_arr is None:
        not_arr = []
    if isinstance(valid_range, int):
        valid_range = [0, valid_range]
    # make sure we have enough valid points in arr
    if isinstance(arr, tuple):
        if min(arr[1], valid_range[1]) - max(arr[0], valid_range[0]) < num_choices:
            raise ValueError("Not enough elements in arr are outside of valid_range!")
        n_arr = arr[1]
        arr0 = arr[0]
        arr = defaultdict(lambda: -1)
        get_arr = lambda x: x
        replace = True
    else:
        greater_than = np.array(arr) > valid_range[0]
        less_than = np.array(arr) < valid_range[1]
        if np.sum(np.logical_and(greater_than, less_than)) < num_choices:
            raise ValueError("Not enough elements in arr are outside of valid_range!")
        # make a copy of arr, since we'll be editing the array
        n_arr = len(arr)
        arr0 = 0
        arr = np.array(arr, copy=True)
        get_arr = lambda x: arr[x]
    not_arr_set = set(not_arr)
    def get_choice():
        arr_idx = randint(arr0, n_arr-1)
        while get_arr(arr_idx) in not_arr_set:
            arr_idx = randint(arr0, n_arr-1)
        return arr_idx
    if isinstance(not_arr, int):
        not_arr = list(not_arr)
    choices = []
    for _ in range(num_choices):
        arr_idx = get_choice()
        while get_arr(arr_idx) <= valid_range[0] or get_arr(arr_idx) >= valid_range[1]:
            arr_idx = get_choice()
        choices.append(int(get_arr(arr_idx)))
        if not replace:
            arr[arr_idx], arr[n_arr-1] = arr[n_arr-1], arr[arr_idx]
            n_arr -= 1
    return choices

def create_pairs_from_labeled_data(x, digit_indices, use_classes=None):
    '''
    Positive and negative pair creation from labeled data.
    Alternates between positive and negative pairs.

    digit_indices:  nested array of depth 2 (in other words a jagged
                    matrix), where row i contains the indices in x of
                    all examples labeled with class i
    use_classes:    in cases where we only want pairs from a subset
                    of the classes, use_classes is a list of the
                    classes to draw pairs from, else it is None
    '''
    n_clusters = len(digit_indices)
    if use_classes == None:
        use_classes = list(range(n_clusters))
    if not isinstance(use_classes, list):
        raise Exception("use_classes must be None or a list of integer indices!")
    pairs = []
    labels = []
    n = min([len(digit_indices[d]) for d in range(n_clusters)]) - 1
    for d in use_classes:
        for i in range(n):
            z1, z2 = digit_indices[d][i], digit_indices[d][i + 1]
            pairs += [[x[z1], x[z2]]]
            inc = random.randrange(1, n_clusters)
            dn = (d + inc) % n_clusters
            z1, z2 = digit_indices[d][i], digit_indices[dn][i]
            pairs += [[x[z1], x[z2]]]
            labels += [1, 0]
    pairs = np.array(pairs).reshape((len(pairs), 2) + x.shape[1:])
    labels = np.array(labels)
    return pairs, labels

def create_pairs_from_unlabeled_data(x1, x2=None, y=None, p=None, k=5, tot_pairs=None, precomputed_knn_path='', use_approx=False, pre_shuffled=False, verbose=None):
    '''
    Generates positive and negative pairs for the siamese network from
    unlabeled data. Draws from the k nearest neighbors (where k is the
    provided parameter) of each point to form pairs. Number of neighbors
    to draw is determined by tot_pairs, if provided, or k if not provided.

    x1: input data array
    x2: parallel data array (pairs will exactly shadow the indices of x1,
        but be drawn from x2)
    y:  true labels (if available) purely for checking how good our pairs are
    p:  permutation vector - in cases where the array is shuffled and we
        use a precomputed knn matrix (where knn is performed on unshuffled
        data), we keep track of the permutations with p, and apply the same
        permutation to the precomputed knn matrix
    k:  the number of neighbors to use (the 'k' in knn)

    tot_pairs:              total number of pairs to produce
    precomputed_knn_path:   location of stored precomputed knn results -
                            empty string means we do not load precomputed
                            neighbors
    use_approx:             flag for running with LSH instead of KNN, in other
                            words, an approximation of KNN
    verbose:                flag for extra debugging printouts

    returns:    pairs for x1, (pairs for x2 if x2 is provided), labels
                (inferred by knn), (labels_true, the absolute truth, if y
                is provided
    '''
    if x2 is not None and x1.shape != x2.shape:
        raise ValueError("x1 and x2 must be the same shape!")

    n = len(p) if p is not None else len(x1)

    pairs_per_pt = max(1, min(k, int(tot_pairs/(n*2)))) if tot_pairs is not None else max(1, k)

    if p is not None and not pre_shuffled:
        x1 = x1[p[:n]]
        y = y[p[:n]]

    pairs = []
    pairs2 = []
    labels = []
    true = []
    verbose = True
    if len(precomputed_knn_path):
        # load precomputed weights
        if verbose:
            print('loading precomputed weights...')
            print('load path:', precomputed_knn_path)
        if precomputed_knn_path.endswith('.h5'):
            with h5py.File(precomputed_knn_path, 'r') as f:
                kn_idxs_untouched = np.asarray(f.get('kn_idxs'), dtype='uint32')
        else:
            kn_idxs_untouched = pickle.load(open(precomputed_knn_path, 'rb'))
        if isinstance(kn_idxs_untouched, tuple):
            kn_idxs_untouched = kn_idxs_untouched[1]

        assert (kn_idxs_untouched >= 0).all()
        if p is None:
            Idx = kn_idxs_untouched
        else:
            # if we have shuffled the array with p, we must convert our neighbors
            # matrix to correspond to the shuffled indices
            import pyximport; pyximport.install()
            from core.convert_idxs import convert_idxs
            Idx = convert_idxs(kn_idxs_untouched.astype(np.int32, copy=False), p.astype(np.int32, copy=False), k, n)
            print('converted all indices')

    else:
        if verbose:
            print('computing k={} nearest neighbors...'.format(k))
        if len(x1.shape)>2:
            x1_flat = x1.reshape(x1.shape[0], np.prod(x1.shape[1:]))[:n]
        else:
            x1_flat = x1[:n]

        if use_approx:
            ann = AnnoyIndex(x1_flat.shape[1], metric='euclidean')
            for i, x_ in enumerate(x1_flat):
                ann.add_item(i, x_)
            ann.build(50)
            Idx = np.empty((len(x1_flat), k+1))
            for i in range(len(x1_flat)):
                nn_i = ann.get_nns_by_item(i, k+1, include_distances=False)
                Idx[i,:] = np.array(nn_i)
        else:
            nbrs = NearestNeighbors(n_neighbors=k+1).fit(x1_flat)
            _, Idx = nbrs.kneighbors(x1_flat)

    # for each row, remove the element itself from its list of neighbors
    # (we don't care that each point is its own closest neighbor)
    new_Idx = np.empty((Idx.shape[0], Idx.shape[1] - 1))
    assert (Idx >= 0).all()
    for i in range(Idx.shape[0]):
        try:
            new_Idx[i] = Idx[i, Idx[i] != i][:Idx.shape[1] - 1]
        except Exception as e:
            print(Idx[i, ...], new_Idx.shape, Idx.shape)
            raise e
    Idx = new_Idx.astype(np.int)
    k_max = min(Idx.shape[1], k+1)

    if verbose:
        print('creating pairs...')
        print("ks", n, k_max, k, pairs_per_pt)

    # pair generation loop (alternates between true and false pairs)
    consecutive_fails = 0
    for i in range(n):
        # get_choices sometimes fails with precomputed results. if this happens
        # too often, we relax the constraint on k
        if consecutive_fails > 5:
            k_max = min(Idx.shape[1], int(k_max*2))
            consecutive_fails = 0
        if verbose and i % 10000 == 0:
            print("Iter: {}/{}".format(i,n))
        # pick points from neighbors of i for positive pairs
        try:
            choices = get_choices(Idx[i,:k_max], pairs_per_pt, replace=False)
            consecutive_fails = 0
        except ValueError:
            consecutive_fails += 1
            continue
        assert i not in choices
        # form the pairs
        new_pos = [[x1[i], x1[c]] for c in choices]
        if x2 is not None:
            new_pos2 = [[x2[i], x2[c]] for c in choices]
        if y is not None:
            pos_labels = [[y[i] == y[c]] for c in choices]
        # pick points *not* in neighbors of i for negative pairs
        try:
            choices = get_choices((0, n), pairs_per_pt, not_arr=Idx[i,:k_max], replace=False)
            consecutive_fails = 0
        except ValueError:
            consecutive_fails += 1
            continue
        # form negative pairs
        new_neg = [[x1[i], x1[c]] for c in choices]
        if x2 is not None:
            new_neg2 = [[x2[i], x2[c]] for c in choices]
        if y is not None:
            neg_labels = [[y[i] == y[c]] for c in choices]

        # add pairs to our list
        labels += [1]*len(new_pos) + [0]*len(new_neg)
        pairs += new_pos + new_neg
        if x2 is not None:
            pairs2 += new_pos2 + new_neg2
        if y is not None:
            true += pos_labels + neg_labels

    # package return parameters for output
    ret = [np.array(pairs).reshape((len(pairs), 2) + x1.shape[1:])]
    if x2 is not None:
        ret.append(np.array(pairs2).reshape((len(pairs2), 2) + x2.shape[1:]))
    ret.append(np.array(labels))
    if y is not None:
        true = np.array(true).astype(np.int).reshape(-1,1)
        if verbose:
            # if true vectors are provided, we can take a peek to check
            # the validity of our kNN approximation
            print("confusion matrix for pairs and approximated labels:")
            print(metrics.confusion_matrix(true, labels)/true.shape[0])
            print(metrics.confusion_matrix(true, labels))
        ret.append(true)

    return ret

