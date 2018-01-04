'''
spectralnet.py: contains run function for spectralnet
'''
import sys, os, pickle
import tensorflow as tf
import numpy as np
import traceback
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'

from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import normalized_mutual_info_score as nmi

import keras.backend as K
from keras.models import Model, load_model
from keras.layers import Input, Lambda
from keras.optimizers import RMSprop

from core import train
from core import costs
from core.layer import stack_layers
from core.util import get_scale, print_accuracy, get_cluster_sols, LearningHandler, make_layer_list, train_gen, get_y_preds

def run_net(data, params):
    #
    # UNPACK DATA
    #

    x_train, y_train, x_val, y_val, x_test, y_test = data['spectral']['train_and_test']
    x_train_unlabeled, y_train_unlabeled, x_train_labeled, y_train_labeled = data['spectral']['train_unlabeled_and_labeled']
    x_val_unlabeled, y_val_unlabeled, x_val_labeled, y_val_labeled = data['spectral']['val_unlabeled_and_labeled']

    if 'siamese' in params['affinity']:
        pairs_train, dist_train, pairs_val, dist_val = data['siamese']['train_and_test']

    x = np.concatenate((x_train, x_val, x_test), axis=0)
    y = np.concatenate((y_train, y_val, y_test), axis=0)

    if len(x_train_labeled):
        y_train_labeled_onehot = OneHotEncoder().fit_transform(y_train_labeled.reshape(-1, 1)).toarray()
    else:
        y_train_labeled_onehot = np.empty((0, len(np.unique(y))))

    #
    # SET UP INPUTS
    #

    # create true y placeholder (not used in unsupervised training)
    y_true = tf.placeholder(tf.float32, shape=(None, params['n_clusters']), name='y_true')

    batch_sizes = {
            'Unlabeled': params['batch_size'],
            'Labeled': params['batch_size'],
            'Orthonorm': params.get('batch_size_orthonorm', params['batch_size']),
            }

    input_shape = x.shape[1:]

    # spectralnet has three inputs -- they are defined here
    inputs = {
            'Unlabeled': Input(shape=input_shape,name='UnlabeledInput'),
            'Labeled': Input(shape=input_shape,name='LabeledInput'),
            'Orthonorm': Input(shape=input_shape,name='OrthonormInput'),
            }

    #
    # DEFINE SIAMESE NET
    #

    # run only if we are using a siamese network
    if params['affinity'] == 'siamese':
        # set up the siamese network inputs as well
        siamese_inputs = {
                'A': inputs['Unlabeled'],
                'B': Input(shape=input_shape),
                'Labeled': inputs['Labeled'],
                }

        # generate layers
        layers = []
        layers += make_layer_list(params['arch'], 'siamese', params.get('siam_reg'))

        # create the siamese net
        siamese_outputs = stack_layers(siamese_inputs, layers)

        # add the distance layer
        distance = Lambda(costs.euclidean_distance, output_shape=costs.eucl_dist_output_shape)([siamese_outputs['A'], siamese_outputs['B']])

        #create the distance model for training
        siamese_net_distance = Model([siamese_inputs['A'], siamese_inputs['B']], distance)

    #
    # TRAIN SIAMESE NET
    #

        # compile the siamese network
        siamese_net_distance.compile(loss=costs.contrastive_loss, optimizer=RMSprop())

        # create handler for early stopping and learning rate scheduling
        siam_lh = LearningHandler(
                lr=params['siam_lr'],
                drop=params['siam_drop'],
                lr_tensor=siamese_net_distance.optimizer.lr,
                patience=params['siam_patience'])

        # initialize the training generator
        train_gen_ = train_gen(pairs_train, dist_train, params['siam_batch_size'])

        # format the validation data for keras
        validation_data = ([pairs_val[:, 0], pairs_val[:, 1]], dist_val)

        # compute the steps per epoch
        steps_per_epoch = int(len(pairs_train) / params['siam_batch_size'])

        # train the network
        hist = siamese_net_distance.fit_generator(train_gen_, epochs=params['siam_ne'], validation_data=validation_data, steps_per_epoch=steps_per_epoch, callbacks=[siam_lh])

        # compute the siamese embeddings of the input data
        all_siam_preds = train.predict(siamese_outputs['A'], x_unlabeled=x_train, inputs=inputs, y_true=y_true, batch_sizes=batch_sizes)

    #
    # DEFINE SPECTRALNET
    #

    # generate layers
    layers = []
    layers = make_layer_list(params['arch'][:-1], 'spectral', params.get('spec_reg'))
    layers += [
              {'type': 'tanh',
               'size': params['n_clusters'],
               'l2_reg': params.get('spec_reg'),
               'name': 'spectral_{}'.format(len(params['arch'])-1)},
              {'type': 'Orthonorm', 'name':'orthonorm'}
              ]

    # create spectralnet
    outputs = stack_layers(inputs, layers)
    spectral_net = Model(inputs=inputs['Unlabeled'], outputs=outputs['Unlabeled'])

    #
    # DEFINE SPECTRALNET LOSS
    #

    # generate affinity matrix W according to params
    if params['affinity'] == 'siamese':
        input_affinity = tf.concat([siamese_outputs['A'], siamese_outputs['Labeled']], axis=0)
        x_affinity = all_siam_preds
    elif params['affinity'] in ['knn', 'full']:
        input_affinity = tf.concat([inputs['Unlabeled'], inputs['Labeled']], axis=0)
        x_affinity = x_train

    # calculate scale for affinity matrix
    scale = get_scale(x_affinity, batch_sizes['Unlabeled'], params['scale_nbr'])

    # create affinity matrix
    if params['affinity'] == 'full':
        W = costs.full_affinity(input_affinity, scale=scale)
    elif params['affinity'] in ['knn', 'siamese']:
        W = costs.knn_affinity(input_affinity, params['n_nbrs'], scale=scale, scale_nbr=params['scale_nbr'])

    # if we have labels, use them
    if len(x_train_labeled):
        # get true affinities (from labeled data)
        W_true = tf.cast(tf.equal(costs.squared_distance(y_true), 0),dtype='float32')

        # replace lower right corner of W with W_true
        unlabeled_end = tf.shape(inputs['Unlabeled'])[0]
        W_u = W[:unlabeled_end, :]                  # upper half
        W_ll = W[unlabeled_end:, :unlabeled_end]    # lower left
        W_l = tf.concat((W_ll, W_true), axis=1)      # lower half
        W = tf.concat((W_u, W_l), axis=0)

        # create pairwise batch distance matrix Dy
        Dy = costs.squared_distance(tf.concat([outputs['Unlabeled'], outputs['Labeled']], axis=0))
    else:
        Dy = costs.squared_distance(outputs['Unlabeled'])

    # define loss
    spectral_net_loss = K.sum(W * Dy) / (2 * params['batch_size'])

    # create the train step update
    learning_rate = tf.Variable(0., name='spectral_net_learning_rate')
    train_step = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(spectral_net_loss, var_list=spectral_net.trainable_weights)

    #
    # TRAIN SPECTRALNET
    #

    # initialize spectralnet variables
    K.get_session().run(tf.variables_initializer(spectral_net.trainable_weights))

    # set up validation/test set inputs
    inputs_test = {'Unlabeled':inputs['Unlabeled'], 'Orthonorm':inputs['Orthonorm']}

    # create handler for early stopping and learning rate scheduling
    spec_lh = LearningHandler(
            lr=params['spec_lr'],
            drop=params['spec_drop'],
            lr_tensor=learning_rate,
            patience=params['spec_patience'])

    # begin spectralnet training loop
    spec_lh.on_train_begin()
    for i in range(params['spec_ne']):
        # train spectralnet
        loss = train.train_step(
                return_var=[spectral_net_loss],
                updates=spectral_net.updates + [train_step],
                x_unlabeled=x_train_unlabeled,
                inputs=inputs,
                y_true=y_true,
                batch_sizes=batch_sizes,
                x_labeled=x_train_labeled,
                y_labeled=y_train_labeled_onehot,
                batches_per_epoch=100)[0]

        # get validation loss
        val_loss = train.predict_sum(
                spectral_net_loss,
                x_unlabeled=x_val_unlabeled,
                inputs=inputs,
                y_true=y_true,
                x_labeled=x[0:0],
                y_labeled=y_train_labeled_onehot,
                batch_sizes=batch_sizes)

        # do early stopping if necessary
        if spec_lh.on_epoch_end(i, val_loss):
            print('STOPPING EARLY')
            break

        # print training status
        print("Epoch: {}, loss={:2f}, val_loss={:2f}".format(i, loss, val_loss))

    print("finished training")

    #
    # EVALUATE
    #

    # get final embeddings
    x_spectralnet = train.predict(
            outputs['Unlabeled'],
            x_unlabeled=x,
            inputs=inputs_test,
            y_true=y_true,
            x_labeled=x_train_labeled[0:0],
            y_labeled=y_train_labeled_onehot[0:0],
            batch_sizes=batch_sizes)

    # get accuracy and nmi
    kmeans_assignments, km = get_cluster_sols(x_spectralnet, ClusterClass=KMeans, n_clusters=params['n_clusters'], init_args={'n_init':10})
    y_spectralnet, _ = get_y_preds(kmeans_assignments, y, params['n_clusters'])
    print_accuracy(kmeans_assignments, y, params['n_clusters'])
    from sklearn.metrics import normalized_mutual_info_score as nmi
    nmi_score = nmi(kmeans_assignments, y)
    print('NMI: ' + str(np.round(nmi_score, 3)))

    if params['generalization_metrics']:
        x_spectralnet_train = train.predict(
                outputs['Unlabeled'],
                x_unlabeled=x_train_unlabeled,
                inputs=inputs_test,
                y_true=y_true,
                x_labeled=x_train_labeled[0:0],
                y_labeled=y_train_labeled_onehot[0:0],
                batch_sizes=batch_sizes)
        x_spectralnet_test = train.predict(
                outputs['Unlabeled'],
                x_unlabeled=x_test,
                inputs=inputs_test,
                y_true=y_true,
                x_labeled=x_train_labeled[0:0],
                y_labeled=y_train_labeled_onehot[0:0],
                batch_sizes=batch_sizes)
        km_train = KMeans(n_clusters=params['n_clusters']).fit(x_spectralnet_train)
        from scipy.spatial.distance import cdist
        dist_mat = cdist(x_spectralnet_test, km_train.cluster_centers_)
        closest_cluster = np.argmin(dist_mat, axis=1)
        print_accuracy(closest_cluster, y_test, params['n_clusters'], ' generalization')
        nmi_score = nmi(closest_cluster, y_test)
        print('generalization NMI: ' + str(np.round(nmi_score, 3)))

    return x_spectralnet, y_spectralnet

