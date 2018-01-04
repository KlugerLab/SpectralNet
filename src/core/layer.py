'''
layer.py: contains functions used to build all spectral and siamese net models
'''
from keras.layers import Dense, BatchNormalization, Flatten, Conv2D, MaxPooling2D, Lambda, Dropout
from keras import backend as K
import tensorflow as tf
import numpy as np
from keras.regularizers import l2

def orthonorm_op(x, epsilon=1e-7):
    '''
    Computes a matrix that orthogonalizes the input matrix x

    x:      an n x d input matrix
    eps:    epsilon to prevent nonzero values in the diagonal entries of x

    returns:    a d x d matrix, ortho_weights, which orthogonalizes x by
                right multiplication
    '''
    x_2 = K.dot(K.transpose(x), x)
    x_2 += K.eye(K.int_shape(x)[1])*epsilon
    L = tf.cholesky(x_2)
    ortho_weights = tf.transpose(tf.matrix_inverse(L)) * tf.sqrt(tf.cast(tf.shape(x)[0], dtype=K.floatx()))
    return ortho_weights

def Orthonorm(x, name=None):
    '''
    Builds keras layer that handles orthogonalization of x

    x:      an n x d input matrix
    name:   name of the keras layer

    returns:    a keras layer instance. during evaluation, the instance returns an n x d orthogonal matrix
                if x is full rank and not singular
    '''
    # get dimensionality of x
    d = x.get_shape().as_list()[-1]
    # compute orthogonalizing matrix
    ortho_weights = orthonorm_op(x)
    # create variable that holds this matrix
    ortho_weights_store = K.variable(np.zeros((d,d)))
    # create op that saves matrix into variable
    ortho_weights_update = tf.assign(ortho_weights_store, ortho_weights, name='ortho_weights_update')
    # switch between stored and calculated weights based on training or validation
    l = Lambda(lambda x: K.in_train_phase(K.dot(x, ortho_weights), K.dot(x, ortho_weights_store)), name=name)

    l.add_update(ortho_weights_update)
    return l

def stack_layers(inputs, layers, kernel_initializer='glorot_uniform'):
    '''
    Builds the architecture of the network by applying each layer specified in layers to inputs.

    inputs:     a dict containing input_types and input_placeholders for each key and value pair, respecively.
                for spectralnet, this means the input_types 'Unlabeled' and 'Orthonorm'*
    layers:     a list of dicts containing all layers to be used in the network, where each dict describes
                one such layer. each dict requires the key 'type'. all other keys are dependent on the layer
                type

    kernel_initializer: initialization configuration passed to keras (see keras initializers)

    returns:    outputs, a dict formatted in much the same way as inputs. it contains input_types and
                output_tensors for each key and value pair, respectively, where output_tensors are
                the outputs of the input_placeholders in inputs after each layer in layers is applied

    * this is necessary since spectralnet takes multiple inputs and performs special computations on the
      orthonorm layer
    '''
    outputs = dict()

    for key in inputs:
        outputs[key]=inputs[key]

    for layer in layers:
        # check for l2_reg argument
        l2_reg = layer.get('l2_reg')
        if l2_reg:
            l2_reg = l2(layer['l2_reg'])

        # create the layer
        if layer['type'] == 'softplus_reg':
            l = Dense(layer['size'], activation='softplus', kernel_initializer=kernel_initializer, kernel_regularizer=l2(0.001), name=layer.get('name'))
        elif layer['type'] == 'softplus':
            l = Dense(layer['size'], activation='softplus', kernel_initializer=kernel_initializer, kernel_regularizer=l2_reg, name=layer.get('name'))
        elif layer['type'] == 'softmax':
            l = Dense(layer['size'], activation='softmax', kernel_initializer=kernel_initializer, kernel_regularizer=l2_reg, name=layer.get('name'))
        elif layer['type'] == 'tanh':
            l = Dense(layer['size'], activation='tanh', kernel_initializer=kernel_initializer, kernel_regularizer=l2_reg, name=layer.get('name'))
        elif layer['type'] == 'relu':
            l = Dense(layer['size'], activation='relu', kernel_initializer=kernel_initializer, kernel_regularizer=l2_reg, name=layer.get('name'))
        elif layer['type'] == 'selu':
            l = Dense(layer['size'], activation='selu', kernel_initializer=kernel_initializer, kernel_regularizer=l2_reg, name=layer.get('name'))
        elif layer['type'] == 'Conv2D':
            l = Conv2D(layer['channels'], kernel_size=layer['kernel'], activation='relu', data_format='channels_last', kernel_regularizer=l2_reg, name=layer.get('name'))
        elif layer['type'] == 'BatchNormalization':
            l = BatchNormalization(name=layer.get('name'))
        elif layer['type'] == 'MaxPooling2D':
            l = MaxPooling2D(pool_size=layer['pool_size'], data_format='channels_first', name=layer.get('name'))
        elif layer['type'] == 'Dropout':
            l = Dropout(layer['rate'], name=layer.get('name'))
        elif layer['type'] == 'Flatten':
            l = Flatten(name=layer.get('name'))
        elif layer['type'] == 'Orthonorm':
            l = Orthonorm(outputs['Orthonorm'], name=layer.get('name'));
        else:
            raise ValueError("Invalid layer type '{}'".format(layer['type']))

        # apply the layer to each input in inputs
        for k in outputs:
            outputs[k]=l(outputs[k])

    return outputs
