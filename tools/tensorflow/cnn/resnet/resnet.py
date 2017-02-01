import skimage.io  # bug. need to import this before tensorflow
import skimage.transform  # bug. need to import this before tensorflow
import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.training import moving_averages

from config import Config

import datetime
import numpy as np
import os
import time

#MOVING_AVERAGE_DECAY = 0.9999
MOVING_AVERAGE_DECAY = 0.0001
BN_DECAY = MOVING_AVERAGE_DECAY
BN_EPSILON = 2e-5 #0.001
#BN_EPSILON = 0.001
#CONV_WEIGHT_DECAY = 0.00004
CONV_WEIGHT_DECAY = 0.0001
CONV_WEIGHT_STDDEV = 0.1
#FC_WEIGHT_DECAY = 0.00004
FC_WEIGHT_DECAY = 0.0001
#FC_WEIGHT_DECAY = 0
FC_WEIGHT_STDDEV = 0.01
RESNET_VARIABLES = 'resnet_variables'
UPDATE_OPS_COLLECTION = 'resnet_update_ops'  # must be grouped with training op

tf.app.flags.DEFINE_integer('input_size', 32, "input image size")


activation = tf.nn.relu


# This is what they use for CIFAR-10 and 100.
# See Section 4.2 in http://arxiv.org/abs/1512.03385
def inference_small(x,
                    is_training,
                    num_blocks=9, # 6n+2 total weight layers will be used.
                    use_bias=False, # defaults to using batch norm
                    num_classes=10):
    c = Config()
    c['is_training'] = tf.convert_to_tensor(is_training,
                                            dtype='bool',
                                            name='is_training')
    c['use_bias'] = use_bias
    c['fc_units_out'] = num_classes
    c['num_blocks'] = num_blocks
    c['num_classes'] = num_classes
    return inference_small_config(x, c)

def inference_small_config(x, c):
    c['bottleneck'] = False
    c['ksize'] = 3
    c['stride'] = 1
    with tf.variable_scope('scale1'):
        c['conv_filters_out'] = 16
        c['block_filters_internal'] = 16
        c['stack_stride'] = 1
        x = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], "CONSTANT")
        x = conv(x, c)
        x = bn(x, c)
        x = activation(x)
        x = stack(x, c)

    with tf.variable_scope('scale2'):
        c['block_filters_internal'] = 32
        c['stack_stride'] = 2
        x = stack(x, c)

    with tf.variable_scope('scale3'):
        c['block_filters_internal'] = 64
        c['stack_stride'] = 2
        x = stack(x, c)

    # post-net
    #x = tf.nn.avg_pool(x, ksize=[1, 8, 8, 1], strides=[1, 1, 1, 1], padding="VALID", name="avg_pool")
    x = tf.reduce_mean(x, axis=[1, 2], name="avg_pool")

    if c['num_classes'] != None:
        with tf.variable_scope('fc'):
            x = fc(x, c)

    return x


def loss(logits, labels):
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
 
    regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)

    loss_ = tf.add_n([cross_entropy_mean] + regularization_losses)
    tf.summary.scalar('loss', loss_)

    return loss_


def stack(x, c):
    for n in range(c['num_blocks']):
        s = c['stack_stride'] if n == 0 else 1
        c['block_stride'] = s
        with tf.variable_scope('block%d' % (n + 1)):
            x = block(x, c)
    return x


def block(x, c):
    filters_in = x.get_shape()[-1]

    # Note: filters_out isn't how many filters are outputed. 
    # That is the case when bottleneck=False but when bottleneck is 
    # True, filters_internal*4 filters are outputted. filters_internal is how many filters
    # the 3x3 convs output internally.
    m = 4 if c['bottleneck'] else 1
    filters_out = m * c['block_filters_internal']

    shortcut = x  # branch 1

    c['conv_filters_out'] = c['block_filters_internal']

    # branch 2
    with tf.variable_scope('A'):
        pad = 1
        x = tf.pad(x, [[0, 0], [pad, pad], [pad, pad], [0, 0]], "CONSTANT")
        c['stride'] = c['block_stride']
        assert c['ksize'] == 3
        x = conv(x, c)
        x = bn(x, c)
        x = activation(x)

    with tf.variable_scope('B'):
        pad = 1
        x = tf.pad(x, [[0, 0], [pad, pad], [pad, pad], [0, 0]], "CONSTANT")
        c['conv_filters_out'] = filters_out
        assert c['ksize'] == 3
        assert c['stride'] == 1
        x = conv(x, c)
        x = bn(x, c)

    with tf.variable_scope('shortcut'):
        if filters_out != filters_in or c['block_stride'] != 1:
            print('shortcut: ', shortcut)
            print('x: ', x)
            stride = c['block_stride']
            print('stride: ', stride)
            if stride > 1:
                shortcut = tf.nn.avg_pool(shortcut, ksize=[1, 1, 1, 1], strides=[1, stride, stride, 1], padding="VALID", name="avg_pool")
            print('shortcut: ', shortcut)
            nChannels = int(shortcut.get_shape()[-1])
            nOutChannels = int(x.get_shape()[-1])
            if nOutChannels > nChannels:
                pad = (nOutChannels - nChannels)//2
                shortcut = tf.pad(shortcut, [[0, 0], [0, 0], [0, 0], [pad, pad]], "CONSTANT")
            print('shortcut: ', shortcut)

            c['ksize'] = 1
            c['stride'] = c['block_stride']
            c['conv_filters_out'] = filters_out

            #shortcut = conv(shortcut, c)
            #shortcut = bn(shortcut, c)

    return activation(x + shortcut)


def bn(x, c):
    x_shape = x.get_shape()
    params_shape = x_shape[-1:]

    if c['use_bias']:
        bias = _get_variable('bias', params_shape,
                             initializer=tf.zeros_initializer())
        return x + bias


    axis = list(range(len(x_shape) - 1))

    beta = _get_variable('beta',
                         params_shape,
                         initializer=tf.zeros_initializer())
    gamma = _get_variable('gamma',
                          params_shape,
                          initializer=tf.ones_initializer())

    moving_mean = _get_variable('moving_mean',
                                params_shape,
                                initializer=tf.zeros_initializer(),
                                trainable=False)
    moving_variance = _get_variable('moving_variance',
                                    params_shape,
                                    initializer=tf.ones_initializer(),
                                    trainable=False)

    # These ops will only be preformed when training.
    mean, variance = tf.nn.moments(x, axis)
    update_moving_mean = moving_averages.assign_moving_average(moving_mean,mean, BN_DECAY)
    update_moving_variance = moving_averages.assign_moving_average(moving_variance, variance, BN_DECAY)
    tf.add_to_collection(UPDATE_OPS_COLLECTION, update_moving_mean)
    tf.add_to_collection(UPDATE_OPS_COLLECTION, update_moving_variance)

    mean, variance = control_flow_ops.cond(
        c['is_training'], lambda: (mean, variance),
        lambda: (moving_mean, moving_variance))

    x = tf.nn.batch_normalization(x, mean, variance, beta, gamma, BN_EPSILON)
    #x.set_shape(inputs.get_shape()) ??

    return x


def fc(x, c):
    num_units_in = x.get_shape()[1]
    num_units_out = c['fc_units_out']
    #weights_initializer = tf.truncated_normal_initializer(stddev=FC_WEIGHT_STDDEV)
    weights_initializer =tf.contrib.layers.xavier_initializer(uniform=True) 

    weights = _get_variable('weights',
                            shape=[num_units_in, num_units_out],
                            initializer=weights_initializer,
                            weight_decay=FC_WEIGHT_DECAY)
    biases = _get_variable('biases',
                           shape=[num_units_out],
                           initializer=tf.zeros_initializer(), weight_decay=FC_WEIGHT_DECAY)
    x = tf.nn.xw_plus_b(x, weights, biases)
    return x


def _get_variable(name,
                  shape,
                  initializer,
                  weight_decay=0.0,
                  dtype='float',
                  trainable=True):
    "A little wrapper around tf.get_variable to do weight decay and add to"
    "resnet collection"
    if weight_decay > 0:
        regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
    else:
        regularizer = None
    collections = [tf.GraphKeys.GLOBAL_VARIABLES, RESNET_VARIABLES]
    return tf.get_variable(name,
                           shape=shape,
                           initializer=initializer,
                           dtype=dtype,
                           regularizer=regularizer,
                           collections=collections,
                           trainable=trainable)


def conv(x, c):
    ksize = c['ksize']
    stride = c['stride']
    filters_out = c['conv_filters_out']

    filters_in = x.get_shape()[-1]
    shape = [ksize, ksize, filters_in, filters_out]
    #initializer = tf.truncated_normal_initializer(stddev=CONV_WEIGHT_STDDEV)
    #stddev = np.sqrt(2/(int(filters_in) * ksize * ksize)) 
    #initializer = tf.random_normal_initializer(stddev=stddev)
    initializer = tf.contrib.layers.xavier_initializer_conv2d(uniform=False)
    weights = _get_variable('weights',
                            shape=shape,
                            dtype='float',
                            initializer=initializer,
                            weight_decay=CONV_WEIGHT_DECAY)
    biases = _get_variable("biases", shape=[filters_out], initializer=tf.constant_initializer(), trainable=True, weight_decay=CONV_WEIGHT_DECAY)
    c = tf.nn.conv2d(x, weights, [1, stride, stride, 1], padding='VALID')
    bias = tf.reshape(tf.nn.bias_add(c, biases), c.get_shape())
    return bias


def _max_pool(x, ksize=3, stride=2):
    return tf.nn.max_pool(x,
                          ksize=[1, ksize, ksize, 1],
                          strides=[1, stride, stride, 1],
                          padding='VALID')
