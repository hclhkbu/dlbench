import tensorflow as tf

parameters = []
conv_counter = 1
pool_counter = 1
affine_counter = 1


FLAGS = tf.app.flags.FLAGS

def _conv(inpOp, nIn, nOut, kH, kW, dH, dW, padType):
    global conv_counter
    global parameters
    name = 'conv' + str(conv_counter)
    conv_counter += 1
    with tf.name_scope(name) as scope:
        kernel = tf.Variable(tf.truncated_normal([kH, kW, nIn, nOut],
                                                 dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
        if FLAGS.data_format == 'NCHW':
          strides = [1, 1, dH, dW]
        else:
          strides = [1, dH, dW, 1]
        conv = tf.nn.conv2d(inpOp, kernel, strides, padding=padType,
                            data_format=FLAGS.data_format)
        biases = tf.Variable(tf.constant(0.0, shape=[nOut], dtype=tf.float32),
                             trainable=True, name='biases')
        bias = tf.reshape(tf.nn.bias_add(conv, biases,
                                         data_format=FLAGS.data_format),
                          conv.get_shape())
        conv1 = tf.nn.relu(bias, name=scope)
        parameters += [kernel, biases]
        return conv1


def _affine(inpOp, nIn, nOut):
    global affine_counter
    global parameters
    name = 'affine' + str(affine_counter)
    affine_counter += 1
    with tf.name_scope(name) as scope:
        kernel = tf.Variable(tf.truncated_normal([nIn, nOut],
                                                 dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
        biases = tf.Variable(tf.constant(0.0, shape=[nOut], dtype=tf.float32),
                             trainable=True, name='biases')
        affine1 = tf.nn.relu_layer(inpOp, kernel, biases, name=name)
        parameters += [kernel, biases]
        return affine1


def _mpool(inpOp, kH, kW, dH, dW):
    global pool_counter
    global parameters
    name = 'pool' + str(pool_counter)
    pool_counter += 1
    if FLAGS.data_format == 'NCHW':
      ksize = [1, 1, kH, kW]
      strides = [1, 1, dH, dW]
    else:
      ksize = [1, kH, kW, 1]
      strides = [1, dH, dW, 1]
    return tf.nn.max_pool(inpOp,
                          ksize=ksize,
                          strides=strides,
                          padding='SAME',
                          data_format=FLAGS.data_format,
                          name=name)


def loss(logits, labels):
    batch_size = tf.size(labels)
    labels = tf.expand_dims(labels, 1)
    indices = tf.expand_dims(tf.range(0, batch_size, 1), 1)
    concated = tf.concat(1, [indices, labels])
    onehot_labels = tf.sparse_to_dense(
        concated, tf.pack([batch_size, 1000]), 1.0, 0.0)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits,
                                                            onehot_labels,
                                                            name='xentropy')
    loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')
    return loss


def inference(images, labels):
    conv1 = _conv (images, 3, 96, 11, 11, 4, 4, 'SAME')
    pool1 = _mpool(conv1,  3, 3, 2, 2)
    conv2 = _conv (pool1,  96, 256, 5, 5, 1, 1, 'SAME')
    pool2 = _mpool(conv2,  3, 3, 2, 2)
    conv3 = _conv (pool2,  256, 384, 3, 3, 1, 1, 'SAME')
    conv4 = _conv (conv3,  384, 384, 3, 3, 1, 1, 'SAME')
    conv5 = _conv (conv4,  384, 256, 3, 3, 1, 1, 'SAME')
    pool5 = _mpool(conv5,  3, 3, 2, 2)
    resh1 = tf.reshape(pool5, [-1, 256 * 7 * 7])
    affn1 = _affine(resh1, 256 * 7 * 7, 4096)
    affn2 = _affine(affn1, 4096, 4096)
    affn3 = _affine(affn2, 4096, labels)
    return affn3


def alexnet(features, labels):
    return inference(features, labels)

