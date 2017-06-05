from datetime import datetime

import time
import cifar10_input
#import unpickle as cifar10_input

import tensorflow as tf
import numpy as np
import os

FLAGS = tf.app.flags.FLAGS

parameters = []
device_str = ''

conv_counter = 1
pool_counter = 1
norm_counter = 1
affine_counter = 1
pad_counter = 1

FLAGS = tf.app.flags.FLAGS
# Basic model parameters.
tf.app.flags.DEFINE_integer('batch_size', 1024, """Number of images to process in a batch.""")
tf.app.flags.DEFINE_integer('epochs', 40, """Max epochs for training.""")
tf.app.flags.DEFINE_integer('log_step', 100, """Log step""")
tf.app.flags.DEFINE_integer('eval_step', 1, """Evaluate step of epoch""")
tf.app.flags.DEFINE_string('device_ids', '', """Device ids. split by comma, e.g. 0,1""")
#tf.app.flags.DEFINE_string('data_dir', '/home/comp/csshshi/data/tensorflow/cifar10/cifar-10-batches-bin', """Data directory""")
tf.app.flags.DEFINE_string('data_dir', os.environ['HOME']+'/data/tensorflow/cifar10/cifar-10-batches-bin', """Data directory""")
#tf.app.flags.DEFINE_string('data_dir', '/home/comp/pengfeixu/Data/tensorflow/cifar10/cifar-10-batches-bin', """Data directory""")
tf.app.flags.DEFINE_string('train_dir', './trained_models/',
                           """Path to the data directory.""")
tf.app.flags.DEFINE_boolean('use_fp16', False,
                            """Train the model using fp16.""")
tf.app.flags.DEFINE_boolean('log_device_placement', True,
                            """Whether to log device placement.""")
tf.app.flags.DEFINE_integer('num_gpus', 2, """How many GPUs to use.""")
tf.app.flags.DEFINE_string('local_ps_device', 'GPU', """Local parameter server GPU if gpus are peered or CPU otherwise try both.""")
tf.app.flags.DEFINE_boolean('use_dataset', False, """True to use datasets""")

EPOCH_SIZE = 50000
TEST_SIZE = 10000

data_format = 'NCHW'
data_format_c = 'channels_first'

def _init_global_variables():
    global conv_counter
    global pool_counter
    global norm_counter
    global affine_counter
    global pad_counter
    conv_counter = 1
    pool_counter = 1
    norm_counter = 1
    affine_counter = 1
    pad_counter = 1


def _conv(inpOp, nIn, nOut, kH, kW, dH, dW, padType):
    global conv_counter
    name = 'conv' + str(conv_counter)
    conv_counter += 1
    with tf.variable_scope(name):
        kernel_initializer = tf.truncated_normal_initializer(stddev=1e-2)
        conv = tf.layers.conv2d(inpOp,
                        nOut, 
                        [kH, kW],
                        strides=[dH, dW],
                        padding=padType,
                        data_format=data_format_c,
                        kernel_initializer=kernel_initializer,
                        use_bias=False)
        biases = tf.get_variable(
                        'biases', [nOut], tf.float32,
                        tf.constant_initializer(0.0))

        bias = tf.reshape(tf.nn.bias_add(conv, biases, data_format=data_format),
                          conv.get_shape())
        return bias


def _relu(inpOp):
    return tf.nn.relu(inpOp)


def _padding(inpOp, pad):
    padded_input = tf.pad(inpOp, [[0, 0], [0, 0], [pad, pad], [pad, pad]], "CONSTANT")
    return padded_input


def _norm(inpOp, local_size, alpha, beta):
    global norm_counter
    name = 'norm' + str(norm_counter)
    norm = tf.nn.lrn(inpOp, local_size, bias=1.0, alpha=alpha, beta=beta, name=name)
    return norm


def _affine(inpOp, nIn, nOut):
    global affine_counter
    global parameters
    name = 'affine' + str(affine_counter)
    affine_counter += 1
    with tf.name_scope(name) as scope:
        kernel = tf.get_variable(
            'weights', [nIn, nOut],
            tf.float32,
            tf.truncated_normal_initializer(stddev=1e-2))
        biases = tf.get_variable('biases', [nOut],
                                 tf.float32,
                                 tf.constant_initializer(0.1))
        logits = tf.matmul(inpOp, kernel) + biases
        return tf.nn.relu(logits, name=name)

def _mpool(inpOp, kH, kW, dH, dW):
    global pool_counter
    global parameters
    name = 'pool' + str(pool_counter)
    pool_counter += 1
    return tf.layers.max_pooling2d(
        inpOp, [kH, kW], [dH, dW],
        padding='VALID',
        data_format=data_format_c,
        name=name)    

def _avgpool(inpOp, kH, kW, dH, dW):
    global pool_counter
    name = 'pool' + str(pool_counter)
    pool_counter += 1
    return tf.layers.average_pooling2d(
        inpOp, [kH, kW], [dH, dW],
        padding='VALID',
        data_format=data_format_c,
        name=name)

def loss_function(logits, labels):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                            labels=labels,
                                                            name='xentropy')
    loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')
    return loss

def inference(images):
    pad1 = _padding(images, 2) 
    conv1 = _conv (pad1, 3, 32, 5, 5, 1, 1, 'VALID')
    pool1 = _mpool(conv1,  3, 3, 2, 2)
    relu1 = _relu(pool1)
    #norm1 = _norm(relu1, 3, 5e-05, 0.75)

    pad2 = _padding(relu1, 2)
    conv2 = _conv (pad2, 32, 32, 5, 5, 1, 1, 'VALID')
    pool2 = _mpool(conv2, 3, 3, 2, 2)
    relu2 = _relu(pool2)
    #norm2 = _norm(relu2, 3, 5e-05, 0.75)

    pad3 = _padding(relu2, 2)
    conv3 = _conv (pad3,  32, 64, 5, 5, 1, 1, 'VALID')
    relu3 = _relu(conv3)
    pool3 = _avgpool(relu3, 3, 3, 2, 2) 

    resh1 = tf.reshape(pool3, [-1, 64 * 3 * 3])
    affn1 = _affine(resh1, 64*3*3, 10)

    return affn1

def average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.

    Note that this function provides a synchronization point across all towers.

    Args:
      tower_grads: List of lists of (gradient, variable) tuples. The outer list
        is over individual gradients. The inner list is over the gradient
        calculation for each tower.
    Returns:
       List of pairs of (gradient, variable) where the gradient has been averaged
       across all towers.
    """
    average_grads = []
    for single_grads in zip(*tower_grads):
        grads = [g for g, _ in single_grads]
        grad = tf.add_n(grads)
        grad = tf.multiply(grad, 1.0/len(grads))
        v = single_grads[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


def train():
    global parameters
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=FLAGS.log_device_placement)
    config.intra_op_parallelism_threads = 1
    config.inter_op_parallelism_threads = 0
    with tf.Graph().as_default(), tf.device("/" + FLAGS.local_ps_device + ":0"):
        global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)

        device_ids = FLAGS.device_ids
        if not device_ids:
            device_ids = [str(i) for i in range(FLAGS.num_gpus)]
        else:
            device_ids = device_ids.split(',')

        print('device_ids: ', device_ids)
        if len(device_ids) > FLAGS.num_gpus:
            print('The device_ids should have the same number of GPUs with num_gpus')
            return

        lr = 0.001
        optimizer = tf.train.MomentumOptimizer(lr, 0.9)

        def assign_to_device(device, ps_device=FLAGS.local_ps_device):
            worker_device = device
            ps_sizes = [0]
            if FLAGS.local_ps_device.lower == 'gpu':
                ps_sizes = [0] * FLAGS.num_gpus
            def _assign(op):
                if op.device:
                  return op.device
                if op.type not in ['Variable', 'VariableV2']:
                  return worker_device
                device_index, _ = min(enumerate(
                    ps_sizes), key=operator.itemgetter(1))
                device_name = '/' + FLAGS.local_ps_device +':' + str(device_index)
                var_size = op.outputs[0].get_shape().num_elements()
                ps_sizes[device_index] += var_size
                return device_name
            return _assign

        images = None
        labels = None
        initalizer = None
        if FLAGS.use_dataset:
            with tf.device('/CPU:0'):
                iterator, initalizer =  cifar10_input.dataSet(FLAGS.data_dir, FLAGS.batch_size)
                images, labels = iterator.get_next()

        tower_grads = []
        average_loss_tensor = []
        reuse_variables = False
        for i in xrange(FLAGS.num_gpus):
            print('what is i: ', i)
            with tf.device('/gpu:%s'%device_ids[i]):
                with tf.name_scope('%s_%s' % ('TOWER', device_ids[i])) as n_scope:
                    _init_global_variables()
                    with tf.device('/cpu:0'):
                        if not FLAGS.use_dataset:
                            images, labels = cifar10_input.inputs(False, FLAGS.data_dir, FLAGS.batch_size)
                    with tf.variable_scope(tf.get_variable_scope(), reuse=reuse_variables):    
                        logits = inference(images)
                    loss = loss_function(logits, tf.contrib.layers.one_hot_encoding(labels, 10))
                    reuse_variables = True

                    average_loss_tensor.append(loss)
                    grads = optimizer.compute_gradients(loss)
                    tower_grads.append(grads)

        grads = average_gradients(tower_grads)
        apply_gradient_op = optimizer.apply_gradients(grads, global_step=global_step)
        train_op = apply_gradient_op
        average_op = tf.reduce_mean(average_loss_tensor)

        # Create a saver.
        saver = tf.train.Saver(tf.global_variables())

        init = tf.global_variables_initializer()
        sess = tf.Session(config=config)
        sess.run(init)
        coord = None
        threads = None
        if FLAGS.use_dataset:
            sess.run(initalizer)
        else:
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        real_batch_size = FLAGS.batch_size * FLAGS.num_gpus
        num_batches_per_epoch = int((EPOCH_SIZE + real_batch_size - 1)/ real_batch_size)
        iterations = FLAGS.epochs * num_batches_per_epoch 
        average_batch_time = 0.0
        epochs_info = []

        step = 0
        average_loss = 0.0
        for step in xrange(iterations):
            start_time = time.time()
            _, loss_v = sess.run([train_op, average_op])
            duration = time.time() - start_time
            average_batch_time += float(duration)

            assert not np.isnan(loss_v), 'Model diverged with loss = NaN'
            average_loss += loss_v

            if step % FLAGS.log_step == 0:
                examples_per_sec = real_batch_size / duration
                sec_per_batch = float(duration)
                format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f sec/batch)')
                print (format_str % (datetime.now(), step, loss_v, examples_per_sec, sec_per_batch))

            if step > 0 and step % (FLAGS.eval_step * num_batches_per_epoch) == 0:
                average_loss /= num_batches_per_epoch * FLAGS.eval_step
                print ('epoch: %d, loss: %.2f' % (step /num_batches_per_epoch, average_loss))
                epochs_info.append('%d:_:%s'%(step/(FLAGS.eval_step*num_batches_per_epoch), average_loss)) 
                average_loss = 0.0

        checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)

        if not FLAGS.use_dataset:
            coord.request_stop()
            coord.join(threads)

        average_batch_time /= iterations
        print 'average_batch_time: ', average_batch_time
        print ('epoch_info: %s' % ','.join(epochs_info))


def main(_):
    os.environ['TF_SYNC_ON_FINISH'] = '0'
    os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'
    train()


if __name__ == '__main__':
    tf.app.run()
