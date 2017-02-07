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
tf.app.flags.DEFINE_string('device_ids', '0,1', """Device ids. split by comma, e.g. 0,1""")
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

EPOCH_SIZE = 50000
TEST_SIZE = 10000


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
    global parameters
    name = 'conv' + str(conv_counter)
    conv_counter += 1
    with tf.variable_scope(name) as scope:
        #kernel = tf.get_variable(name='weights', initializer=tf.random_normal([kH, kW, nIn, nOut], dtype=tf.float32, stddev=1e-2))
        kernel = tf.get_variable(name='weights', shape=[kH, kW, nIn, nOut], initializer=tf.truncated_normal_initializer(dtype=tf.float32, stddev=1e-2))
        strides = [1, dH, dW, 1]
        conv = tf.nn.conv2d(inpOp, kernel, strides, padding=padType)
        #biases = tf.Variable(tf.constant(0.0, shape=[nOut], dtype=tf.float32),
        #                     trainable=True, name='biases')
        biases = tf.get_variable(name='biases', initializer=tf.constant(0.0, shape=[nOut], dtype=tf.float32), dtype=tf.float32)
        bias = tf.reshape(tf.nn.bias_add(conv, biases),
                          conv.get_shape())
        parameters += [kernel, biases]
        return bias


def _relu(inpOp):
    return tf.nn.relu(inpOp)


def _padding(inpOp, pad):
    global pad_counter 
    name = 'pad' + str(pad_counter)
    pad_counter += 1
    with tf.name_scope(name) as scope:
        padded_input = tf.pad(inpOp, [[0, 0], [pad, pad], [pad, pad], [0, 0]], "CONSTANT", name='pad')
        print('padded_input: ', padded_input)
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
    with tf.variable_scope(name) as scope:
        #kernel = tf.get_variable(name='weights', initializer=tf.random_normal([nIn, nOut],
        #                                         dtype=tf.float32,
        #                                         stddev=1e-2))
        kernel = tf.get_variable(name='weights', shape=[nIn, nOut], initializer=tf.truncated_normal_initializer(dtype=tf.float32,
                                                 stddev=1e-2))
        biases = tf.get_variable(name='biases', shape=[nOut], initializer=tf.constant_initializer())
        affine1 = tf.nn.relu_layer(inpOp, kernel, biases, name=name)
        parameters += [kernel, biases]
        return affine1

def _mpool(inpOp, kH, kW, dH, dW):
    global pool_counter
    global parameters
    name = 'pool' + str(pool_counter)
    pool_counter += 1
    ksize = [1, kH, kW, 1]
    strides = [1, dH, dW, 1]
    return tf.nn.max_pool(inpOp,
                          ksize=ksize,
                          strides=strides,
                          padding='VALID',
                          name=name)

def _avgpool(inpOp, kH, kW, dH, dW):
    global pool_counter
    name = 'pool' + str(pool_counter)
    pool_counter += 1
    ksize = [1, kH, kW, 1]
    strides = [1, dH, dW, 1]
    return tf.nn.avg_pool(inpOp,
                          ksize=ksize,
                          strides=strides,
                          padding='VALID',
                          name=name)

def loss_function(logits, labels):
    batch_size = tf.size(labels)
    labels = tf.expand_dims(labels, 1)
    indices = tf.expand_dims(tf.range(0, batch_size, 1), 1)
    concated = tf.concat(1, [indices, labels])
    onehot_labels = tf.sparse_to_dense(
        concated, tf.pack([batch_size, 10]), 1.0, 0.0)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits,
                                                            onehot_labels,
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
    conv2 = _conv (pad2,  32, 32, 5, 5, 1, 1, 'VALID')
    pool2 = _mpool(conv2,  3, 3, 2, 2)
    relu2 = _relu(pool2)
    #norm2 = _norm(relu2, 3, 5e-05, 0.75)

    pad3 = _padding(relu2, 2)
    conv3 = _conv (pad3,  32, 64, 5, 5, 1, 1, 'VALID')
    relu3 = _relu(conv3)
    pool3 = _avgpool(relu3, 3, 3, 2, 2) 
    print('pool3: ', pool3)

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
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(0, grads)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


def train():
    global parameters
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=FLAGS.log_device_placement)
    with tf.Graph().as_default(), tf.device("/cpu:0"):
        global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)

        device_ids = FLAGS.device_ids.split(',')
        print('device_ids: ', device_ids)
        if len(device_ids) > FLAGS.num_gpus:
            print('The device_ids should have the same number of GPUs with num_gpus')
            return

        lr = 0.001
        #optimizer = tf.train.GradientDescentOptimizer(lr)
        optimizer = tf.train.MomentumOptimizer(lr, 0.9)

        def assign_to_device(device, ps_device="/cpu:0"):
            def _assign(op):
                node_def = op if isinstance(op, tf.NodeDef) else op.node_def
                if node_def.op == "Variable":
                    return ps_device
                else:
                    return device
            return _assign

        tower_grads = []
        average_loss_tensor = []
        for i in xrange(FLAGS.num_gpus):
            print('what is i: ', i)
            #with tf.device(assign_to_device('/gpu:%s'%device_ids[i])):
            with tf.device('/gpu:%s'%device_ids[i]):
                with tf.name_scope('%s_%s' % ('TOWER', device_ids[i])) as n_scope:
                    _init_global_variables()
                    images, labels = cifar10_input.inputs(False, FLAGS.data_dir, FLAGS.batch_size)
                    logits = inference(images)
                    loss = loss_function(logits, labels)

                    tf.add_to_collection('losses', loss)
                    tf.add_n(tf.get_collection('losses'), name='total_loss')

                    losses = tf.get_collection('losses', n_scope)
                    total_loss = tf.add_n(losses, name='total_loss')
                    average_loss_tensor.append(total_loss)

                    tf.get_variable_scope().reuse_variables()
                    print('total_loss: ', total_loss)
                    grads = optimizer.compute_gradients(total_loss)
                    print('grads: ', grads)

                    tower_grads.append(grads)

        print('tower_grads: ', tower_grads)
        print('len0: ', len(tower_grads[0]))
        print('len1: ', len(tower_grads[1]))

        grads = average_gradients(tower_grads)
        apply_gradient_op = optimizer.apply_gradients(grads, global_step=global_step)
        train_op = apply_gradient_op
        average_op = tf.reduce_mean(average_loss_tensor, 0)

        # Create a saver.
        saver = tf.train.Saver(tf.all_variables())

        init = tf.initialize_all_variables()
        sess = tf.Session(config=config)
        sess.run(init)
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
            #_, loss_v = sess.run([train_op, total_loss])
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

        coord.request_stop()
        coord.join(threads)

        average_batch_time /= iterations
        print 'average_batch_time: ', average_batch_time
        print ('epoch_info: %s' % ','.join(epochs_info))


def main(_):
    train()


if __name__ == '__main__':
    tf.app.run()
