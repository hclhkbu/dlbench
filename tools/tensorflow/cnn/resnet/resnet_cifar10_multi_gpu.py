from datetime import datetime

import six
import time
import cifar10_input

import tensorflow as tf
import numpy as np
import os
import operator
#from resnet import inference, loss
from resnet import inference_small, loss


FLAGS = tf.app.flags.FLAGS
# Basic model parameters.
tf.app.flags.DEFINE_integer('batch_size', 128, """Number of images to process in a batch.""")
tf.app.flags.DEFINE_integer('epochs', 40, """Max epochs for training.""")
tf.app.flags.DEFINE_integer('log_step', 100, """Log step""")
tf.app.flags.DEFINE_integer('eval_step', 1, """Evaluate step of epoch""")
tf.app.flags.DEFINE_string('device_ids', None, """Device ids. split by comma, e.g. 0,1""")
#tf.app.flags.DEFINE_string('data_dir', '/home/comp/csshshi/data/tensorflow/cifar10/cifar-10-batches-bin',"""Data directory""")
tf.app.flags.DEFINE_string('data_dir', os.environ['HOME']+'/data/tensorflow/cifar10/cifar-10-batches-bin', """Data directory""")
#tf.app.flags.DEFINE_string('data_dir', '/home/comp/pengfeixu/Data/tensorflow/cifar10/cifar-10-batches-bin', """Data directory""")
tf.app.flags.DEFINE_string('train_dir', './trained_models/',
                           """Path to the data directory.""")
tf.app.flags.DEFINE_boolean('use_fp16', False,
                            """Train the model using fp16.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
tf.app.flags.DEFINE_integer('num_gpus', 2, """How many GPUs to use.""")
# CPU:0 is best for ResNet regardless of peering.  I do not know about CIFAR on P100 but even 
# on the P100 via the DGX-1 CPU is the better choice.  
tf.app.flags.DEFINE_string('local_ps_device', 'CPU', """Local parameter server GPU if gpus are peered or CPU otherwise try both.""")
tf.app.flags.DEFINE_boolean('use_dataset', False, """True to use datasets""")
tf.app.flags.DEFINE_string('data_format', 'NCHW', """NCHW for GPU and NHWC for CPU.""")

EPOCH_SIZE = 50000
TEST_SIZE = 10000


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
    config.allow_soft_placement = True
    config.intra_op_parallelism_threads = 1
    config.inter_op_parallelism_threads = 0

    with tf.Graph().as_default(), tf.device("/" + FLAGS.local_ps_device):
        global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)

        device_ids = FLAGS.device_ids
        if not device_ids:
            device_ids = [str(i) for i in range(FLAGS.num_gpus)]
        else:
            device_ids = device_ids.split(',')

        if len(device_ids) > FLAGS.num_gpus:
            print('The device_ids should have the same number of GPUs with num_gpus')
            return

        lr = 0.01
        #optimizer = tf.train.GradientDescentOptimizer(lr)
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
                iterator, initalizer =  cifar10_input.dataSet(FLAGS.data_dir,
                                                              FLAGS.batch_size,
                                                              device='gpu',
                                                              data_format=FLAGS.data_format)
                images, labels = iterator.get_next()

        tower_grads = []
        reuse_variables = None
        losses = []
        for i in six.moves.range(FLAGS.num_gpus):
            with tf.device(assign_to_device('/gpu:%s'%device_ids[i])):
                with tf.name_scope('%s_%s' % ('TOWER', device_ids[i])) as n_scope:
                    with tf.device('/cpu:0'):
                        if not FLAGS.use_dataset:
                            images, labels = cifar10_input.inputs(False, FLAGS.data_dir, FLAGS.batch_size, data_format=FLAGS.data_format)
                    with tf.variable_scope(tf.get_variable_scope(), reuse=reuse_variables):
                        logits = inference_small(images, is_training=True, num_blocks=9, data_format=FLAGS.data_format)
                    hot_labels = tf.contrib.layers.one_hot_encoding(labels, 10)
                    tower_loss = loss(logits, hot_labels)
                    losses.append(tower_loss)
                    grads = optimizer.compute_gradients(tower_loss)
                    tower_grads.append(grads)
                    reuse_variables = True
        
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, 'TOWER_0')
        with tf.control_dependencies(update_ops):
            # Average losses accross towers (GPUs)
            total_loss = tf.reduce_mean(losses, 0)
            grads = average_gradients(tower_grads)
            apply_gradient_op = optimizer.apply_gradients(grads, global_step=global_step)
        train_op = apply_gradient_op

        # Create a saver.
        saver = tf.train.Saver(tf.global_variables())

        # Build an initialization operation.
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
        for step in six.moves.xrange(iterations):
            start_time = time.time()
            _, loss_v = sess.run([train_op, total_loss])
            duration = time.time() - start_time
            average_loss += loss_v
            average_batch_time += float(duration)
            assert not np.isnan(loss_v), 'Model diverged with loss = NaN'
            if step % FLAGS.log_step == 0:
                examples_per_sec = FLAGS.batch_size / duration
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
        print('average_batch_time: ', average_batch_time)
        print ('epoch_info: %s'% ','.join(epochs_info))


def main(_):
    os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'
    os.environ['TF_SYNC_ON_FINISH'] = '0'
    train()


if __name__ == '__main__':
    tf.app.run()
