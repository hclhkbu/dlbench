from datetime import datetime

import six
import time
import cifar10_input

import tensorflow as tf
import numpy as np
import os
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
tf.app.flags.DEFINE_string('local_ps_device', 'GPU:0', """Local parameter server GPU:0 if gpus are peered or CPU:0 otherwise try both.""")

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
        grad = tf.concat(axis=0, values=grads)
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

        def assign_to_device(device, ps_device="/" + FLAGS.local_ps_device):
            #if FLAGS.num_gpus == 1:
            #    ps_device="/gpu:0"
            def _assign(op):
                node_def = op if isinstance(op, tf.NodeDef) else op.node_def
                if node_def.op in ["Variable","VariableV2"]:
                    return ps_device
                else:
                    return device
            return _assign

        tower_grads = []
        reuse_variables = None
        losses = []
        for i in six.moves.range(FLAGS.num_gpus):
            print('what is i: ', i)

            with tf.device(assign_to_device('/gpu:%s'%device_ids[i])):
                with tf.name_scope('%s_%s' % ('TOWER', device_ids[i])) as n_scope:
                    with tf.device('/cpu:0'):
                        images, labels = cifar10_input.inputs(False, FLAGS.data_dir, FLAGS.batch_size)
                    #logits = inference(images, is_training=True)
                    with tf.variable_scope(tf.get_variable_scope(), reuse=reuse_variables):
                        logits = inference_small(images, is_training=True, num_blocks=9)
                    tower_loss = loss(logits, labels)
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
            #_, loss_v = sess.run([train_op, total_loss])
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

        coord.request_stop()
        coord.join(threads)
        average_batch_time /= iterations
        print('average_batch_time: ', average_batch_time)
        print ('epoch_info: %s'% ','.join(epochs_info))


def main(_):
    os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'
    train()


if __name__ == '__main__':
    tf.app.run()
