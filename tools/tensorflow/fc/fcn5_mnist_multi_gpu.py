import os
import operator
import tensorflow as tf
import models
import time
import numpy as np
from datetime import datetime
from tensorflow.examples.tutorials.mnist import input_data


FLAGS = tf.app.flags.FLAGS
# Basic model parameters.

tf.app.flags.DEFINE_string('train_dir', './multigpu-trained',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('batch_size', 1024, """Number of images to process in a batch.""")
tf.app.flags.DEFINE_integer('epochs', 40, """Max epochs for training.""")
tf.app.flags.DEFINE_integer('log_step', 10, """Log step""")
tf.app.flags.DEFINE_integer('eval_step', 1, """Evaluate step of epoch""")
tf.app.flags.DEFINE_string('device_ids', '', """Device ids. split by comma, e.g. 0,1""")
#tf.app.flags.DEFINE_string('data_dir', '/home/comp/csshshi/data/tensorflow/MNIST_data/',
tf.app.flags.DEFINE_string('data_dir', os.environ['HOME']+'/data/tensorflow/MNIST_data/',
#tf.app.flags.DEFINE_string('data_dir', '/home/comp/pengfeixu/Data/tensorflow/MNIST_data/',
                           """Path to the data directory.""")
tf.app.flags.DEFINE_boolean('use_fp16', False,
                            """Train the model using fp16.""")
tf.app.flags.DEFINE_boolean('log_device_placement', True,
                            """Whether to log device placement.""")
tf.app.flags.DEFINE_integer('num_gpus', 2, """How many GPUs to use.""")
tf.app.flags.DEFINE_string('local_ps_device', 'GPU', """Local parameter server GPU if gpus are peered or CPU otherwise try both.""")
tf.app.flags.DEFINE_boolean('use_dataset', False,
                            """Whether to use datasets vs. feed_dict.""")
tf.app.flags.DEFINE_boolean('xla', False,
                            """True to use XLA, which has to be compiled in.""")

EPOCH_SIZE = 60000
TEST_SIZE = 10000


def createFakeData(count, featureDim, labelDim):
    features = np.random.randn(count, featureDim)
    labels = np.random.randint(0, labelDim, size=(count, 1))
    return features, labels

features, labels = createFakeData(1024, 32*32*3, 10)


def getFakeMinibatch(minibatchSize, labelDim):
    feat = features[:minibatchSize]
    l = labels[:minibatchSize]
    lab = np.zeros((minibatchSize, labelDim))
    for i in range(lab.shape[0]):
        lab[i][l[i]] = 1
    return feat, lab

mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)


def get_real_batch_data(batch_size, label_dim):
    batch_xs, batch_ys = mnist.train.next_batch(batch_size)
    return batch_xs, batch_ys 


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


def train(model='fcn5'):

    config = tf.ConfigProto(allow_soft_placement=True,log_device_placement=FLAGS.log_device_placement)

    if FLAGS.xla:
        # Turns on XLA.  XLA is not included in the standard build.  For single GPU this shows ~5% improvement
        config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1

    with tf.Graph().as_default(), tf.device("/" + FLAGS.local_ps_device + ":0"):
        global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)

        device_ids = FLAGS.device_ids
        if not device_ids:
            device_ids = [str(i) for i in range(FLAGS.num_gpus)]
        else:
            device_ids = device_ids.split(',')

        lr = 0.05
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
        if FLAGS.use_dataset:
            with tf.device('/CPU:0'):
                d_features = mnist.train.images
                d_labels = mnist.train.labels
                dataset = tf.contrib.data.Dataset.from_tensor_slices((d_features, d_labels))
                dataset = dataset.shuffle(buffer_size=60000)
                dataset = dataset.repeat()
                dataset = dataset.batch(FLAGS.batch_size)
                # Trick to get datasets to buffer the next epoch.  This is needed because
                # the data loading is occuring outside DataSets in python.  Normally preprocessing
                # would occur in DataSets and this odd looking line is not needed.  
                dataset = dataset.map(lambda x,y:(x,y),
                    num_threads=FLAGS.num_gpus,
                    output_buffer_size=FLAGS.num_gpus)
                iterator = dataset.make_initializable_iterator()
                images,labels = iterator.get_next()

        tower_grads = []
        feed_vars = []
        average_loss_tensor = []
        reuse_variables = False
        accuracy = None
        for i in xrange(FLAGS.num_gpus):
            with tf.device(assign_to_device('/gpu:%s'%device_ids[i])):
                with tf.name_scope('%s_%s' % ('TOWER', device_ids[i])) as scope:
                    if not FLAGS.use_dataset:
                        feature_dim = models.feature_dim
                        label_dim = models.label_dim
                        images = tf.placeholder(tf.float32, [None, feature_dim], name='images')
                        labels = tf.placeholder(tf.int64, [None, label_dim], name='labels')
                        feed_vars.append((images, labels))
                    with tf.variable_scope(tf.get_variable_scope(), reuse=reuse_variables): 
                        logits = models.model_fcn5(images)
                    if i == 0:
                        # Prediction only on GPU:0
                        predictionCorrectness = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
                        accuracy = tf.reduce_mean(tf.cast(predictionCorrectness, "float"))
                    loss = models.loss(logits, labels)
                    reuse_variables = True
                    average_loss_tensor.append(loss)
                    grads = optimizer.compute_gradients(loss)
                    tower_grads.append(grads)

        grads = average_gradients(tower_grads)
        apply_gradient_op = optimizer.apply_gradients(grads, global_step=global_step)

        train_op = apply_gradient_op
        average_op = tf.reduce_mean(average_loss_tensor)
        saver = tf.train.Saver(tf.global_variables())

        init = tf.global_variables_initializer()
        sess = tf.Session(config=config)
        sess.run(init)
        if FLAGS.use_dataset:
            sess.run(iterator.initializer)
            
        real_batch_size = FLAGS.batch_size * FLAGS.num_gpus
        num_batches_per_epoch = int((EPOCH_SIZE + real_batch_size - 1)/ real_batch_size)
        iterations = FLAGS.epochs * num_batches_per_epoch 
        average_batch_time = 0.0
        epochs_info = []

        step = 0
        average_loss = 0.0
        for step in range(iterations):
            start_time = time.time()
            feed_dict = {}
            if not FLAGS.use_dataset:
                imgs, labs = get_real_batch_data(real_batch_size, 10)
                for i in range(FLAGS.num_gpus):
                    feed_dict[feed_vars[i][0]] = imgs[i*FLAGS.batch_size:(i+1)*FLAGS.batch_size]
                    feed_dict[feed_vars[i][1]] = labs[i*FLAGS.batch_size:(i+1)*FLAGS.batch_size] 
            _, loss_value = sess.run([train_op, average_op], feed_dict=feed_dict)
            duration = time.time() - start_time
            average_batch_time += float(duration)
            average_loss += loss_value

            assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

            if step % FLAGS.log_step == 0:
                examples_per_sec = (FLAGS.batch_size * FLAGS.num_gpus) / duration
                sec_per_batch = float(duration)
                format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f sec/batch)')
                print (format_str % (datetime.now(), step, loss_value, examples_per_sec, sec_per_batch))

            if step > 0 and step % (FLAGS.eval_step * num_batches_per_epoch) == 0:
                average_loss /= num_batches_per_epoch * FLAGS.eval_step
                print ('epoch: %d, loss: %.2f' % (step/(FLAGS.eval_step*num_batches_per_epoch), average_loss))
                epochs_info.append('%d:-:%s'%(step/(FLAGS.eval_step*num_batches_per_epoch), average_loss)) 
                average_loss = 0.0
                feed_dict = { images: mnist.test.images, labels :mnist.test.labels }
                if not FLAGS.use_dataset:
                    feed_dict = {}
                    feed_dict[feed_vars[0][0]] = mnist.test.images
                    feed_dict[feed_vars[0][1]] = mnist.test.labels
                accuracy_value = accuracy.eval(session=sess, feed_dict=feed_dict)
                print("test accuracy %g"%accuracy_value)

        checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)

        average_batch_time /= iterations
        print 'average_batch_time: ', average_batch_time
        print ('epoch_info: %s' % ','.join(epochs_info))


def main(argv=None):
    os.environ['TF_SYNC_ON_FINISH'] = '0'
    os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'
    train(model='fcn5')


if __name__ == '__main__':
    tf.app.run()
