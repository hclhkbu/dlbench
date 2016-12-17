import tensorflow as tf
import models
import time
import os
import numpy as np
from datetime import datetime
from tensorflow.examples.tutorials.mnist import input_data


FLAGS = tf.app.flags.FLAGS
# Basic model parameters.
tf.app.flags.DEFINE_integer('batch_size', 1024, """Number of images to process in a batch.""")
tf.app.flags.DEFINE_integer('epochs', 40, """Max epochs for training.""")
tf.app.flags.DEFINE_integer('log_step', 10, """Log step""")
tf.app.flags.DEFINE_integer('eval_step', 1, """Evaluate step of epoch""")
tf.app.flags.DEFINE_integer('device_id', 0, """Device id.""")
#tf.app.flags.DEFINE_string('data_dir', '/home/comp/csshshi/data/tensorflow/MNIST_data/',
tf.app.flags.DEFINE_string('data_dir', os.environ['HOME']+'/data/tensorflow/MNIST_data/',
#tf.app.flags.DEFINE_string('data_dir', '/home/comp/pengfeixu/Data/tensorflow/MNIST_data/',
                           """Path to the data directory.""")
tf.app.flags.DEFINE_boolean('use_fp16', False,
                            """Train the model using fp16.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
tf.app.flags.DEFINE_integer('num_gpus', 1, """How many GPUs to use.""")

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


def train(model='fcn5'):
    config = tf.ConfigProto(allow_soft_placement=False,log_device_placement=FLAGS.log_device_placement)
    device_id = FLAGS.device_id
    device_str = ''
    if int(device_id) >= 0:
        device_str = '/gpu:%d'%int(device_id)
    else:
        device_str = '/cpu:0'
        num_threads = os.getenv('OMP_NUM_THREADS', 1)
        config = tf.ConfigProto(allow_soft_placement=True, intra_op_parallelism_threads=int(num_threads))


    with tf.Graph().as_default(), tf.device(device_str), tf.Session(config=config) as sess:
        feature_dim = models.feature_dim
        label_dim = models.label_dim
        images = tf.placeholder(tf.float32, [None, feature_dim])
        labels = tf.placeholder(tf.float32, [None, label_dim])

        logits = None
        if model == 'fcn5':
            logits = models.model_fcn5(images)
        else:
            logits = models.model_fcn8(images)
        loss = models.loss(logits, labels)

        predictionCorrectness = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
        accuracy = tf.reduce_mean(tf.cast(predictionCorrectness, "float"))

        lr = 0.05
        #optimizer = tf.train.GradientDescentOptimizer(lr).minimize(loss)
        optimizer = tf.train.MomentumOptimizer(lr, 0.9).minimize(loss)

        init = tf.initialize_all_variables()
        sess.run(init)
        tf.train.start_queue_runners(sess=sess)
        batch_size_per_epoch = int((EPOCH_SIZE + FLAGS.batch_size - 1)/ FLAGS.batch_size)
        iterations = FLAGS.epochs * batch_size_per_epoch 
        average_batch_time = 0.0
        epochs_info = []
        average_loss = 0.0
        for step in range(iterations):
            start_time = time.time()
            imgs, labs = get_real_batch_data(FLAGS.batch_size, 10)
            _, loss_value = sess.run([optimizer, loss], feed_dict={images:imgs,labels:labs})
            average_loss += loss_value
            duration = time.time() - start_time
            average_batch_time += float(duration)
            assert not np.isnan(loss_value), 'Model diverged with loss = NaN'
            if step % FLAGS.log_step == 0:
                examples_per_sec = FLAGS.batch_size / duration
                sec_per_batch = float(duration)
                format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f sec/batch)')
                print (format_str % (datetime.now(), step, loss_value, examples_per_sec, sec_per_batch))
            if step > 0 and step % (FLAGS.eval_step * batch_size_per_epoch) == 0:
                average_loss /= FLAGS.eval_step * batch_size_per_epoch
                accuracy_value = accuracy.eval(feed_dict={images: mnist.test.images, labels: mnist.test.labels})
                print("test accuracy %g"%accuracy_value)
                epochs_info.append('%d:%g:%s'%(step/(FLAGS.eval_step*batch_size_per_epoch), accuracy_value, average_loss)) 
                average_loss = 0.0
        average_batch_time /= iterations
        print 'average_batch_time: ', average_batch_time
        print ('epoch_info: %s' % ','.join(epochs_info))


def main(argv=None):
    train(model='fcn5')


if __name__ == '__main__':
    tf.app.run()
