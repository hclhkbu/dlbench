import tensorflow as tf
import models
import cifar10_input
import time
import numpy as np
from datetime import datetime

IMAGE_SIZE = cifar10_input.IMAGE_SIZE
NUM_CLASSES = cifar10_input.NUM_CLASSES
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 350.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.1       # Initial learning rate.

FLAGS = tf.app.flags.FLAGS
# Basic model parameters.
tf.app.flags.DEFINE_integer('batch_size', 128, """Number of images to process in a batch.""")
tf.app.flags.DEFINE_integer('iterations', 1000, """Max iterations for training.""")
tf.app.flags.DEFINE_integer('log_step', 100, """Log step""")
tf.app.flags.DEFINE_integer('deviceId', 0, """Device id.""")
tf.app.flags.DEFINE_string('data_dir', '/home/comp/csshshi/data/tensorflow/cifar10/cifar-10-batches-bin',
                           """Path to the CIFAR-10 data directory.""")
tf.app.flags.DEFINE_boolean('use_fp16', False,
                            """Train the model using fp16.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")


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


def train(model='fcn5'):
    config = tf.ConfigProto(allow_soft_placement=False,log_device_placement=FLAGS.log_device_placement)
    #with tf.Graph().as_default(), tf.device('/gpu:%d'%FLAGS.deviceId), tf.Session(config=config) as sess:
    with tf.Graph().as_default(), tf.Session(config=config) as sess:
        #images, labels = None, None
        #images, labels = cifar10_input.inputs(eval_data=False, data_dir=FLAGS.data_dir, batch_size=FLAGS.batch_size, reshape_to_one=True)
        images = tf.placeholder(tf.float32, [None, 32*32*3])
        labels = tf.placeholder(tf.float32, [None, 10])
        print 'images: ', images
        print 'labels: ', labels 
        logits = None
        if model == 'fcn5':
            logits = models.model_fcn5(images)
        else:
            logits = models.model_fcn8(images)
        loss = models.loss(logits, labels)
        lr = 0.5
        global_step = tf.Variable(0, name='global_step', trainable=False)
        optimizer = tf.train.GradientDescentOptimizer(lr).minimize(loss, global_step=global_step)

        print '1'
        init = tf.initialize_all_variables()
        print '2'
        #sess = tf.Session(config=config)
        print '3'
        sess.run(init)

        print '4'
        tf.train.start_queue_runners(sess=sess)
        print 'am I here??'

        for step in xrange(FLAGS.iterations):
            start_time = time.time()
            imgs, labs = getFakeMinibatch(FLAGS.batch_size, 10)
            _, loss_value = sess.run([optimizer, loss], feed_dict={images:imgs,labels:labs})
            duration = time.time() - start_time
            assert not np.isnan(loss_value), 'Model diverged with loss = NaN'
            if step % FLAGS.log_step == 0:
                examples_per_sec = FLAGS.batch_size / duration
                sec_per_batch = float(duration)
                format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f sec/batch)')
                print (format_str % (datetime.now(), step, loss_value, examples_per_sec, sec_per_batch))


def main(argv=None):
    train(model='fcn5')


if __name__ == '__main__':
    tf.app.run()
