from datetime import datetime

import time
import cifar10_input

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

FLAGS = tf.app.flags.FLAGS
# Basic model parameters.
tf.app.flags.DEFINE_integer('batch_size', 1024, """Number of images to process in a batch.""")
tf.app.flags.DEFINE_integer('epochs', 40, """Max epochs for training.""")
tf.app.flags.DEFINE_integer('log_step', 50, """Log step""")
tf.app.flags.DEFINE_integer('eval_step', 1, """Evaluate step of epoch""")
tf.app.flags.DEFINE_integer('device_id', 0, """Device id.""")
#tf.app.flags.DEFINE_string('data_dir', '/home/comp/csshshi/data/tensorflow/cifar10/cifar-10-batches-bin', """Data directory""")
tf.app.flags.DEFINE_string('data_dir', os.environ['HOME']+'/data/tensorflow/cifar10/cifar-10-batches-bin', """Data directory""")
#tf.app.flags.DEFINE_string('data_dir', '/home/comp/pengfeixu/Data/tensorflow/cifar10/cifar-10-batches-bin', """Data directory""")
tf.app.flags.DEFINE_string('train_dir', './trained_models/',
                           """Path to the data directory.""")
tf.app.flags.DEFINE_boolean('use_fp16', False,
                            """Train the model using fp16.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
tf.app.flags.DEFINE_boolean('use_dataset', False, """True to use datasets""")

data_format = 'NCHW'
data_format_c = 'channels_first'

EPOCH_SIZE = 50000
TEST_SIZE = 10000

def get_device_str(device_id):
    global device_str
    if int(device_id) >= 0:
        device_str = '/gpu:%d'%int(device_id)
    else:
        device_str = '/cpu:0'
    return device_str


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
    #ksize = [1, kH, kW, 1]
    #strides = [1, 1, dH, dW]
    return tf.layers.average_pooling2d(
        inpOp, [kH, kW], [dH, dW],
        padding='VALID',
        data_format=data_format_c,
        name=name)

def loss(logits, labels):
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


def train():
  global parameters
  config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=FLAGS.log_device_placement)
  config.intra_op_parallelism_threads = 1
  config.inter_op_parallelism_threads = 0
  device_str = get_device_str(FLAGS.device_id)
  if device_str.find('cpu') >= 0: # cpu version
      num_threads = os.getenv('OMP_NUM_THREADS', 1)
      print 'num_threads: ', num_threads
      config = tf.ConfigProto(allow_soft_placement=True, intra_op_parallelism_threads=int(num_threads))
  with tf.Graph().as_default(), tf.device(device_str), tf.Session(config=config) as sess:
      initalizer = None
      images = None
      labels = None
      with tf.device('/cpu:0'):
        if FLAGS.use_dataset:
          iterator, initalizer =  cifar10_input.dataSet(FLAGS.data_dir, FLAGS.batch_size)
          images, labels = iterator.get_next()
        else:
          images, labels = cifar10_input.inputs(False, FLAGS.data_dir, FLAGS.batch_size)

      labels = tf.contrib.layers.one_hot_encoding(labels, 10)
      logits = inference(images)
      # Add a simple objective so we can calculate the backward pass.
      loss_value = loss(logits, labels)
      # Compute the gradient with respect to all the parameters.
      lr = 0.001
      grad = tf.train.MomentumOptimizer(lr, 0.9).minimize(loss_value)

      # Create a saver.
      saver = tf.train.Saver(tf.global_variables())

      # Build an initialization operation.
      init = tf.global_variables_initializer()
      # Start running operations on the Graph.
      sess.run(init)
      coord = None
      threads = None
      if FLAGS.use_dataset:
        sess.run(initalizer)
      else:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
      real_batch_size = FLAGS.batch_size
      num_batches_per_epoch = int((EPOCH_SIZE + real_batch_size - 1)/ real_batch_size)
      iterations = FLAGS.epochs * num_batches_per_epoch 
      average_batch_time = 0.0

      epochs_info = []
      average_loss = 0.0
      for step in xrange(iterations):
          start_time = time.time()
          _, loss_v = sess.run([grad, loss_value])
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
          if step == iterations-1:
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
