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
    global parameters
    name = 'conv' + str(conv_counter)
    conv_counter += 1
    with tf.name_scope(name) as scope:
        kernel = tf.Variable(tf.truncated_normal([kH, kW, nIn, nOut],
                                                 dtype=tf.float32,
                                                 stddev=1e-2), name='weights')
        #kernel = tf.Variable(tf.random_normal([kH, kW, nIn, nOut],
        #                                         dtype=tf.float32,
        #                                         stddev=1e-2), name='weights')
        strides = [1, dH, dW, 1]
        conv = tf.nn.conv2d(inpOp, kernel, strides, padding=padType)
        biases = tf.Variable(tf.constant(0.0, shape=[nOut], dtype=tf.float32),
                             trainable=True, name='biases')
        bias = tf.reshape(tf.nn.bias_add(conv, biases),
                          conv.get_shape())
        parameters += [kernel, biases]
        return bias


def _relu(inpOp):
    return tf.nn.relu(inpOp)


def _padding(inpOp, pad):
    padded_input = tf.pad(inpOp, [[0, 0], [pad, pad], [pad, pad], [0, 0]], "CONSTANT")
    print('padded_input: ', padded_input)
    return padded_input
    #return inpOp


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
        kernel = tf.Variable(tf.truncated_normal([nIn, nOut],
                                                 dtype=tf.float32,
                                                 stddev=1e-2), name='weights')
        #kernel = tf.Variable(tf.random_normal([nIn, nOut],
        #                                         dtype=tf.float32,
        #                                         stddev=1e-2), name='weights')
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

def loss(logits, labels):
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

def inference2(images):
    #pad1 = _padding(images, 2) 
    conv1 = _conv (images, 3, 32, 5, 5, 1, 1, 'SAME')
    pool1 = _mpool(conv1,  3, 3, 2, 2)
    relu1 = _relu(pool1)
    #norm1 = _norm(relu1, 3, 5e-05, 0.75)

    #pad2 = _padding(relu1, 2)
    conv2 = _conv (relu1,  32, 32, 5, 5, 1, 1, 'SAME')
    pool2 = _mpool(conv2,  3, 3, 2, 2)
    relu2 = _relu(pool2)
    #norm2 = _norm(relu2, 3, 5e-05, 0.75)

    #pad3 = _padding(relu2, 2)
    conv3 = _conv (relu2,  32, 64, 5, 5, 1, 1, 'SAME')
    relu3 = _relu(conv3)
    pool3 = _avgpool(relu3, 3, 3, 2, 2) 
    print('pool3: ', pool3)

    resh1 = tf.reshape(pool3, [-1, 64 * 3 * 3])
    affn1 = _affine(resh1, 64*3*3, 10)

    return affn1


def train():
  global parameters
  config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=FLAGS.log_device_placement)
  device_str = get_device_str(FLAGS.device_id)
  if device_str.find('cpu') >= 0: # cpu version
      num_threads = os.getenv('OMP_NUM_THREADS', 1)
      print 'num_threads: ', num_threads
      config = tf.ConfigProto(allow_soft_placement=True, intra_op_parallelism_threads=int(num_threads))
  with tf.Graph().as_default(), tf.device(device_str), tf.Session(config=config) as sess:
      #image_size = 32 
      #images, labels = cifar10_input.distorted_inputs(FLAGS.data_dir, FLAGS.batch_size)
      images, labels = cifar10_input.inputs(False, FLAGS.data_dir, FLAGS.batch_size)
      print('Images: ', images)

      logits = inference(images)
      #logits = inference2(images)
      # Add a simple objective so we can calculate the backward pass.
      loss_value = loss(logits, labels)
      # Compute the gradient with respect to all the parameters.
      lr = 0.001
      #grad = tf.train.GradientDescentOptimizer(lr).minimize(loss_value)
      grad = tf.train.MomentumOptimizer(lr, 0.9).minimize(loss_value)

      # Create a saver.
      saver = tf.train.Saver(tf.all_variables())

      # Build an initialization operation.
      init = tf.initialize_all_variables()
      # Start running operations on the Graph.
      sess.run(init)
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
          average_loss += loss_v
          duration = time.time() - start_time
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
      coord.request_stop()
      coord.join(threads)
      average_batch_time /= iterations
      print 'average_batch_time: ', average_batch_time
      print ('epoch_info: %s' % ','.join(epochs_info))


def main(_):
    train()


if __name__ == '__main__':
    tf.app.run()
