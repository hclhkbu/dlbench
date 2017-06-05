from datetime import datetime

import time
import cifar10_input

import tensorflow as tf
import numpy as np
import os
from resnet import inference_small, loss

FLAGS = tf.app.flags.FLAGS

parameters = []
device_str = ''

conv_counter = 1
pool_counter = 1
norm_counter = 1
affine_counter = 1

FLAGS = tf.app.flags.FLAGS
# Basic model parameters.
tf.app.flags.DEFINE_integer('batch_size', 128, """Number of images to process in a batch.""")
tf.app.flags.DEFINE_integer('epochs', 40, """Max epochs for training.""")
tf.app.flags.DEFINE_integer('log_step', 100, """Log step""")
tf.app.flags.DEFINE_integer('eval_step', 1, """Evaluate step of epoch""")
tf.app.flags.DEFINE_integer('device_id', 0, """Device id.""")
#tf.app.flags.DEFINE_string('data_dir', '/home/comp/csshshi/data/tensorflow/cifar10/cifar-10-batches-bin',"""Data directory""")
tf.app.flags.DEFINE_string('data_dir', os.environ['HOME']+'/data/tensorflow/cifar10/cifar-10-batches-bin', """Data directory""")
#tf.app.flags.DEFINE_string('data_dir', '/home/comp/csshshi/data/tensorflow/cifar10/cifar-10-batches-bin',"""Data directory""")
#tf.app.flags.DEFINE_string('data_dir', '/home/ipdps/Data/tensorflow/cifar10/cifar-10-batches-bin', """Data directory""")
tf.app.flags.DEFINE_string('train_dir', './trained_models/',
                           """Path to the data directory.""")
tf.app.flags.DEFINE_boolean('use_fp16', False,
                            """Train the model using fp16.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
tf.app.flags.DEFINE_boolean('use_dataset', False, """True to use datasets""")
tf.app.flags.DEFINE_string('data_format', 'NCHW', """NCHW for GPU and NHWC for CPU.""")

EPOCH_SIZE = 50000
TEST_SIZE = 10000


def train():
  global parameters
  data_format = FLAGS.data_format
  config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=FLAGS.log_device_placement)
  #config.gpu_options.force_gpu_compatible = 1
  device_id = FLAGS.device_id
  if int(device_id) >= 0:
      device_str = '/gpu:%d'%int(device_id)
      config.allow_soft_placement = True
      config.intra_op_parallelism_threads = 1
      config.inter_op_parallelism_threads = 0
  else:
      device_str = '/cpu:0'
      num_threads = os.getenv('OMP_NUM_THREADS', 1)
      config = tf.ConfigProto(allow_soft_placement=True, intra_op_parallelism_threads=int(num_threads))
      # Default format for CPU.  When using MKL NCHW might be better but that has not been proven.
      data_format = 'NHWC'
  print('Using data format:{}'.format(data_format))
  with tf.Graph().as_default(), tf.device(device_str), tf.Session(config=config) as sess:
      initalizer = None
      images = None
      labels = None
      with tf.device('/cpu:0'):
        if FLAGS.use_dataset:
          iterator, initalizer =  cifar10_input.dataSet(FLAGS.data_dir, FLAGS.batch_size,
                                                        data_format=data_format,
                                                        device=device_str)
          images, labels = iterator.get_next()
        else:
          images, labels = cifar10_input.inputs(False, FLAGS.data_dir, FLAGS.batch_size, data_format=data_format)
        labels = tf.contrib.layers.one_hot_encoding(labels, 10)
      logits = inference_small(images, is_training=True, num_blocks=9, data_format=data_format)
      # Add a simple objective so we can calculate the backward pass.
      loss_value = loss(logits, labels)
      # Compute the gradient with respect to all the parameters.
      lr = 0.01
      update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
      with tf.control_dependencies(update_ops):
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
          average_batch_time += float(duration)
          average_loss += loss_v
          assert not np.isnan(loss_v), 'Model diverged with loss = NaN'
          if step % FLAGS.log_step == 0:
              examples_per_sec = FLAGS.batch_size / duration
              sec_per_batch = float(duration)
              format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f sec/batch)')
              print (format_str % (datetime.now(), step, loss_v, examples_per_sec, sec_per_batch))
          if step > 0 and step % (FLAGS.eval_step * num_batches_per_epoch) == 0:
              average_loss /= num_batches_per_epoch * FLAGS.eval_step
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
      print ('epoch_info: %s'% ','.join(epochs_info))


def main(_):
    os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'
    os.environ['TF_SYNC_ON_FINISH'] = '0'
    train()


if __name__ == '__main__':
    tf.app.run()
