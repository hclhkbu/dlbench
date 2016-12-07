import skimage.io  # bug. need to import this before tensorflow
import skimage.transform  # bug. need to import this before tensorflow
from resnet_train  import train, set_parameters, get_device_str 
from resnet import *
import tensorflow as tf
import time
import os
import sys
import re
import numpy as np

from synset import *
import argparse
from image_processing import image_preprocessing

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('data_dir', '/home/ryan/data/ILSVRC2012/ILSVRC2012_img_train',
                           'imagenet dir')


def file_list(data_dir):
    dir_txt = data_dir + ".txt"
    filenames = []
    with open(dir_txt, 'r') as f:
        for line in f:
            if line[0] == '.': continue
            line = line.rstrip()
            fn = os.path.join(data_dir, line)
            filenames.append(fn)
    return filenames


def load_data(data_dir):
    data = []
    i = 0

    print "listing files in", data_dir
    start_time = time.time()
    files = file_list(data_dir)
    duration = time.time() - start_time
    print "took %f sec" % duration

    for img_fn in files:
        ext = os.path.splitext(img_fn)[1]
        if ext != '.JPEG': continue

        label_name = re.search(r'(n\d+)', img_fn).group(1)
        fn = os.path.join(data_dir, img_fn)

        label_index = synset_map[label_name]["index"]

        data.append({
            "filename": fn,
            "label_name": label_name,
            "label_index": label_index,
            "desc": synset[label_index],
        })

    return data


def distorted_inputs():
    data = load_data(FLAGS.data_dir)

    filenames = [ d['filename'] for d in data ]
    label_indexes = [ d['label_index'] for d in data ]

    filename, label_index = tf.train.slice_input_producer([filenames, label_indexes], shuffle=True)

    num_preprocess_threads = 4
    images_and_labels = []
    for thread_id in range(num_preprocess_threads):
        image_buffer = tf.read_file(filename)

        bbox = []
        train = True
        image = image_preprocessing(image_buffer, bbox, train, thread_id)
        images_and_labels.append([image, label_index])

    images, label_index_batch = tf.train.batch_join(
        images_and_labels,
        batch_size=FLAGS.batch_size,
        capacity=2 * num_preprocess_threads * FLAGS.batch_size)

    height = FLAGS.input_size
    width = FLAGS.input_size
    depth = 3

    images = tf.cast(images, tf.float32)
    images = tf.reshape(images, shape=[FLAGS.batch_size, height, width, depth])

    return images, tf.reshape(label_index_batch, [FLAGS.batch_size])


def main(_):

    print '-----with device: %s'%get_device_str()
    with tf.Graph().as_default(), tf.device(get_device_str()):
        image_size = 224
        #image_shape = [FLAGS.batch_size, image_size + 3, image_size + 3, 3]
        with tf.device('/cpu:0'):
            image_shape = [FLAGS.batch_size, image_size, image_size, 3]

            labels = tf.Variable(tf.ones([FLAGS.batch_size],
                                         dtype=tf.int32))
            images = tf.Variable(tf.random_normal(image_shape,
                                                  dtype=tf.float32,
                                                  stddev=1e-1))
        #images, labels = distorted_inputs()

        logits = inference(images,
                           num_classes=1000,
                           is_training=True,
                           bottleneck=True) # use default: resnet-50
                           #num_blocks=[2, 2, 2, 2])

        train(True, logits, images, labels)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--epochs", help="the number of epochs", type=int, default=4)
    parser.add_argument("-b", "--minibatch", help="minibatch size", type=int, default=16)
    parser.add_argument("-i", "--iterations", help="iterations", type=int, default=2)
    parser.add_argument("-d", "--deviceid", help="specified device id", type=int, default=0)
    args = parser.parse_args()

    epochs = args.epochs 
    minibatch = args.minibatch 
    iterations = args.iterations 
    device_id = args.deviceid 
    set_parameters(epochs, minibatch, iterations, device_id)
    tf.app.run()


