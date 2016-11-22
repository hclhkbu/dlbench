import cPickle
import numpy as np
import tensorflow as tf


PATH = './cifar-10-batches-py'
TARGETPATH = '/home/comp/csshshi/tensorflow/cifar-10-batches-py'
TEST_FILES = ['test_batch']
FILES = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5']
TRAIN_COUNT = 50000
EVAL_COUNT = 10000
IMAGE_SIZE = 32 
NUM_CLASSES = 10

unpickled = {}

def unpickle(file):
    dict = unpickled.get(file)
    if dict:
        return dict
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    unpickled[file] = dict
    return dict


def get_next_batch(batch_size, step, is_test=False):
    files = FILES
    if is_test:
        files = TEST_FILES
    file_index = step % len(FILES)
    filename = files[file_index]
    filename = '%s/%s'%(PATH, filename)
    dict = unpickle(filename)
    data_index = step/len(files) * batch_size
    images = dict['data'][data_index:data_index+batch_size]
    labels = dict['labels'][data_index:data_index+batch_size]
    reshaped_images = [np.reshape(image, (IMAGE_SIZE, IMAGE_SIZE, 3)) for image in images]
    return reshaped_images, labels
