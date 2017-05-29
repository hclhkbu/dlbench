"""
Train mnist, see more explanation at http://mxnet.io/tutorials/python/mnist.html
"""
import os
import argparse
import logging
logging.basicConfig(level=logging.DEBUG)
from common import find_mxnet, fit
from common.util import download_file
import mxnet as mx
import numpy as np
import gzip, struct


def read_data(data_path, label, image):
    #with gzip.open(data_path + label) as flbl:
    with open(data_path + label) as flbl:
        magic, num = struct.unpack(">II", flbl.read(8))
        label = np.fromstring(flbl.read(), dtype=np.int8)
    #with gzip.open(data_path + image) as fimg:
    with open(data_path + image) as fimg:
        magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
        image = np.fromstring(fimg.read(), dtype=np.uint8).reshape(len(label), rows, cols)
    return (label, image)


def to4d(img):
    """
    reshape to 4D arrays
    """
    return img.reshape(img.shape[0], 1, 28, 28)#.astype(np.float32)/255

def get_mnist_iter(args, kv):
    """
    create data iterator with NDArrayIter
    """
    #(train_lbl, train_img) = read_data(args.data_dir, 'train-labels-idx1-ubyte.gz', 'train-images-idx3-ubyte.gz')
    #(val_lbl, val_img) = read_data(args.data_dir, 't10k-labels-idx1-ubyte.gz', 't10k-images-idx3-ubyte.gz')
    (train_lbl, train_img) = read_data(args.data_dir, 'train-labels-idx1-ubyte', 'train-images-idx3-ubyte')
    (val_lbl, val_img) = read_data(args.data_dir, 't10k-labels-idx1-ubyte', 't10k-images-idx3-ubyte')
    train = mx.io.NDArrayIter( to4d(train_img), train_lbl, args.batch_size, shuffle=True)
    val = mx.io.NDArrayIter( to4d(val_img), val_lbl, args.batch_size)
    return (train, val)

if __name__ == '__main__':
    # parse args
    parser = argparse.ArgumentParser(description="train mnist", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--num-classes', type=int, default=10, help='the number of classes')
    parser.add_argument('--num-examples', type=int, default=60000, help='the number of training examples')
    parser.add_argument('--data-dir', type=str, default=os.environ['HOME'] + '/data/mxnet/mnist/')
    fit.add_fit_args(parser)
    parser.set_defaults(
        # network
        network        = 'mlp',
        # train
        gpus           = None,
        batch_size      = 64,
        disp_batches = 100,
        num_epochs     = 20,
        lr             = .05,
        lr_step_epochs = '10',
    )
    args = parser.parse_args()

    # load network
    from importlib import import_module
    net = import_module('symbols.'+args.network)
    sym = net.get_symbol(**vars(args))

    # train
    fit.fit(args, sym, get_mnist_iter, mx.init.Xavier(factor_type="in", magnitude=3))
