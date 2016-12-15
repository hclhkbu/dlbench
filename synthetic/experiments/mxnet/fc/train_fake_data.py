import sys
import mxnet as mx
import os
import argparse
import train_model

parser = argparse.ArgumentParser(description='train fcn5 and fcn8 with fake data')
parser.add_argument('--data-dir', type=str, default= os.environ['HOME'] + '/data/mxnet/fakedata/',
                    help='the input data directory')
parser.add_argument('--num-nodes', type=int, default=1,
                    help='number of nodes')
parser.add_argument('--gpus', type=str,
                    help='the gpus will be used, e.g "0,1,2,3"')
parser.add_argument('--num-examples', type=int, default=26752,
                    help='the number of training examples')
parser.add_argument('--batch-size', type=int, default=1024,
                    help='the batch size')
parser.add_argument('--lr', type=float, default=.05,
                    help='the initial learning rate')
parser.add_argument('--lr-factor', type=float, default=1,
                    help='times the lr with a factor for every lr-factor-epoch epoch')
parser.add_argument('--lr-factor-epoch', type=float, default=1,
                    help='the number of epoch to factor the lr, could be .5')
parser.add_argument('--model-prefix', type=str,
                    help='the prefix of the model to load')
parser.add_argument('--save-model-prefix', type=str,
                    help='the prefix of the model to save')
parser.add_argument('--num-epochs', type=int, default=20,
                    help='the number of training epochs')
parser.add_argument('--load-epoch', type=int,
                    help="load the model on an epoch using the model-prefix")
parser.add_argument('--kv-store', type=str, default='device',
                    help='the kvstore type')
parser.add_argument('--network', type=str, default='fcn5',
                    help='the kvstore type')
args = parser.parse_args()

# network
import importlib
net = importlib.import_module('symbol_' + args.network).get_symbol(26752) 

# data
def get_iterator(args, kv):
    data_shape = (1, 1, 26752)

    train = mx.io.ImageRecordIter(
        path_imgrec = args.data_dir + "train.rec",
        mean_img    = args.data_dir + "mean.bin",
        data_shape  = data_shape,
	preprocess_threads = 20,
        batch_size  = args.batch_size,
        rand_crop   = False,
        rand_mirror = False,
        num_parts   = kv.num_workers,
        part_index  = kv.rank)

    return (train, None)

# train
train_model.fit(args, net, get_iterator)
