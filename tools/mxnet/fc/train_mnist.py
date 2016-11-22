import os, sys
sys.path[1] = '/home/comp/pengfeixu/.local/lib/python2.7/site-packages/mxnet-0.7.0-py2.7.egg'
import mxnet as mx
import argparse
import train_model


def get_mlp():
    """
    multi-layer perceptron, tested accuracy over 94%
    """
    data = mx.symbol.Variable('data')
    fc1  = mx.symbol.FullyConnected(data = data, name='fc1', num_hidden=2048)
    act1 = mx.symbol.Activation(data = fc1, name='relu1', act_type="sigmoid")
    fc2  = mx.symbol.FullyConnected(data = act1, name = 'fc2', num_hidden = 4096)
    act2 = mx.symbol.Activation(data = fc2, name='relu2', act_type="sigmoid")
    fc3  = mx.symbol.FullyConnected(data = act2, name = 'fc3', num_hidden = 1024)
    act3 = mx.symbol.Activation(data = fc3, name='relu3', act_type="sigmoid")
    fc4  = mx.symbol.FullyConnected(data = act3, name='fc4', num_hidden=10)
    mlp  = mx.symbol.SoftmaxOutput(data = fc4, name = 'softmax')
    return mlp


def get_iterator(data_shape):
    def get_iterator_impl(args, kv):
        data_dir = args.data_dir
        flat = False if len(data_shape) == 3 else True

        train           = mx.io.MNISTIter(
            image       = data_dir + "train-images-idx3-ubyte",
            label       = data_dir + "train-labels-idx1-ubyte",
            input_shape = data_shape,
            batch_size  = args.batch_size,
            shuffle     = True,
            flat        = flat,
            num_parts   = kv.num_workers,
            part_index  = kv.rank)

#        val = mx.io.MNISTIter(
#            image       = data_dir + "t10k-images-idx3-ubyte",
#            label       = data_dir + "t10k-labels-idx1-ubyte",
#            input_shape = data_shape,
#            batch_size  = args.batch_size,
#            flat        = flat,
#            num_parts   = kv.num_workers,
#            part_index  = kv.rank)

        #return (train, val)
        return (train, None)
    return get_iterator_impl

def parse_args():
    parser = argparse.ArgumentParser(description='train an image classifer on mnist')
    parser.add_argument('--network', type=str, default='mlp',
                        choices = ['mlp'],
                        help = 'the cnn to use')
    parser.add_argument('--data-dir', type=str, default= os.environ['HOME'] + '/data/mxnet/mnist/',
                        help='the input data directory')
    parser.add_argument('--num-nodes', type=int, default=1,
                        help='number of nodes')
    parser.add_argument('--gpus', type=str,
                        help='the gpus will be used, e.g "0,1,2,3"')
    parser.add_argument('--num-examples', type=int, default=50000,
                        help='the number of training examples')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='the batch size')
    parser.add_argument('--lr', type=float, default=.1,
                        help='the initial learning rate')
    parser.add_argument('--model-prefix', type=str,
                        help='the prefix of the model to load/save')
    parser.add_argument('--save-model-prefix', type=str,
                        help='the prefix of the model to save')
    parser.add_argument('--num-epochs', type=int, default=10,
                        help='the number of training epochs')
    parser.add_argument('--load-epoch', type=int,
                        help="load the model on an epoch using the model-prefix")
    parser.add_argument('--kv-store', type=str, default='local',
                        help='the kvstore type')
    parser.add_argument('--lr-factor', type=float, default=1,
                        help='times the lr with a factor for every lr-factor-epoch epoch')
    parser.add_argument('--lr-factor-epoch', type=float, default=1,
                        help='the number of epoch to factor the lr, could be .5')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()


    data_shape = (784, )
    net = get_mlp()
    # train
    train_model.fit(args, net, get_iterator(data_shape))
