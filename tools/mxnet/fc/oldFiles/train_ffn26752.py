import mxnet as mx
import numpy as np
import argparse
import ffn26752_symbol as ffn 
import train_model

dataPath = "/home/dl/data/mxnet/ffn26752.bin"

def load_data(path):
    raw = mx.nd.load(path)
    features = raw['features']
    labels = raw['lables']
    return features, labels

def parse_args():
    parser = argparse.ArgumentParser(description='train an image classifer on mnist')
    parser.add_argument('--data-dir', type=str, default='mnist/',
            help='the input data directory')
    parser.add_argument('--gpus', type=str,
            help='the gpus will be used, e.g "0,1,2,3"')
    parser.add_argument('--num-examples', type=int, default=61440,
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

if __name__=='__main__':
    args = parse_args()
    net = ffn.get_ffn26752()
#    features, labels = load_data(dataPath)
    features = np.random.randn(1024, 26752)
    labels = np.random.randint(0, 26752, size=(1024, 1))
    itr = mx.io.NDArrayIter(data=features, label=labels, batch_size=args.batch_size, shuffle=True)
    train_model.fit(args, net, itr)

