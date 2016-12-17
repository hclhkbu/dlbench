'''
Reproducing https://github.com/gcr/torch-residual-networks
For image size of 32x32

Test accuracy are 0.9309 and 0.9303 in this patch and Kaiming He's paper, respectively.
The accuracy is the best one of the last 3 epochs (0.930288, 0.930889 and 0.929587),
while the original paper select the best one in 5 runs.
The dockerfile and log are in: https://gist.github.com/Answeror/f9160145e1c64bb509f52c00014bdb77

The only difference between this patch and Facebook's implementation
(https://github.com/gcr/torch-residual-networks and https://github.com/facebook/fb.resnet.torch) are:

1. The kernel of shortcut with downsampling is 2x2 rather than 1x1.
   I can't reproduce this accuracy with 1x1 kernel. Note the shortcut does not contain learnable parameters.
2. I use a BatchNorm after data layer to simulate z-score normalization.
   Although subtract (127, 127, 127) and divide 60 works equally well.
3. An eps of 2e-5 is used in BatchNorm instead of 1e-5 because cuDNN v5 don't allow such small eps.

Some details affect the accuracy:

1. Z-score normalization of the input.
2. Weight decay of all parameters (weight, bias, gamma, beta). See comments in `train_cifar10_resnet.py `for details.
3. Nesterov momentum
4. `fix_gamma=False` in BatchNorm (gamma is necessary because of the weight decay of the conv weight)
5. Initialization
6. 4 pixel padding

And thanks #1230 (@freesouls) and #1041 (@shuokay) to provide preliminary implementations.

## update@2016-06-08

With #2366 and a batch size of 64, I got an accuracy of 0.939704 after 200 epochs on 2 GPUs.
Note, **the accuracy is strongly affected by the batch size**, the more GPU you use, the smaller batch size should be.
See https://gist.github.com/Answeror/f9160145e1c64bb509f52c00014bdb77#file-resnet-dual-gpu-log for the full log.

References:

Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun. "Deep Residual Learning for Image Recognition"
'''
from __future__ import division
import sys
import argparse
import math
import os
os.environ["MXNET_CUDNN_AUTOTUNE_DEFAULT"] = "1"
import mxnet as mx
import logging
import numpy as np


parser = argparse.ArgumentParser(description='train an image classifer on cifar10')
parser.add_argument('--data-dir', type=str, default= os.environ['HOME'] + '/data/mxnet/imagenet/',
                    help='the input data directory')
parser.add_argument('--num-nodes', type=int, default=1,
                    help='number of nodes')
parser.add_argument('--optimizer', type=str, default="ccSGD",
		    help='Optimizer options: Nesterov | SGD | NAG ... see http://mxnet.io/api/python/model.html')
parser.add_argument('--gpus', type=str,
                    help='the gpus will be used, e.g "0,1,2,3"')
parser.add_argument('--num-examples', type=int, default=1281167,
                    help='the number of training examples')
parser.add_argument('--batch-size', type=int, default=32,
                    help='the batch size')
parser.add_argument('--lr', type=float, default=0.1,
                    help='the initial learning rate')
parser.add_argument('--model-prefix', type=str,
                    help='the prefix of the model to load')
parser.add_argument('--save-model-prefix', type=str,
                    help='the prefix of the model to save')
parser.add_argument('--num-epochs', type=int, default=4,
                    help='the number of training epochs')
parser.add_argument('--load-epoch', type=int,
                    help='load the model on an epoch using the model-prefix')
parser.add_argument('--kv-store', type=str, default='local',
                    help='the kvstore type')
args = parser.parse_args()


# network
from symbol_resnet import get_symbol
net = get_symbol(num_classes=1000, num_layers=50, image_shape='3, 224, 224', conv_workspace=512)


# data
def get_iterator(args, kv):
    kargs = dict(
        data_shape=(3, 224, 224),
        # Use mean and scale works equally well
        # We use BatchNorm after data for simplicity
        # mean_r=127,
        # mean_g=127,
        # mean_b=127,
        # scale=1 / 60
    )

    train = mx.io.ImageRecordIter(
        path_imgrec=args.data_dir + 'train.rec',
        batch_size=args.batch_size,
	preprocess_threads=16,
        rand_mirror=False,
	shuffle = True,
	shuffle_chunk_size = 32,
        num_parts=kv.num_workers,
        part_index=kv.rank,
        **kargs
    )
#    val = mx.io.ImageRecordIter(
#        path_imgrec=args.data_dir + 'test.rec',
#        rand_crop=False,
#        rand_mirror=False,
#        batch_size=args.batch_size,
#        num_parts=kv.num_workers,
#        part_index=kv.rank,
#        **kargs
#    )

    #return (train, val)
    return (train, None)


class Init(mx.init.Xavier):

    def __call__(self, name, arr):
        """Override () function to do Initialization

        Parameters
        ----------
        name : str
            name of corrosponding ndarray

        arr : NDArray
            ndarray to be Initialized
        """
        if not isinstance(name, mx.base.string_types):
            raise TypeError('name must be string')
        if not isinstance(arr, mx.ndarray.NDArray):
            raise TypeError('arr must be NDArray')
        if 'proj' in name and name.endswith('weight'):
            self._init_proj(name, arr)
        elif name.endswith('upsampling'):
            self._init_bilinear(name, arr)
        elif name.endswith('bias'):
            self._init_bias(name, arr)
        elif name.endswith('gamma'):
            self._init_gamma(name, arr)
        elif name.endswith('beta'):
            self._init_beta(name, arr)
        elif name.endswith('weight'):
            self._init_weight(name, arr)
        elif name.endswith("moving_mean"):
            self._init_zero(name, arr)
        elif name.endswith("moving_var"):
            self._init_zero(name, arr)
        elif name.endswith("moving_inv_var"):
            self._init_zero(name, arr)
        elif name.endswith("moving_avg"):
            self._init_zero(name, arr)
        else:
            self._init_default(name, arr)

    def _init_proj(self, _, arr):
        '''Initialization of shortcut of kenel (2, 2)'''
        w = np.zeros(arr.shape, np.float32)
        for i in range(w.shape[1]):
            w[i, i, ...] = 0.25
        arr[:] = w


class Scheduler(mx.lr_scheduler.MultiFactorScheduler):

    def __init__(self, epoch_step, factor, epoch_size):
        super(Scheduler, self).__init__(
            step=[epoch_size * s for s in epoch_step],
            factor=factor
        )

def fit(args, network, data_loader, batch_end_callback=None):
    # kvstore
    kv = mx.kvstore.create(args.kv_store)

    # logging
    head = '%(asctime)-15s Node[' + str(kv.rank) + '] %(message)s'
    if 'log_file' in args and args.log_file is not None:
        log_file = args.log_file
        log_dir = args.log_dir
        log_file_full_name = os.path.join(log_dir, log_file)
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)
        logger = logging.getLogger()
        handler = logging.FileHandler(log_file_full_name)
        formatter = logging.Formatter(head)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
        logger.info('start with arguments %s', args)
    else:
        logging.basicConfig(level=logging.DEBUG, format=head)
        logging.info('start with arguments %s', args)

    # load model
    model_prefix = args.model_prefix
    if model_prefix is not None:
        model_prefix += '-%d' % (kv.rank)
    model_args = {}
    if args.load_epoch is not None:
        assert model_prefix is not None
        tmp = mx.model.FeedForward.load(model_prefix, args.load_epoch)
        model_args = {'arg_params': tmp.arg_params,
                      'aux_params': tmp.aux_params,
                      'begin_epoch': args.load_epoch}
    # save model
    save_model_prefix = args.save_model_prefix
    if save_model_prefix is None:
        save_model_prefix = model_prefix
    checkpoint = None if save_model_prefix is None else mx.callback.do_checkpoint(save_model_prefix)

    # data
    (train, val) = data_loader(args, kv)

    # train
    devs = mx.cpu() if args.gpus is None else [
        mx.gpu(int(i)) for i in args.gpus.split(',')]

    epoch_size = args.num_examples / args.batch_size

    if args.kv_store == 'dist_sync':
        epoch_size /= kv.num_workers
        model_args['epoch_size'] = epoch_size

    if 'clip_gradient' in args and args.clip_gradient is not None:
        model_args['clip_gradient'] = args.clip_gradient

    # disable kvstore for single device
    if 'local' in kv.type and (
            args.gpus is None or len(args.gpus.split(',')) is 1):
        kv = None
    model = mx.model.FeedForward(
        ctx=devs,
        symbol=network,
        num_epoch=args.num_epochs,
        learning_rate=args.lr,
        momentum=0.9,
        wd=0.0001,
	optimizer=args.optimizer,
	epoch_size = epoch_size,
        # Note we initialize BatchNorm beta and gamma as that in
        # https://github.com/facebook/fb.resnet.torch/
        # i.e. constant 0 and 1, rather than
        # https://github.com/gcr/torch-residual-networks/blob/master/residual-layers.lua
        # FC layer is initialized as that in torch default
        # https://github.com/torch/nn/blob/master/Linear.lua
    #    initializer=mx.init.Mixed(
    #        ['.*fc.*', '.*'],
    #        [mx.init.Xavier(rnd_type='uniform', factor_type='in', magnitude=1),
    #         Init(rnd_type='gaussian', factor_type='in', magnitude=1)]
    #    ),
        #lr_scheduler=Scheduler(epoch_step=[80, 160], factor=0.1, epoch_size=epoch_size),
        **model_args)

    eval_metrics = ['accuracy', 'ce']

    if batch_end_callback is not None:
        if not isinstance(batch_end_callback, list):
            batch_end_callback = [batch_end_callback]
    else:
        batch_end_callback = []
    #batch_end_callback.append(mx.callback.Speedometer(args.batch_size, int((args.num_examples/args.num_nodes)/args.batch_size)))
    batch_end_callback.append(mx.callback.Speedometer(args.batch_size, 1))

    model.fit(
        X=train,
#        eval_data=val,
        eval_metric=eval_metrics,
        kvstore=kv,
        batch_end_callback=batch_end_callback,
        epoch_end_callback=checkpoint
    )

# train
fit(args, net, get_iterator)
