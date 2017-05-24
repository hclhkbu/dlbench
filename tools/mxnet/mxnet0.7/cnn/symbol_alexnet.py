"""
Reference:
    https://code.google.com/p/cuda-convnet/source/browse/trunk/example-layers/layers-18pct.cfg
"""
import mxnet as mx

def get_symbol(num_classes = 10):
    input_data = mx.symbol.Variable(name="data")
    # stage 1
    conv1 = mx.symbol.Convolution( data=input_data, pad=(2, 2), kernel=(5, 5), stride=(1, 1), num_filter=32)
#    conv1 = mx.symbol.Convolution( data=input_data, pad=(5, 5), kernel=(5, 5), stride=(1, 1), num_filter=32)
    pool1 = mx.symbol.Pooling( data=conv1, pool_type="max", kernel=(3, 3), stride=(2,2))
    relu1 = mx.symbol.Activation(data=pool1, act_type="relu")
#    lrn1 = mx.symbol.LRN(data=relu1, alpha=0.00005, beta=0.75, knorm=1, nsize=3)
    # stage 2
    conv2 = mx.symbol.Convolution(data=relu1, kernel=(5, 5), pad=(2, 2), num_filter=32)
    relu2 = mx.symbol.Activation(data=conv2, act_type="relu")
    pool2 = mx.symbol.Pooling(data=relu2, kernel=(3, 3), stride=(2, 2), pool_type="max")
#    lrn2 = mx.symbol.LRN(data=pool2, alpha=0.00005, beta=0.75, knorm=1, nsize=3)
    # stage 3
    conv3 = mx.symbol.Convolution(data=pool2, kernel=(5, 5), pad=(2, 2), num_filter=64)
    relu3 = mx.symbol.Activation(data=conv3, act_type="relu")
    pool3 = mx.symbol.Pooling(data=relu3, kernel=(3, 3), stride=(2, 2), pool_type="avg")
    fc3 = mx.symbol.FullyConnected(data=pool3, num_hidden=num_classes)
    softmax = mx.symbol.SoftmaxOutput(data=fc3, name='softmax')
    return softmax
