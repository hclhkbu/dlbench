"""
Reference:
    https://code.google.com/p/cuda-convnet/source/browse/trunk/example-layers/layers-18pct.cfg
"""
import mxnet as mx

def get_symbol(num_classes = 1000):
    input_data = mx.symbol.Variable(name="data")
    # stage 1
    conv1 = mx.symbol.Convolution( data=input_data, pad=(1, 1), kernel=(11, 11), stride=(4, 4), num_filter=96)
    relu1 = mx.symbol.Activation(data=conv1, act_type="relu")
    pool1 = mx.symbol.Pooling( data=relu1, pool_type="max", kernel=(3, 3), stride=(2,2))
    # stage 2
    conv2 = mx.symbol.Convolution(data=pool1, kernel=(5, 5), pad=(1, 1), num_filter=256)
    relu2 = mx.symbol.Activation(data=conv2, act_type="relu")
    pool2 = mx.symbol.Pooling(data=relu2, kernel=(3, 3), stride=(2, 2), pool_type="max")
    # stage 3
    conv3 = mx.symbol.Convolution(data=pool2, kernel=(3, 3), pad=(1, 1), num_filter=384)
    relu3 = mx.symbol.Activation(data=conv3, act_type="relu")
    conv4 = mx.symbol.Convolution(data=relu3, kernel=(3, 3), pad=(1, 1), num_filter=384)
    relu4 = mx.symbol.Activation(data=conv4, act_type="relu")
    conv5 = mx.symbol.Convolution(data=relu3, kernel=(3, 3), pad=(1, 1), num_filter=256)
    relu5 = mx.symbol.Activation(data=conv5, act_type="relu")
    pool5 = mx.symbol.Pooling(data=relu3, kernel=(3, 3), stride=(2, 2), pool_type="max")
    fc6 = mx.symbol.FullyConnected(data=pool5, num_hidden=4096)
    relu6 = mx.symbol.Activation(data=fc6, act_type="relu")
    fc7 = mx.symbol.FullyConnected(data=relu6, num_hidden=4096)
    relu7 = mx.symbol.Activation(data=fc7, act_type="relu")
    fc8 = mx.symbol.FullyConnected(data=pool5, num_hidden=num_classes)
    softmax = mx.symbol.SoftmaxOutput(data=fc8, name='softmax')
    return softmax
