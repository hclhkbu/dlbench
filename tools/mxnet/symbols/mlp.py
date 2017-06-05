import mxnet as mx

def get_symbol(num_classes=10, **kwargs):
    data = mx.symbol.Variable('data')
    data = mx.sym.Flatten(data=data)
    fc1  = mx.symbol.FullyConnected(data = data, name='fc1', num_hidden=2048)
    act1 = mx.symbol.Activation(data = fc1, name='relu1', act_type="sigmoid")
    fc2  = mx.symbol.FullyConnected(data = act1, name = 'fc2', num_hidden = 4096)
    act2 = mx.symbol.Activation(data = fc2, name='relu2', act_type="sigmoid")
    fc3  = mx.symbol.FullyConnected(data = act2, name = 'fc3', num_hidden = 1024)
    act3 = mx.symbol.Activation(data = fc3, name='relu3', act_type="sigmoid")
    fc4  = mx.symbol.FullyConnected(data = act3, name='fc4', num_hidden=num_classes)
    mlp  = mx.symbol.SoftmaxOutput(data = fc4, name = 'softmax')
    return mlp
