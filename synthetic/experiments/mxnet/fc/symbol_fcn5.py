import mxnet as mx

def get_symbol(num_classes = 26752):
    input_data = mx.symbol.Variable(name="data")
    fc1 = mx.symbol.FullyConnected(data = input_data, name='fc1', num_hidden=2048)
    act1 = mx.symbol.Activation(data=fc1, name='sigmoid1', act_type="sigmoid")
    fc2 = mx.symbol.FullyConnected(data = act1, name='fc2', num_hidden=2048)
    act2 = mx.symbol.Activation(data=fc2, name='sigmoid2', act_type="sigmoid")
    fc3 = mx.symbol.FullyConnected(data = act2, name='fc3', num_hidden=2048)
    act3 = mx.symbol.Activation(data=fc3, name='sigmoid3', act_type="sigmoid")
    fc4 = mx.symbol.FullyConnected(data=act3, num_hidden=num_classes)
    softmax = mx.symbol.SoftmaxOutput(data=fc4, name='softmax')
    return softmax
