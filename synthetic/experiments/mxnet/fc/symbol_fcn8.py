import mxnet as mx

def get_symbol(num_classes = 26752):
    input_data = mx.symbol.Variable(name="data")
    fc1 = mx.symbol.FullyConnected(data = input_data, name='fc1', num_hidden=2048)
    act1 = mx.symbol.Activation(data=fc1, name='sigmoid1', act_type="sigmoid")
    fc2 = mx.symbol.FullyConnected(data = act1, name='fc2', num_hidden=2048)
    act2 = mx.symbol.Activation(data=fc2, name='sigmoid2', act_type="sigmoid")
    fc3 = mx.symbol.FullyConnected(data = act2, name='fc4', num_hidden=2048)
    act3 = mx.symbol.Activation(data=fc3, name='sigmoid3', act_type="sigmoid")
    fc4 = mx.symbol.FullyConnected(data = act3, name='fc5', num_hidden=2048)
    act4 = mx.symbol.Activation(data=fc4, name='sigmoid4', act_type="sigmoid")
    fc5 = mx.symbol.FullyConnected(data = act4, name='fc6', num_hidden=2048)
    act5 = mx.symbol.Activation(data=fc5, name='sigmoid5', act_type="sigmoid")
    fc6 = mx.symbol.FullyConnected(data = act5, name='fc7', num_hidden=2048)
    act6 = mx.symbol.Activation(data=fc6, name='sigmoid6', act_type="sigmoid")
    fc7 = mx.symbol.FullyConnected(data=act6, num_hidden=num_classes)
    softmax = mx.symbol.SoftmaxOutput(data=fc7, name='softmax')
    return softmax
