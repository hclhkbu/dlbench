import mxnet as mx

def get_ffn(num_class=10):
    data = mx.symbol.Variable(name = 'data')
    H1 = mx.symbol.FullyConnected(data = data, num_hidden = 512)
    H1_A = mx.symbol.Activation(data = H1, act_type="sigmoid")
    H2 = mx.symbol.FullyConnected(data = H1_A, num_hidden = 1024)
    H2_A = mx.symbol.Activation(data = H2, act_type="sigmoid")
    H3 = mx.symbol.FullyConnected(data = H2_A, num_hidden = 512)
    H3_A = mx.symbol.Activation(data = H3, act_type="sigmoid")
    L = mx.symbol.FullyConnected(data = H3_A, num_hidden = num_class)
    loss = mx.symbol.SoftmaxOutput(data = L, name = 'softmax')
    return loss
