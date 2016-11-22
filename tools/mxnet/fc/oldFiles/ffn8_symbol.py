import mxnet as mx

def get_ffn(num_class=10):
    data = mx.symbol.Variable(name = 'data')
#    H1 = mx.symbol.FullyConnected(data = data, num_hidden = 64)
#    H1_A = mx.symbol.Activation(data = H1, act_type="sigmoid")
#    H2 = mx.symbol.FullyConnected(data = H1_A, num_hidden = 128)
#    H2_A = mx.symbol.Activation(data = H2, act_type="sigmoid")
#    H3 = mx.symbol.FullyConnected(data = data, num_hidden = 1024)
#    H3_A = mx.symbol.Activation(data = H3, act_type="sigmoid")
    H4 = mx.symbol.FullyConnected(data = data, num_hidden = 32768)
    H4_A = mx.symbol.Activation(data = H4, act_type="sigmoid")
    H5 = mx.symbol.FullyConnected(data = H4_A, num_hidden = 4096 )
    H5_A = mx.symbol.Activation(data = H5, act_type="sigmoid")
    H6 = mx.symbol.FullyConnected(data = H5_A, num_hidden = 2048)
    H6_A = mx.symbol.Activation(data = H6, act_type="sigmoid")
    H7 = mx.symbol.FullyConnected(data = H6_A, num_hidden = 768)
    H7_A = mx.symbol.Activation(data = H7, act_type="sigmoid")
    L = mx.symbol.FullyConnected(data = H7_A, num_hidden = num_class)
    loss = mx.symbol.SoftmaxOutput(data = L, name = 'softmax')
    return loss
