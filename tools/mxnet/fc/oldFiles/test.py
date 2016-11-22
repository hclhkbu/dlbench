import mxnet as mx
import logging
import numpy as np

data = mx.symbol.Variable(name = 'data')
H1 = mx.symbol.FullyConnected(data = data, num_hidden = 2048)
H1_A = mx.symbol.Activation(data = H1, act_type="sigmoid")
H2 = mx.symbol.FullyConnected(data = H1_A, num_hidden = 2048)
H2_A = mx.symbol.Activation(data = H2, act_type="sigmoid")
H3 = mx.symbol.FullyConnected(data = H2_A, num_hidden = 2048)
H3_A = mx.symbol.Activation(data = H3, act_type="sigmoid")
L = mx.symbol.FullyConnected(data = H3_A, num_hidden = 26752)
loss = mx.symbol.SoftmaxOutput(data = L, name = 'softmax')

train = mx.io.NDArrayIter(np.random.randn(10240, 26752), np.random.randint(0, 26752, size=(10240, 1)), batch_size=1024)

logging.basicConfig(level=logging.DEBUG)

model = mx.model.FeedForward(
        ctx = mx.cpu(), symbol = loss, num_epoch = 10,
        learning_rate = 0.1, momentum = 0.9, wd = 0.00001)

model.fit(X=train)
