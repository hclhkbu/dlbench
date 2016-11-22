import mxnet as mx


def get_net():	data = mx.symbol.Variable(name = 'data')
	conv1 = mx.symbol.Convolution(data = data, kernel = (11, 11), stride = (4, 4), num_filter = 96)
	relu1 = mx.symbol.Activation(data = conv1, act_type="relu")
	pool1 = mx.symbol.Pooling(data = relu1, kernel = (3,3), stride = (2,2), pool_type = "max")
	conv2 = mx.symbol.Convolution(data = pool1, kernel = (5, 5), num_filter = 256)
	relu2 = mx.symbol.Activation(data = conv2, act_type="relu")
	pool2 = mx.symbol.Pooling(data = relu2, kernel = (3,3), stride = (2,2), pool_type = "max")
	conv3 = mx.symbol.Convolution(data = pool2, kernel = (3, 3), num_filter = 384)
	relu3 = mx.symbol.Activation(data = conv3, act_type="relu")
	conv4 = mx.symbol.Convolution(data = relu3, kernel = (3, 3), num_filter = 384)
	relu4 = mx.symbol.Activation(data = conv4, act_type="relu")
	conv5 = mx.symbol.Convolution(data = relu4, kernel = (3, 3), num_filter = 256)
	relu5 = mx.symbol.Activation(data = conv5, act_type="relu")
	pool5 = mx.symbol.Pooling(data = relu5, kernel = (3,3), stride = (2,2), pool_type = "max")
	fc6 = mx.symbol.FullyConnected(data = pool5, num_hidden = 4096)
	relu6 = mx.symbol.Activation(data = fc6, act_type="relu")
	fc7 = mx.symbol.FullyConnected(data = relu6, num_hidden = 4096)
	relu7 = mx.symbol.Activation(data = fc7, act_type="relu")
	fc8 = mx.symbol.FullyConnected(data = relu7, num_hidden = 1000)
	loss = mx.symbol.SoftmaxOutput(data = label, name = 'softmax')
	return loss