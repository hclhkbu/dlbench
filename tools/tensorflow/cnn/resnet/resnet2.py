def _build_graph(self, input_vars):
    image, label = input_vars
    image = image / 128.0 - 1

    def residual(name, l, increase_dim=False, first=False):
        shape = l.get_shape().as_list()
        in_channel = shape[3]

        if increase_dim:
            out_channel = in_channel * 2
            stride1 = 2
        else:
            out_channel = in_channel
            stride1 = 1

        with tf.variable_scope(name) as scope:
            if not first:
                b1 = BatchNorm('bn1', l)
                b1 = tf.nn.relu(b1)
            else:
                b1 = l
                c1 = Conv2D('conv1', b1, out_channel, stride=stride1)
                b2 = BatchNorm('bn2', c1)
                b2 = tf.nn.relu(b2)
                c2 = Conv2D('conv2', b2, out_channel)

            if increase_dim:
                l = AvgPooling('pool', l, 2)
                l = tf.pad(l, [[0,0], [0,0], [0,0], [in_channel//2, in_channel//2]])

                l = c2 + l
        return l

    with argscope(Conv2D, nl=tf.identity, use_bias=False, kernel_shape=3,
            W_init=variance_scaling_initializer(mode='FAN_OUT')):
        l = Conv2D('conv0', image, 16)
    l = BatchNorm('bn0', l)
        l = tf.nn.relu(l)
        l = residual('res1.0', l, first=True)
        for k in range(1, self.n):
            l = residual('res1.{}'.format(k), l)
# 32,c=16

            l = residual('res2.0', l, increase_dim=True)
            for k in range(1, self.n):
                l = residual('res2.{}'.format(k), l)
# 16,c=32

                l = residual('res3.0', l, increase_dim=True)
                for k in range(1, self.n):
                    l = residual('res3.' + str(k), l)
    l = BatchNorm('bnlast', l)
                    l = tf.nn.relu(l)
# 8,c=64
                    l = GlobalAvgPooling('gap', l)

                    logits = FullyConnected('linear', l, out_dim=10, nl=tf.identity)
                    prob = tf.nn.softmax(logits, name='output')

cost = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, label)
    cost = tf.reduce_mean(cost, name='cross_entropy_loss')
