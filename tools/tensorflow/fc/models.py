import tensorflow as tf

#feature_dim = 32*32*3
feature_dim = 28*28 
label_dim = 10 
hidden_layer_dim = 2048
num_minibatch = 100


# Get random parameters initialized with a iniform distribution between -0.5 and 0.5
def get_variable(name, shape, is_bias=False):
    #return tf.get_variable(name, shape, initializer=tf.random_uniform_initializer(-0.5, 0.5))
    if is_bias:
        #return tf.get_variable(name, shape, initializer=tf.random_uniform_initializer(0.1, 0.5))
        return tf.get_variable(name, shape, initializer=tf.constant_initializer(0))
    #return tf.get_variable(name, shape, initializer=tf.contrib.layers.xavier_initializer())
    return tf.get_variable(name, shape, initializer=tf.contrib.layers.xavier_initializer())


def sigmoid_DNN_layer(layer_idx, input, input_dim, output_dim):
    W = get_variable("W" + str(layer_idx), [input_dim, output_dim])
    B = get_variable("B" + str(layer_idx), [output_dim], is_bias=True)
    return tf.nn.sigmoid(tf.nn.xw_plus_b(input, W, B))


def model_fcn5(features):
    HL0 = sigmoid_DNN_layer(0, features, feature_dim, 2048)
    HL1 = sigmoid_DNN_layer(1, HL0, 2048, 4096)
    HL2 = sigmoid_DNN_layer(2, HL1, 4096, 1024)

    outputLayerW = get_variable("W5", [1024, label_dim])
    outputLayerB = get_variable("B5", [label_dim], is_bias=True)
    outputLayer = tf.nn.xw_plus_b(HL2, outputLayerW, outputLayerB)
    #outputLayer = tf.nn.softmax(tf.nn.xw_plus_b(HL2, outputLayerW, outputLayerB))
    return outputLayer 


def model_fcn8(features):
    HL0 = sigmoid_DNN_layer(0, features, feature_dim, hidden_layer_dim)
    HL1 = sigmoid_DNN_layer(1, HL0, hidden_layer_dim, hidden_layer_dim)
    HL2 = sigmoid_DNN_layer(2, HL1, hidden_layer_dim, hidden_layer_dim)
    HL3 = sigmoid_DNN_layer(3, HL2, hidden_layer_dim, hidden_layer_dim)
    HL4 = sigmoid_DNN_layer(4, HL3, hidden_layer_dim, hidden_layer_dim)
    HL5 = sigmoid_DNN_layer(5, HL4, hidden_layer_dim, hidden_layer_dim)

    outputLayerW = get_variable("W8", [hidden_layer_dim, label_dim])
    outputLayerB = get_variable("B8", [label_dim])
    outputLayer = tf.nn.xw_plus_b(HL5, outputLayerW, outputLayerB)
    return outputLayer 


def loss(logits, labels):
    labels = tf.cast(labels, tf.float32)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, labels)
    loss = tf.reduce_mean(cross_entropy, name='cross_entropy_mean')
    return loss

def train(features, labels, batch_size, model='fcn5'):
    logits = model_fcn5(features)
    loss_value = loss(logits, labels)
    lr = 0.5
    global_step = tf.Variable(0, name='global_step', trainable=False)
    optimizer = tf.train.GradientDescentOptimizer(lr).minimize(loss_value, global_step=global_step)
    return optimizer
    


