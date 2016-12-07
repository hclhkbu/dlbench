# A feed-forward DNN with 5 hidden layers using sigmoid activations.

import time
import tensorflow as tf
import ffn

from ffn import *

if __name__ == '__main__':
    minibatchSize = 1024 
    
    program_start_time = time.time()
    
    # Create the model
    if (FLAGS.noInputFeed):
      features, labels = getFakeMinibatch(minibatchSize)
    else:
      features = tf.placeholder("float", [None, featureDim])
      labels = tf.placeholder("float", [None, labelDim])
    
    crossEntropy, accuracy = getLossAndAccuracyForSubBatch(features, labels)
    trainStep = tf.train.GradientDescentOptimizer(0.01).minimize(crossEntropy)
    
    # Train
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=FLAGS.logDevicePlacement))
    init = tf.initialize_all_variables()
    sess.run(init)
    
    perMinibatchTime = []
    for i in range(numMinibatches):
      if (FLAGS.noInputFeed == False):
        minibatchFeatures, minibatchLabels = getFakeMinibatch(minibatchSize)
    
      startTime = time.time()
      if (FLAGS.noInputFeed):
        sess.run([trainStep, accuracy])
      else:
        sess.run([trainStep, accuracy], feed_dict={features: minibatchFeatures, labels: minibatchLabels})
    
      currMinibatchDuration = time.time() - startTime
      perMinibatchTime.append(currMinibatchDuration)
    
    printTrainingStats(1, minibatchSize, perMinibatchTime)
    
    program_end_time = time.time()
    print('Program finished, Total seconds: %s' % (program_end_time - program_start_time))
