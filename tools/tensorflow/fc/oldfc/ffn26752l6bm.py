# A feed-forward DNN with 5 hidden layers using sigmoid activations.
import os
import time
import tensorflow as tf
#import ffn
import argparse

from ffn26752l6 import *

device_str = ''

def set_parameters(epochs, minibatch, iterations, device_id):
    """
    iterations means the number of iterations in each epoch
    """
    global device_str
    if int(device_id) >= 0:
        device_str = '/gpu:%d'%int(device_id)
    else:
        # cpus
        device_str = '/cpu:0'
    global numMinibatches
    numMinibatches = iterations*epochs
    #numMinibatches = (138493+minibatch-1)/minibatch * epochs


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--epochs", help="the number of epochs", type=int, default=4)
    parser.add_argument("-b", "--minibatch", help="minibatch size", type=int, default=128)
    parser.add_argument("-i", "--iterations", help="iterations", type=int, default=2)
    parser.add_argument("-d", "--deviceid", help="specified device id", type=int, default=0)
    args = parser.parse_args()
    
    epochs = args.epochs 
    minibatch = args.minibatch 
    iterations = args.iterations 
    device_id = args.deviceid 
    minibatchSize = args.minibatch

    set_parameters(epochs, minibatch, iterations, device_id)
    
    program_start_time = time.time()
    
    # Create the model
    if (FLAGS.noInputFeed):
      features, labels = getFakeMinibatch(minibatchSize)
    else:
      features = tf.placeholder("float", [None, featureDim])
      labels = tf.placeholder("float", [None, labelDim])
    config = tf.ConfigProto(allow_soft_placement=True)
    if device_str.find('cpu') >= 0: # cpu version
        num_threads = os.getenv('OMP_NUM_THREADS', 1)
        print 'num_threads: ', num_threads
        config = tf.ConfigProto(allow_soft_placement=True, intra_op_parallelism_threads=int(num_threads))


    with tf.device(device_str):
        crossEntropy, accuracy = getLossAndAccuracyForSubBatch(features, labels)
        trainStep = tf.train.GradientDescentOptimizer(0.01).minimize(crossEntropy)
        
        # Train
        #sess = tf.Session(config=tf.ConfigProto(log_device_placement=FLAGS.logDevicePlacement, allow_soft_placement=True))
        sess = tf.Session(config=config)
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
        #print('Program finished, Total seconds: %s' % (program_end_time - program_start_time))
