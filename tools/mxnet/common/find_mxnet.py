# You may need to alter this file to fit your environment
import os, sys
print 'finding mxnet...'
os.environ["MXNET_CUDNN_AUTOTUNE_DEFAULT"] = "1"
sys.path[1] = os.environ['HOME'] + '/.local/lib/python2.7/site-packages/mxnet-0.9.5-py2.7.egg'
import mxnet as mx
