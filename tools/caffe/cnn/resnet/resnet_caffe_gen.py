#!/usr/bin/env python2

from caffe.proto import caffe_pb2
import math
import google.protobuf as pb
from copy import deepcopy

data_train_str = '''
  name: "data"
  type: "Data"
  top: "data"
  top: "label"
  include {
    phase: TRAIN
  }
  transform_param {
    mean_file: "HOME/data/caffe/cifar10/mean.binaryproto"
  }
  data_param {
    source: "HOME/data/caffe/cifar10/cifar10_train_lmdb"
    batch_size: 128 
    backend: LMDB
  }
'''

data_test_str = '''
  name: "data"
  type: "Data"
  top: "data"
  top: "label"
  include {
    phase: TEST
  }
  transform_param {
    mean_file: "HOME/data/caffe/cifar10/mean.binaryproto"
  }
  data_param {
    source: "HOME/data/caffe/cifar10/cifar10_test_lmdb"
    batch_size: 100
    backend: LMDB
  }
'''

conv_str = '''
  name: "conv_"
  type: "Convolution"
  bottom: "norm_"
  top: "conv_"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 1
    decay_mult: 1
  }
  convolution_param {
    num_output: 16
    kernel_size: 3
    pad: 1
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
'''

relu_str = '''
  name: "relu_"
  type: "ReLU"
  bottom: "conv_"
  top: "relu_"
'''

norm_str = '''
  name: "norm_"
  type: "BatchNorm"
  bottom: "relu_"
  top: "norm_"
  batch_norm_param {
      use_global_stats: 0
  }
'''
#lrn_param
#batch_norm_param
#
'''
message BatchNormParameter {
  // If false, accumulate global mean/variance values via a moving average. If
  // true, use those accumulated values instead of computing mean/variance
  // across the batch.
  optional bool use_global_stats = 1;
  // How much does the moving average decay each iteration?
  optional float moving_average_fraction = 2 [default = .999];
  // Small value to add to the variance estimate so that we don't divide by
  // zero.
  optional float eps = 3 [default = 1e-5];
}
'''


pool_str = '''
  name: "pool_"
  type: "Pooling"
  bottom: "relu_"
  top: "pool_"
  pooling_param {
    pool: AVE
    kernel_size: 2
    stride: 2
  }
'''

fc_str = '''
  name: "fc_"
  type: "InnerProduct"
  bottom: "pool_"
  top: "fc_"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 1
    decay_mult: 1
  }
  inner_product_param {
    num_output: 10
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
'''

elem_str = '''
  name: "shortcut_"
  type: "Eltwise"
  bottom: "relu_"
  bottom: "relu_"
  top: "elem_"
  eltwise_param { operation: SUM }
'''

_conv = caffe_pb2.LayerParameter()
pb.text_format.Merge(conv_str, _conv)
_norm = caffe_pb2.LayerParameter()
pb.text_format.Merge(norm_str, _norm)
_relu = caffe_pb2.LayerParameter()
pb.text_format.Merge(relu_str, _relu)
_elem = caffe_pb2.LayerParameter()
pb.text_format.Merge(elem_str, _elem)
_pool = caffe_pb2.LayerParameter()
pb.text_format.Merge(pool_str, _pool)
_fc = caffe_pb2.LayerParameter()
pb.text_format.Merge(fc_str, _fc)

layers = []

data_train = caffe_pb2.LayerParameter()
pb.text_format.Merge(data_train_str, data_train)
data_test = caffe_pb2.LayerParameter()
pb.text_format.Merge(data_test_str, data_test)

layers.extend([data_train, data_test])

layer_idx = 0
layer_str = str(layer_idx)

conv = deepcopy(_conv)
conv.name = 'conv_' + layer_str
conv.top[0] = 'conv_' + layer_str
conv.bottom[0] = 'data'
conv.convolution_param.weight_filler.std \
    = math.sqrt(2./(3*3*3))

norm = deepcopy(_norm)
norm.name = 'norm_' + layer_str
norm.top[0] = 'norm_' + layer_str
norm.bottom[0] = 'conv_' + layer_str

relu = deepcopy(_relu)
relu.name = 'relu_' + layer_str
relu.top[0] = 'relu_' + layer_str
relu.bottom[0] = 'norm_' + layer_str

layers.extend([conv, norm, relu])


for n_const in [9]:
    for output_size in [16, 32, 64]:
        for i in range(n_const):
            # 1
            layer_idx += 1
            layer_str = str(layer_idx)

            conv = deepcopy(_conv)
            conv.name = 'conv_' + layer_str
            conv.top[0] = 'conv_' + layer_str
            conv.bottom[0] = 'relu_' + str(layer_idx-1)
            conv.convolution_param.num_output = output_size
            for prev_conv_layer in reversed(layers):
                if prev_conv_layer.name.startswith('conv_'):
                    conv.convolution_param.weight_filler.std \
                        = math.sqrt(2./(prev_conv_layer.convolution_param.num_output*3*3))
                    break

            norm = deepcopy(_norm)
            norm.name = 'norm_' + layer_str
            norm.top[0] = 'norm_' + layer_str
            norm.bottom[0] = 'conv_' + layer_str

            relu = deepcopy(_relu)
            relu.name = 'relu_' + layer_str
            relu.top[0] = 'relu_' + layer_str
            relu.bottom[0] = 'norm_' + layer_str

            layers.extend([conv, norm, relu])
            #################
            # 2
            layer_idx += 1
            layer_str = str(layer_idx)

            conv = deepcopy(_conv)
            conv.name = 'conv_' + layer_str
            conv.top[0] = 'conv_' + layer_str
            conv.bottom[0] = 'relu_' + str(layer_idx-1)
            conv.convolution_param.num_output = output_size
            for prev_conv_layer in reversed(layers):
                if prev_conv_layer.name.startswith('conv_'):
                    conv.convolution_param.weight_filler.std \
                        = math.sqrt(2./(prev_conv_layer.convolution_param.num_output*3*3))
                    break

            norm = deepcopy(_norm)
            norm.name = 'norm_' + layer_str
            norm.top[0] = 'norm_' + layer_str
            norm.bottom[0] = 'conv_' + layer_str

            ##################
            # shortcut
            elem = deepcopy(_elem)
            elem.name = 'elem_' + layer_str
            elem.top[0] = 'elem_' + layer_str
            elem.bottom[0] = 'norm_' + layer_str
            elem.bottom[1] = 'relu_' + str(layer_idx-2)

            relu = deepcopy(_relu)
            relu.name = 'relu_' + layer_str
            relu.top[0] = 'relu_' + layer_str
            relu.bottom[0] = 'elem_' + layer_str

            layers.extend([conv, norm])

            # short cut with projection
            if layer_idx in [2*n_const+2, 4*n_const+2]:

                conv = deepcopy(_conv)
                conv.name = 'proj_' + str(layer_idx-1)
                conv.top[0] = 'proj_' + str(layer_idx-1)
                conv.bottom[0] = 'relu_' + str(layer_idx-2)
                conv.convolution_param.num_output = output_size
                conv.convolution_param.kernel_size[0] = 1
                conv.convolution_param.stride[0] = 2
                conv.convolution_param.pad[0] = 0

                layers.extend([conv])


            layers.extend([elem, relu])


    prev_layer_idx = layer_idx

    for layer_idx, output_size in zip([2*n_const+1, 4*n_const+1], [32, 64]):
        layer_str = str(layer_idx)
        for layer in layers:

            if layer.name == 'elem_' + str(layer_idx+1):
                layer.bottom[1] = 'proj_' + layer_str

            if layer.name == 'conv_' + str(layer_idx):
                layer.convolution_param.stride[0] = 2


    layer_idx = 1 + prev_layer_idx
    layer_str = str(layer_idx)

    pool = deepcopy(_pool)
    pool.name = 'pool_' + layer_str
    pool.bottom[0] = 'relu_' + str(layer_idx-1)
    pool.top[0] = 'pool_' + layer_str

    fc = deepcopy(_fc)
    fc.name = 'fc_' + layer_str
    fc.bottom[0] = 'pool_' + layer_str
    fc.top[0] = 'fc_' + layer_str

    layers.extend([pool, fc])

    loss_str = '''
      name: "loss"
      type: "SoftmaxWithLoss"
      bottom: "fc_"
      bottom: "label"
      top: "loss"
    '''
    _loss = caffe_pb2.LayerParameter()
    pb.text_format.Merge(loss_str, _loss)
    loss = _loss
    loss.bottom[0] = 'fc_' + layer_str

    accu_str = '''
      name: "accuracy"
      type: "Accuracy"
      bottom: "fc_"
      bottom: "label"
      top: "accuracy"
      include {
        phase: TEST
      }
    '''
    _accu = caffe_pb2.LayerParameter()
    pb.text_format.Merge(accu_str, _accu)
    accu = _accu
    accu.bottom[0] = 'fc_' + layer_str

    layers.extend([loss, accu])

    net = caffe_pb2.NetParameter()
    net.name = "resnet" # "resnet_cifar10_" + str(n_const * 6 + 2)
    net.layer.extend(layers)
    open(net.name+'.prototxt', 'w').write(str(net))
