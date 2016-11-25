logprefix='testrun'
host=`hostname`
nohup python benchmark.py -config configs/$host-caffe.config >& $logprefix\caffe.log &
nohup python benchmark.py -config configs/$host-mxnet.config >& $logprefix\mxnet.log &
