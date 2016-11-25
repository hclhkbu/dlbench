logprefix='testrun'
host=`hostname`
nohup python benchmark.py -config configs/$host-torch.config >& $logprefix\torch.log &
nohup python benchmark.py -config configs/$host-tensorflow.config >& $logprefix\tensorflow.log &
