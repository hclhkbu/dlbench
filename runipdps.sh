logprefix='sgbenchmark6'
host=`hostname`
nohup python benchmark.py -config configs/gpu1080.config >& $logprefix\1080.log &
nohup python benchmark.py -config configs/gpu980.config >& $logprefix\980.log &
