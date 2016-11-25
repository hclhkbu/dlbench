logprefix='testrun'
host=`hostname`
nohup python benchmark.py -config configs/$host-cntk.config >& $logprefix\cntk.log &
