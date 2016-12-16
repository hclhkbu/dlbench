#logprefix='testrun'
#host=`hostname`
#nohup python benchmark.py -config configs/$host-cntk.config >& $logprefix\cntk.log &
python benchmark.py -config configs/synbm2cpu4.config 
python benchmark.py -config configs/synbm2cpu32.config 
