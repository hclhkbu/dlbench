#logprefix='testrun'
#host=`hostname`
#nohup python benchmark.py -config configs/$host-torch.config >& $logprefix\torch.log &
#nohup python benchmark.py -config configs/$host-tensorflow.config >& $logprefix\tensorflow.log &

python benchmark.py -config configs/synbm2cpu2.config 
python benchmark.py -config configs/synbm2cpu16.config 

#python benchmark.py -config configs/bm2cpu2.config 
#python benchmark.py -config configs/bm2cpu16.config 
