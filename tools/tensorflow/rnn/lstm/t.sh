#!/bin/bash
source ~/tf11/bin/activate

start=`date +%s.%N`
CUDA_VISIBLE_DEVICES=$deviceId python lstm.py --batchsize $batch_size --device $deviceId --max_max_epoch $epochs
end=`date +%s.%N`
runtime=$( echo "$end - $start" | bc -l )
echo "finished with execute time: ${runtime}" 
deactivate 
