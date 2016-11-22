#!/bin/bash
#source ~/tf11/bin/activate

start=`date +%s.%N`
CUDA_VISIBLE_DEVICES=$device_id python lstm.py --batchsize $batch_size --device $device_id --max_max_epoch $epochs
end=`date +%s.%N`
runtime=$( echo "$end - $start" | bc -l )
echo "finished with execute time: ${runtime}" 
#deactivate 
