#!/bin/bash
source ~/tf11/bin/activate
start=`date +%s.%N`
CUDA_VISIBLE_DEVICES=$deviceId python fcn5_mnist.py  --batch_size=$batch_size --epochs=$epochs --device_id=$deviceId
end=`date +%s.%N`
runtime=$( echo "$end - $start" | bc -l )
echo "finished with execute time: ${runtime}" 
