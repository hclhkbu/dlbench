#!/bin/bash
start=`date +%s.%N`
mkdir multigpu-trained
python fcn5_mnist.py  --batch_size=$batch_size --epochs=$epochs --device_id=$deviceId
end=`date +%s.%N`
runtime=$( echo "$end - $start" | bc -l )
echo "finished with execute time: ${runtime}" 
rm -rf multigpu-trained
