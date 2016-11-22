#!/bin/bash
#source ~/tf11/bin/activate
start=`date +%s.%N`
CUDA_VISIBLE_DEVICES=$deviceId python alexnet_cifar10_multi_gpu.py --batch_size=$batch_size --epochs=$epochs --device_ids=$deviceId --num_gpus=${gpu_count}
end=`date +%s.%N`
runtime=$( echo "$end - $start" | bc -l )
echo "finished with execute time: ${runtime}" 
python cifar10_eval.py 
#deactivate 
