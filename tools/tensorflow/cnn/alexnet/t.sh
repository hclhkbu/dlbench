#!/bin/bash
mkdir -p trained_models
mkdir -p train_eval 
start=`date +%s.%N`
# Please be noted that we now update TF to the version of 1.2, and the last two parameters can work. Thanks to tfboyd
CUDA_VISIBLE_DEVICES=$deviceId python alexnet_cifar10.py --batch_size=$batch_size --epochs=$epochs --device_id=$deviceId --xla=True --use_datasets=True 
end=`date +%s.%N`
runtime=$( echo "$end - $start" | bc -l )
echo "finished with execute time: ${runtime}" 
python cifar10_eval.py 
rm trained_models/*
rm train_eval/* 
