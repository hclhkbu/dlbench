#!/bin/bash
logfile="resnet_cifar10_$(date +%m%d%H%M%S).log"
echo log will be saved in "$logfile"
nohup python train_cifar10_resnet.py --gpus 0 --lr 0.1 --num-epochs 400 --batch-size 256 > "$logfile" &
tailf "$logfile"
