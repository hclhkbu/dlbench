#!/bin/bash
logfile="ffn6_lr05_cifar10_$(date +%m%d%H%M%S).log"
echo log will be saved in "$logfile"
nohup python train_cifar10.py --batch-size 512 --gpus 1 --data-dir ~/Data/mxnet/cifar10_32/ --num-epochs 100 --lr 0.05 > "$logfile" &
tailf "$logfile"
