#!/bin/bash
logfile="alexnet_cifar10_$(date +%m%d%H%M%S).log"
echo log will be saved in "$logfile"
nohup python train_cifar10_alexnet.py --gpus 2 --lr 0.001 --lr-factor 0.9 --lr-factor-epoch 300 --num-epochs 600 --batch-size 512 > "$logfile" &
tailf "$logfile"
