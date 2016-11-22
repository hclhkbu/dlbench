#!/bin/bash
'''
Tested accuracy over 94%, with ffn sigmoid activation:
	data	784
	h1	2048
	h2	4096
	h3	1024
	output	10
'''
logfile="ffn_mnist_$(date +%m%d%H%M%S).log"
echo log will be saved in "$logfile"
nohup python train_mnist.py --gpus 1 --lr 0.05 --num-epochs 5 --batch-size 1024 > "$logfile" &
tailf "$logfile"
