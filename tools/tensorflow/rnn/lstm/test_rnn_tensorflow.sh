#!/bin/bash
. export-local-var.sh
# GPU 0
minibatch=256 iterations=100 device_id=0 seqlen=32 hiddensize=256 numlayer=2 ./rnn_tensorflow.sh
# GPU 1
minibatch=256 iterations=100 device_id=1 seqlen=32 hiddensize=256 numlayer=2 ./rnn_tensorflow.sh
# CPU
minibatch=256 iterations=100 device_id=-1 seqlen=32 hiddensize=256 numlayer=2 ./rnn_tensorflow.sh
