#!/bin/bash
. export-local-var.sh
# 2-layers LSTM for GPU 0 gtx1080
minibatch=256 iterations=1000 device_id=0 seqlen=32 hiddensize=256 numlayer=2 ./rnn_tensorflow.sh
minibatch=256 iterations=2000 device_id=0 seqlen=32 hiddensize=256 numlayer=2 ./rnn_tensorflow.sh
minibatch=256 iterations=3000 device_id=0 seqlen=32 hiddensize=256 numlayer=2 ./rnn_tensorflow.sh
minibatch=128 iterations=2000 device_id=0 seqlen=32 hiddensize=256 numlayer=2 ./rnn_tensorflow.sh
minibatch=512 iterations=2000 device_id=0 seqlen=32 hiddensize=256 numlayer=2 ./rnn_tensorflow.sh
minibatch=256 iterations=2000 device_id=0 seqlen=64 hiddensize=256 numlayer=2 ./rnn_tensorflow.sh
minibatch=256 iterations=2000 device_id=0 seqlen=128 hiddensize=256 numlayer=2 ./rnn_tensorflow.sh
