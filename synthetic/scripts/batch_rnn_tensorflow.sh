#!/bin/bash
. export-local-var.sh
# 2-layers LSTM for GPU 0 gtx1080
#minibatch=128 iterations=50 device_id=0 seqlen=32 hiddensize=256 numlayer=2 ./rnn_tensorflow.sh
#minibatch=256 iterations=50 device_id=0 seqlen=32 hiddensize=256 numlayer=2 ./rnn_tensorflow.sh
#minibatch=512 iterations=50 device_id=0 seqlen=32 hiddensize=256 numlayer=2 ./rnn_tensorflow.sh
#minibatch=128 iterations=50 device_id=0 seqlen=64 hiddensize=256 numlayer=2 ./rnn_tensorflow.sh
#minibatch=256 iterations=50 device_id=0 seqlen=64 hiddensize=256 numlayer=2 ./rnn_tensorflow.sh
#minibatch=512 iterations=50 device_id=0 seqlen=64 hiddensize=256 numlayer=2 ./rnn_tensorflow.sh

# 2-layers LSTM for GPU 1 k80 
minibatch=128 iterations=50 device_id=1 seqlen=32 hiddensize=256 numlayer=2 ./rnn_tensorflow.sh
minibatch=256 iterations=50 device_id=1 seqlen=32 hiddensize=256 numlayer=2 ./rnn_tensorflow.sh
minibatch=512 iterations=50 device_id=1 seqlen=32 hiddensize=256 numlayer=2 ./rnn_tensorflow.sh
minibatch=128 iterations=50 device_id=1 seqlen=64 hiddensize=256 numlayer=2 ./rnn_tensorflow.sh
minibatch=256 iterations=50 device_id=1 seqlen=64 hiddensize=256 numlayer=2 ./rnn_tensorflow.sh
minibatch=512 iterations=50 device_id=1 seqlen=64 hiddensize=256 numlayer=2 ./rnn_tensorflow.sh

# 4-layers LSTM for GPU 0 gtx1080
#minibatch=128 iterations=50 device_id=0 seqlen=32 hiddensize=256 numlayer=4 ./rnn_tensorflow.sh
#minibatch=256 iterations=50 device_id=0 seqlen=32 hiddensize=256 numlayer=4 ./rnn_tensorflow.sh
#minibatch=512 iterations=50 device_id=0 seqlen=32 hiddensize=256 numlayer=4 ./rnn_tensorflow.sh
#minibatch=128 iterations=50 device_id=0 seqlen=64 hiddensize=256 numlayer=4 ./rnn_tensorflow.sh
#minibatch=256 iterations=50 device_id=0 seqlen=64 hiddensize=256 numlayer=4 ./rnn_tensorflow.sh
#minibatch=512 iterations=50 device_id=0 seqlen=64 hiddensize=256 numlayer=4 ./rnn_tensorflow.sh

# 4-layers LSTM for GPU 1 gtx1080
minibatch=128 iterations=50 device_id=1 seqlen=32 hiddensize=256 numlayer=4 ./rnn_tensorflow.sh
minibatch=256 iterations=50 device_id=1 seqlen=32 hiddensize=256 numlayer=4 ./rnn_tensorflow.sh
minibatch=512 iterations=50 device_id=1 seqlen=32 hiddensize=256 numlayer=4 ./rnn_tensorflow.sh
minibatch=128 iterations=50 device_id=1 seqlen=64 hiddensize=256 numlayer=4 ./rnn_tensorflow.sh
minibatch=256 iterations=50 device_id=1 seqlen=64 hiddensize=256 numlayer=4 ./rnn_tensorflow.sh
minibatch=512 iterations=50 device_id=1 seqlen=64 hiddensize=256 numlayer=4 ./rnn_tensorflow.sh

