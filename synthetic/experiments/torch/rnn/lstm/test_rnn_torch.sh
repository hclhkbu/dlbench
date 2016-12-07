#!/bin/bash
# GPU 0
minibatch=256 iterations=100 device_id=0 seqlen=32 hiddensize={256,256} ./rnn_torch.sh
# GPU 1
minibatch=256 iterations=100 device_id=1 seqlen=32 hiddensize={256,256} ./rnn_torch.sh
# CPU
minibatch=256 iterations=100 device_id=-1 seqlen=32 hiddensize={256,256} ./rnn_torch.sh
