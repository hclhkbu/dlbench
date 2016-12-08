#!/bin/bash
source ~/.bashrc
# 2-layers LSTM for GPU 0 gtx1080
#minibatch=128 iterations=50 device_id=0 seqlen=32 hiddensize={256,256} ./rnn_torch.sh
#minibatch=256 iterations=50 device_id=0 seqlen=32 hiddensize={256,256} ./rnn_torch.sh
#minibatch=512 iterations=50 device_id=0 seqlen=32 hiddensize={256,256} ./rnn_torch.sh
#minibatch=128 iterations=50 device_id=0 seqlen=64 hiddensize={256,256} ./rnn_torch.sh
#minibatch=256 iterations=50 device_id=0 seqlen=64 hiddensize={256,256} ./rnn_torch.sh
#minibatch=512 iterations=50 device_id=0 seqlen=64 hiddensize={256,256} ./rnn_torch.sh

# 2-layers LSTM for GPU 1 gtx980
minibatch=128 iterations=50 device_id=1 seqlen=32 hiddensize={256,256} ./rnn_torch.sh
minibatch=256 iterations=50 device_id=1 seqlen=32 hiddensize={256,256} ./rnn_torch.sh
minibatch=512 iterations=50 device_id=1 seqlen=32 hiddensize={256,256} ./rnn_torch.sh
minibatch=128 iterations=50 device_id=1 seqlen=64 hiddensize={256,256} ./rnn_torch.sh
minibatch=256 iterations=50 device_id=1 seqlen=64 hiddensize={256,256} ./rnn_torch.sh
minibatch=512 iterations=50 device_id=1 seqlen=64 hiddensize={256,256} ./rnn_torch.sh

# 4-layers LSTM for GPU 0 gtx1080
#minibatch=128 iterations=50 device_id=0 seqlen=32 hiddensize={256,256,256,256} ./rnn_torch.sh
#minibatch=256 iterations=50 device_id=0 seqlen=32 hiddensize={256,256,256,256} ./rnn_torch.sh
#minibatch=512 iterations=50 device_id=0 seqlen=32 hiddensize={256,256,256,256} ./rnn_torch.sh
#minibatch=128 iterations=50 device_id=0 seqlen=64 hiddensize={256,256,256,256} ./rnn_torch.sh
#minibatch=256 iterations=50 device_id=0 seqlen=64 hiddensize={256,256,256,256} ./rnn_torch.sh
#minibatch=512 iterations=50 device_id=0 seqlen=64 hiddensize={256,256,256,256} ./rnn_torch.sh

# 4-layers LSTM for GPU 1 gtx1080
minibatch=128 iterations=50 device_id=1 seqlen=32 hiddensize={256,256,256,256} ./rnn_torch.sh
minibatch=256 iterations=50 device_id=1 seqlen=32 hiddensize={256,256,256,256} ./rnn_torch.sh
minibatch=512 iterations=50 device_id=1 seqlen=32 hiddensize={256,256,256,256} ./rnn_torch.sh
minibatch=128 iterations=50 device_id=1 seqlen=64 hiddensize={256,256,256,256} ./rnn_torch.sh
minibatch=256 iterations=50 device_id=1 seqlen=64 hiddensize={256,256,256,256} ./rnn_torch.sh
minibatch=512 iterations=50 device_id=1 seqlen=64 hiddensize={256,256,256,256} ./rnn_torch.sh

