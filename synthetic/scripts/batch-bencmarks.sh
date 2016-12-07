#!/bin/bash
# The benchmarks of all toolkits 

# GPU-0 AlexNet 
#minibatch=16    iterations=8    epochs=4    device_id=0     network_name=alexnet    ./cnn-benchmarks.sh
#minibatch=32    iterations=8    epochs=4    device_id=0     network_name=alexnet    ./cnn-benchmarks.sh
#minibatch=64    iterations=8    epochs=4    device_id=0     network_name=alexnet    ./cnn-benchmarks.sh
#minibatch=128   iterations=8    epochs=4    device_id=0     network_name=alexnet    ./cnn-benchmarks.sh

# GPU-1 AlexNet 
minibatch=16    iterations=8     epochs=4    device_id=1     network_name=alexnet    ./cnn-benchmarks.sh
minibatch=32    iterations=8     epochs=4    device_id=1     network_name=alexnet    ./cnn-benchmarks.sh
minibatch=64    iterations=8     epochs=4    device_id=1     network_name=alexnet    ./cnn-benchmarks.sh
minibatch=128   iterations=8     epochs=4    device_id=1     network_name=alexnet    ./cnn-benchmarks.sh


# GPU-0 RetNet 
#minibatch=8     iterations=8    epochs=4    device_id=0     network_name=resnet     ./cnn-benchmarks.sh
#minibatch=16    iterations=8    epochs=4    device_id=0     network_name=resnet     ./cnn-benchmarks.sh
#minibatch=32    iterations=8    epochs=4    device_id=0     network_name=resnet     ./cnn-benchmarks.sh
#minibatch=64    iterations=8    epochs=4    device_id=0     network_name=resnet     ./cnn-benchmarks.sh

# GPU-1 RetNet 
#minibatch=8     iterations=8    epochs=4    device_id=1     network_name=resnet     ./cnn-benchmarks.sh
#minibatch=16    iterations=8    epochs=4    device_id=1     network_name=resnet     ./cnn-benchmarks.sh
#minibatch=32    iterations=8    epochs=4    device_id=1     network_name=resnet     ./cnn-benchmarks.sh
#minibatch=64    iterations=8    epochs=4    device_id=1     network_name=resnet     ./cnn-benchmarks.sh


# GPU-0 Fully Connected: FFN26752 
#./batch-fc.sh
#minibatch=256   iterations=8    epochs=4     device_id=0    network_name=ffn26752   ./fc-benchmarks.sh
#minibatch=512   iterations=8    epochs=4     device_id=0    network_name=ffn26752   ./fc-benchmarks.sh
#minibatch=1024  iterations=8    epochs=4     device_id=0    network_name=ffn26752   ./fc-benchmarks.sh
#minibatch=2048  iterations=8    epochs=4     device_id=0    network_name=ffn26752   ./fc-benchmarks.sh
#minibatch=4096  iterations=8    epochs=4     device_id=0    network_name=ffn26752   ./fc-benchmarks.sh
#
#
## GPU-1 Fully Connected FFN26752 
#minibatch=256   iterations=8    epochs=4     device_id=1    network_name=ffn26752   ./fc-benchmarks.sh
#minibatch=512   iterations=8    epochs=4     device_id=1    network_name=ffn26752   ./fc-benchmarks.sh
#minibatch=1024  iterations=8    epochs=4     device_id=1    network_name=ffn26752   ./fc-benchmarks.sh
#minibatch=2048  iterations=8    epochs=4     device_id=1    network_name=ffn26752   ./fc-benchmarks.sh
#minibatch=4096  iterations=8    epochs=4     device_id=1    network_name=ffn26752   ./fc-benchmarks.sh
#
## GPU-0 Fully Connected: FFN26752 6 Hidden Layers 
#minibatch=256   iterations=8    epochs=4     device_id=0    network_name=ffn26752l6   ./fc-benchmarks.sh
#minibatch=512   iterations=8    epochs=4     device_id=0    network_name=ffn26752l6   ./fc-benchmarks.sh
#minibatch=1024  iterations=8    epochs=4     device_id=0    network_name=ffn26752l6   ./fc-benchmarks.sh
#minibatch=2048  iterations=8    epochs=4     device_id=0    network_name=ffn26752l6   ./fc-benchmarks.sh
#minibatch=4096  iterations=8    epochs=4     device_id=0    network_name=ffn26752l6   ./fc-benchmarks.sh
#
#
## GPU-1 Fully Connected: FFN26752 6 Hidden Layers 
#minibatch=64    iterations=8    epochs=4     device_id=1    network_name=ffn26752l6   ./fc-benchmarks.sh
#minibatch=128   iterations=8    epochs=4     device_id=1    network_name=ffn26752l6   ./fc-benchmarks.sh
#minibatch=256   iterations=8    epochs=4     device_id=1    network_name=ffn26752l6   ./fc-benchmarks.sh
#minibatch=512   iterations=8    epochs=4     device_id=1    network_name=ffn26752l6   ./fc-benchmarks.sh
#minibatch=1024  iterations=8    epochs=4     device_id=1    network_name=ffn26752l6   ./fc-benchmarks.sh
#minibatch=2048  iterations=8    epochs=4     device_id=1    network_name=ffn26752l6   ./fc-benchmarks.sh
#minibatch=4096  iterations=8    epochs=4     device_id=1    network_name=ffn26752l6   ./fc-benchmarks.sh
#
#
## CPU Version
minibatch=16    iterations=8    epochs=4    device_id=-1     network_name=alexnet       OMP_NUM_THREADS=1 ./cnn-benchmarks.sh
minibatch=16    iterations=8    epochs=4    device_id=-1     network_name=alexnet       OMP_NUM_THREADS=2 ./cnn-benchmarks.sh
minibatch=16    iterations=8    epochs=4    device_id=-1     network_name=alexnet       OMP_NUM_THREADS=4 ./cnn-benchmarks.sh
minibatch=16    iterations=8    epochs=4    device_id=-1     network_name=alexnet       OMP_NUM_THREADS=8 ./cnn-benchmarks.sh
minibatch=16    iterations=8    epochs=4    device_id=-1     network_name=alexnet       OMP_NUM_THREADS=16 ./cnn-benchmarks.sh
minibatch=16    iterations=8    epochs=4    device_id=-1     network_name=alexnet       OMP_NUM_THREADS=32 ./cnn-benchmarks.sh

#minibatch=16    iterations=8    epochs=4    device_id=-1     network_name=resnet        OMP_NUM_THREADS=1 ./cnn-benchmarks.sh
#minibatch=16    iterations=8    epochs=4    device_id=-1     network_name=resnet        OMP_NUM_THREADS=2 ./cnn-benchmarks.sh
#minibatch=16    iterations=8    epochs=4    device_id=-1     network_name=resnet        OMP_NUM_THREADS=4 ./cnn-benchmarks.sh
#minibatch=16    iterations=8    epochs=4    device_id=-1     network_name=resnet        OMP_NUM_THREADS=8 ./cnn-benchmarks.sh
#minibatch=16    iterations=8    epochs=4    device_id=-1     network_name=resnet        OMP_NUM_THREADS=16 ./cnn-benchmarks.sh
#minibatch=16    iterations=8    epochs=4    device_id=-1     network_name=resnet        OMP_NUM_THREADS=32 ./cnn-benchmarks.sh

#minibatch=64    iterations=8    epochs=4     device_id=-1   network_name=ffn26752       OMP_NUM_THREADS=1  ./fc-benchmarks.sh
#minibatch=64    iterations=8    epochs=4     device_id=-1   network_name=ffn26752       OMP_NUM_THREADS=2  ./fc-benchmarks.sh
#minibatch=64    iterations=8    epochs=4     device_id=-1   network_name=ffn26752       OMP_NUM_THREADS=4  ./fc-benchmarks.sh
#minibatch=64    iterations=8    epochs=4     device_id=-1   network_name=ffn26752       OMP_NUM_THREADS=8  ./fc-benchmarks.sh
#
#minibatch=64    iterations=8    epochs=4     device_id=-1   network_name=ffn26752l6     OMP_NUM_THREADS=1  ./fc-benchmarks.sh
#minibatch=64    iterations=8    epochs=4     device_id=-1   network_name=ffn26752l6     OMP_NUM_THREADS=2  ./fc-benchmarks.sh
#minibatch=64    iterations=8    epochs=4     device_id=-1   network_name=ffn26752l6     OMP_NUM_THREADS=4  ./fc-benchmarks.sh
#minibatch=64    iterations=8    epochs=4     device_id=-1   network_name=ffn26752l6     OMP_NUM_THREADS=8  ./fc-benchmarks.sh
#
#
## RNN
#./batch_rnn_torch.sh
#
#./batch_rnn_tensorflow.sh
