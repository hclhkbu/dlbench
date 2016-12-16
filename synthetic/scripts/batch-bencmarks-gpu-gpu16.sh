#!/bin/bash
# The benchmarks of all toolkits 

# GPU-0 AlexNet 
#minibatch=16    iterations=8    epochs=4    device_id=0     network_name=alexnet    ./cnn-benchmarks.sh
#minibatch=32    iterations=8    epochs=4    device_id=0     network_name=alexnet    ./cnn-benchmarks.sh
#minibatch=64    iterations=8    epochs=4    device_id=0     network_name=alexnet    ./cnn-benchmarks.sh
#minibatch=128   iterations=8    epochs=4    device_id=0     network_name=alexnet    ./cnn-benchmarks.sh
#
## GPU-0 RetNet 
#minibatch=8     iterations=8    epochs=4    device_id=0     network_name=resnet     ./cnn-benchmarks.sh
#minibatch=16    iterations=8    epochs=4    device_id=0     network_name=resnet     ./cnn-benchmarks.sh
#minibatch=32    iterations=8    epochs=4    device_id=0     network_name=resnet     ./cnn-benchmarks.sh
#minibatch=64    iterations=8    epochs=4    device_id=0     network_name=resnet     ./cnn-benchmarks.sh
#

# GPU-0 Fully Connected: FFN26752 
#minibatch=256   iterations=8    epochs=4     device_id=0    network_name=ffn26752   ./fc-benchmarks.sh
#minibatch=512   iterations=8    epochs=4     device_id=0    network_name=ffn26752   ./fc-benchmarks.sh
#minibatch=1024  iterations=8    epochs=4     device_id=0    network_name=ffn26752   ./fc-benchmarks.sh
#minibatch=2048  iterations=8    epochs=4     device_id=0    network_name=ffn26752   ./fc-benchmarks.sh
#minibatch=4096  iterations=8    epochs=4     device_id=0    network_name=ffn26752   ./fc-benchmarks.sh
#
## GPU-0 Fully Connected: FFN26752 6 Hidden Layers 
minibatch=32    iterations=8    epochs=4     device_id=0    network_name=ffn26752l6   ./fc-benchmarks.sh
minibatch=64    iterations=8    epochs=4     device_id=0    network_name=ffn26752l6   ./fc-benchmarks.sh
minibatch=128   iterations=8    epochs=4     device_id=0    network_name=ffn26752l6   ./fc-benchmarks.sh
minibatch=256   iterations=8    epochs=4     device_id=0    network_name=ffn26752l6   ./fc-benchmarks.sh
minibatch=512   iterations=8    epochs=4     device_id=0    network_name=ffn26752l6   ./fc-benchmarks.sh
minibatch=1024  iterations=8    epochs=4     device_id=0    network_name=ffn26752l6   ./fc-benchmarks.sh
