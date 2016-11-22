#minibatch=2048  iterations=8    maxEpochs=3  lstm=32   device_id=0    network_name=rnn   ./benchmarks.sh
#minibatch=4096  iterations=8    maxEpochs=3  lstm=32   device_id=0    network_name=rnn   ./benchmarks.sh
#minibatch=8192  iterations=8    maxEpochs=3  lstm=32   device_id=0    network_name=rnn   ./benchmarks.sh
#minibatch=16384  iterations=8   maxEpochs=3  lstm=32   device_id=0    network_name=rnn   ./benchmarks.sh
#
#minibatch=4096  iterations=8    maxEpochs=3  lstm=64   device_id=0    network_name=rnn   ./benchmarks.sh
#minibatch=8192  iterations=8    maxEpochs=3  lstm=64   device_id=0    network_name=rnn   ./benchmarks.sh
#minibatch=16384  iterations=8   maxEpochs=3  lstm=64   device_id=0    network_name=rnn   ./benchmarks.sh
minibatch=32768  iterations=8   maxEpochs=3  lstm=64   device_id=0    network_name=rnn    ./benchmarks.sh

#minibatch=2048  iterations=8    maxEpochs=3  lstm=32  device_id=1    network_name=rnn   ./benchmarks.sh
#minibatch=4096  iterations=8    maxEpochs=3  lstm=32  device_id=1    network_name=rnn   ./benchmarks.sh
#minibatch=8192  iterations=8    maxEpochs=3  lstm=32  device_id=1    network_name=rnn   ./benchmarks.sh
#minibatch=16384  iterations=8   maxEpochs=3  lstm=32  device_id=1   network_name=rnn   ./benchmarks.sh
#
#minibatch=4096  iterations=8    maxEpochs=3  lstm=64   device_id=1    network_name=rnn   ./benchmarks.sh
#minibatch=8192  iterations=8    maxEpochs=3  lstm=64   device_id=1    network_name=rnn   ./benchmarks.sh
#minibatch=16384  iterations=8   maxEpochs=3  lstm=64   device_id=1    network_name=rnn   ./benchmarks.sh
#minibatch=32768  iterations=8   maxEpochs=3  lstm=64   device_id=1    network_name=rnn    ./benchmarks.sh

#cpu 
#minibatch=2048  iterations=8    maxEpochs=3  lstm=32  device_id=-1    network_name=rnn  OMP_NUM_THREADS=1 ./benchmarks.sh
#minibatch=2048  iterations=8    maxEpochs=3  lstm=32  device_id=-1    network_name=rnn  OMP_NUM_THREADS=2 ./benchmarks.sh
#minibatch=2048  iterations=8    maxEpochs=3  lstm=32  device_id=-1    network_name=rnn  OMP_NUM_THREADS=4 ./benchmarks.sh
#minibatch=2048  iterations=8    maxEpochs=3  lstm=32  device_id=-1    network_name=rnn  OMP_NUM_THREADS=8 ./benchmarks.sh
#
#minibatch=4096  iterations=8    maxEpochs=3  lstm=64  device_id=-1    network_name=rnn  OMP_NUM_THREADS=1 ./benchmarks.sh
#minibatch=4096  iterations=8    maxEpochs=3  lstm=64  device_id=-1    network_name=rnn  OMP_NUM_THREADS=2 ./benchmarks.sh
#minibatch=4096  iterations=8    maxEpochs=3  lstm=64  device_id=-1    network_name=rnn  OMP_NUM_THREADS=4 ./benchmarks.sh
#minibatch=4096  iterations=8    maxEpochs=3  lstm=64  device_id=-1    network_name=rnn  OMP_NUM_THREADS=8 ./benchmarks.sh
