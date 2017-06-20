tool=mxnet

rm *debug*.log 

# Multi GPU tests
#python $tool\bm.py -log mdebugfcn5 -batchSize 1024 -network fcn5    -devId 2,3 -numEpochs 2 -epochSize 60000 -gpuCount 1 -lr 0.05 -netType fc
#python $tool\bm.py -log mdebugalex -batchSize 512  -network alexnet -devId 2,3 -numEpochs 2 -epochSize 50000 -gpuCount 1 -lr 0.01 -netType cnn
#python $tool\bm.py -log mdebugresn -batchSize 128  -network resnet  -devId 2,3 -numEpochs 2 -epochSize 50000 -gpuCount 1 -lr 0.01 -netType cnn
#python $tool\bm.py -log mdebuglstm -batchSize 1024 -network lstm    -devId 2,3 -numEpochs 2 -epochSize -1    -gpuCount 1 -lr 0.01 -netType rnn

# Single GPU test
python $tool\bm.py -log debugfcn5 -batchSize 1024 -network fcn5   -devId 2 -numEpochs 2 -epochSize 60000 -gpuCount 1 -lr 0.05 -netType fc
python $tool\bm.py -log debugalex -batchSize 512 -network alexnet -devId 2 -numEpochs 2 -epochSize 50000 -gpuCount 1 -lr 0.01 -netType cnn
python $tool\bm.py -log debugresn -batchSize 128 -network resnet  -devId 2 -numEpochs 2 -epochSize 50000 -gpuCount 1 -lr 0.01 -netType cnn
python $tool\bm.py -log debuglstm -batchSize 1024 -network lstm   -devId 2 -numEpochs 2 -epochSize -1    -gpuCount 1 -lr 0.01 -netType rnn

# CPU test
#python $tool\bm.py -log cpu_debugfcn5 -batchSize 1024 -network fcn5   -devId -1 -numEpochs 2 -epochSize 60000 -cpuCount 20 -lr 0.05 -netType fc
#python $tool\bm.py -log cpu_debugalex -batchSize 512 -network alexnet -devId -1 -numEpochs 2 -epochSize 50000 -cpuCount 20 -lr 0.01 -netType cnn
#python $tool\bm.py -log cpu_debugresn -batchSize 128 -network resnet  -devId -1 -numEpochs 2 -epochSize 50000 -cpuCount 20 -lr 0.01 -netType cnn
# rnn not supported on CPU
#python $tool\bm.py -log cpu_debuglstm -batchSize 1024 -network lstm   -devId -1 -numEpochs 2 -epochSize -1    -cpuCount 20 -lr 0.01 -netType rnn

