tool=`printf "${PWD##*/}"`

# Single GPU test
python $tool\bm.py -log sgpu_debugfcn5 -batchSize 1024 -network fcn5 -devId 0 -numEpochs 2 -epochSize 60000 -gpuCount 1 -lr 0.05 -netType fc
python $tool\bm.py -log sgpu_debugalex -batchSize 1024 -network alexnet -devId 0 -numEpochs 2 -epochSize 50000 -gpuCount 1 -lr 0.05 -netType cnn
python $tool\bm.py -log sgpu_debugresn -batchSize 128 -network resnet -devId 0 -numEpochs 2 -epochSize 50000 -gpuCount 1 -lr 0.01 -netType cnn
python $tool\bm.py -log sgpu_debuglstm -batchSize 1024 -network lstm -devId 0 -numEpochs 2 -epochSize -1 -gpuCount 1 -lr 0.01 -netType rnn

# Multi GPU tests
python $tool\bm.py -log mgpu_debugfcn5 -batchSize 1024 -network fcn5 -devId 0,1 -numEpochs 2 -epochSize 60000 -gpuCount 2 -lr 0.05 -netType fc
python $tool\bm.py -log mgpu_debugalex -batchSize 1024 -network alexnet -devId 0,1 -numEpochs 2 -epochSize 50000 -gpuCount 2 -lr 0.05 -netType cnn
python $tool\bm.py -log mgpu_debugresn -batchSize 128 -network resnet -devId 0,1 -numEpochs 2 -epochSize 50000 -gpuCount 2 -lr 0.01 -netType cnn
python $tool\bm.py -log mgpu_debuglstm -batchSize 1024 -network lstm -devId 0,1 -numEpochs 2 -epochSize -1 -gpuCount 1 -lr 0.01 -netType rnn

# CPU test
python $tool\bm.py -log cpu_debugfcn5 -batchSize 1024 -network fcn5   -devId -1 -numEpochs 2 -epochSize 60000 -cpuCount 20 -lr 0.05 -netType fc
python $tool\bm.py -log cpu_debugalex -batchSize 512 -network alexnet -devId -1 -numEpochs 2 -epochSize 50000 -cpuCount 20 -lr 0.01 -netType cnn
python $tool\bm.py -log cpu_debugresn -batchSize 128 -network resnet  -devId -1 -numEpochs 2 -epochSize 50000 -cpuCount 20 -lr 0.01 -netType cnn
python $tool\bm.py -log cpu_debuglstm -batchSize 1024 -network lstm   -devId -1 -numEpochs 2 -epochSize -1    -cpuCount 20 -lr 0.01 -netType rnn
