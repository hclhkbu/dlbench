tool=mxnet

# Single GPU test
python $tool\bm.py -debug True -log debug -batchSize 1024 -network fcn5 -devId 2 -numEpochs 2 -epochSize 60000 -gpuCount 1 -lr 0.05 -netType fc
#python $tool\bm.py -debug True -log debug -batchSize 1024 -network alexnet -devId 2 -numEpochs 2 -epochSize 50000 -gpuCount 1 -lr 0.05 -netType cnn
#python $tool\bm.py -debug True -log debug -batchSize 128 -network resnet -devId 2 -numEpochs 2 -epochSize 50000 -gpuCount 1 -lr 0.01 -netType cnn
#python $tool\bm.py -debug True -log debug -batchSize 1024 -network lstm -devId 2 -numEpochs 2 -epochSize -1 -gpuCount 1 -lr 1 -netType rnn
#
## Multi GPU tests
#python $tool\bm.py -debug True -log debug -batchSize 1024 -network fcn5 -devId 2,3 -numEpochs 2 -epochSize 60000 -gpuCount 2 -lr 0.05 -netType fc
#python $tool\bm.py -debug True -log debug -batchSize 1024 -network alexnet -devId 2,3 -numEpochs 2 -epochSize 50000 -gpuCount 2 -lr 0.05 -netType cnn
#python $tool\bm.py -debug True -log debug -batchSize 128 -network resnet -devId 2,3 -numEpochs 2 -epochSize 50000 -gpuCount 2 -lr 0.01 -netType cnn
