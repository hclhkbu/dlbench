tool=caffe

# Single GPU test
#python $tool\bm.py -log debug -batchSize 1024 -network fcn5 -devId 0 -numEpochs 2 -epochSize 60000 -gpuCount 1 -lr 0.05 -netType fc
#python $tool\bm.py -log debug -batchSize 1024 -network alexnet -devId 0 -numEpochs 2 -epochSize 50000 -gpuCount 1 -lr 0.05 -netType cnn
python $tool\bm.py -log debug -batchSize 128 -network resnet -devId 0 -numEpochs 2 -epochSize 50000 -gpuCount 1 -lr 0.01 -netType cnn
#
## Multi GPU tests
#python $tool\bm.py -log debug -batchSize 1024 -network fcn5 -devId 0,1 -numEpochs 2 -epochSize 60000 -gpuCount 2 -lr 0.05 -netType fc
#python $tool\bm.py -log debug -batchSize 1024 -network alexnet -devId 0,1 -numEpochs 2 -epochSize 50000 -gpuCount 2 -lr 0.05 -netType cnn
#python $tool\bm.py -log debug -batchSize 128 -network resnet -devId 0,1 -numEpochs 2 -epochSize 50000 -gpuCount 2 -lr 0.01 -netType cnn
