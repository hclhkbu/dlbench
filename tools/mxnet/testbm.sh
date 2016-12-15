tool=mxnet

# Single GPU test
<<<<<<< HEAD
python $tool\bm.py -log debug -batchSize 1024 -network fcn5 -devId 0 -numEpochs 2 -epochSize 60000 -gpuCount 1 -lr 0.05 -netType fc
python $tool\bm.py -log debug -batchSize 1024 -network alexnet -devId 0 -numEpochs 2 -epochSize 50000 -gpuCount 1 -lr 0.05 -netType cnn
python $tool\bm.py -log debug -batchSize 128 -network resnet -devId 0 -numEpochs 2 -epochSize 50000 -gpuCount 1 -lr 0.01 -netType cnn
python $tool\bm.py -log debug -batchSize 1024 -network lstm -devId 0 -numEpochs 2 -epochSize -1 -gpuCount 1 -lr 1 -netType rnn
=======
#python $tool\bm.py -log debug -batchSize 1024 -network fcn5 -devId 2 -numEpochs 2 -epochSize 60000 -gpuCount 1 -lr 0.05 -netType fc
#python $tool\bm.py -log debug -batchSize 1024 -network alexnet -devId 2 -numEpochs 2 -epochSize 50000 -gpuCount 1 -lr 0.05 -netType cnn
#python $tool\bm.py -log debug -batchSize 128 -network resnet -devId 2 -numEpochs 2 -epochSize 50000 -gpuCount 1 -lr 0.01 -netType cnn
#python $tool\bm.py -log debug -batchSize 1024 -network lstm -devId 2 -numEpochs 2 -epochSize -1 -gpuCount 1 -lr 1 -netType rnn
>>>>>>> cd284c63b485f1cb44c7b5012a948f9e3e3d0516

# Multi GPU tests
python $tool\bm.py -log debug -batchSize 1024 -network fcn5 -devId 2,3 -numEpochs 2 -epochSize 60000 -gpuCount 2 -lr 0.05 -netType fc
python $tool\bm.py -log debug -batchSize 1024 -network alexnet -devId 2,3 -numEpochs 2 -epochSize 50000 -gpuCount 2 -lr 0.05 -netType cnn
python $tool\bm.py -log debug -batchSize 128 -network resnet -devId 2,3 -numEpochs 2 -epochSize 50000 -gpuCount 2 -lr 0.01 -netType cnn
