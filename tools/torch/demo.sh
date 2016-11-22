#!/bin/bash
dev=0,1,2,3
echo "Test ffn5 CPU"
#python torchbm.py -numThreads 1 -log test1 -batchSize 32 -devId -1 -network fcn5 -numEpochs 5
#python torchbm.py -numThreads 32 -log test2 -batchSize 1024 -devId -1 -network fcn5 -numEpochs 5

echo "Test ffn5 GPU"
#python torchbm.py -log test3 -batchSize 32 -devId $dev -network fcn5 -numEpochs 5
python torchbm.py -log test4 -batchSize 1024 -devId $dev -network fcn5 -numEpochs 2

echo "Test alexnet GPU"
#python torchbm.py -log test5 -batchSize 16 -devId $dev -network alexnet -numEpochs 5
python torchbm.py -log test6 -batchSize 512 -devId $dev -network alexnet -numEpochs 2

echo "Test resnet GPU"
#python torchbm.py -log test7 -batchSize 16 -devId $dev -network resnet -numEpochs 5
python torchbm.py -log test8 -batchSize 512 -devId $dev -network resnet -numEpochs 2

echo "Test lstm GPU"
python torchbm.py -log testrnn -batchSize 128 -network lstm -devId 0 -numEpochs 2
