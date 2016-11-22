#!/bin/bash
dev=0,1,2,3
tool="mxnet"
echo "Note that batch size is for each GPU core"
#echo "Test alexnet GPU"
##python mxnetbm.py -log test5$tool -batchSize 16 -devId $dev -network alexnet -numEpochs 5
#python mxnetbm.py -log test6$tool -batchSize 512 -devId $dev -network alexnet -numEpochs 2
#cat cnn/test6$tool\.log
#
#echo "Test resnet GPU"
##python mxnetbm.py -log test7$tool -batchSize 16 -devId $dev -network resnet -numEpochs 5
#python mxnetbm.py -log test8$tool -batchSize 512 -devId $dev -network resnet -numEpochs 2
#cat cnn/test8$tool\.log
#
#echo "Test ffn5 CPU"
##python mxnetbm.py -numThreads 1 -log test1$tool -batchSize 32 -devId -1 -network fcn5 -numEpochs 5
##python mxnetbm.py -numThreads 32 -log test2$tool -batchSize 1024 -devId -1 -network fcn5 -numEpochs 2
#
#echo "Test ffn5 GPU"
##python mxnetbm.py -log test3$tool -batchSize 32 -devId $dev -network fcn5 -numEpochs 5
#python mxnetbm.py -log test4$tool -batchSize 1024 -devId $dev -network fcn5 -numEpochs 2
#cat fc/test4$tool\.log
#
#echo "Test lstm32 GPU"
#python mxnetbm.py -log rnntest -batchSize 64 -network lstm32 -devId 0,1,2,3 -numEpochs 2
#cat rnn/rnntest.log

echo "Multiple machines all GPU:"
echo "Resnet:"
python mxnetbm.py -log MMtestlogresnet -batchSize 128 -network resnet -devId 0,1,2,3 -numEpochs 4 -hostFile `pwd`/hosts
sleep 5
echo "Alexnet:"
python mxnetbm.py -log MMtestlogalexnet -batchSize 512 -network alexnet -devId 0,1,2,3 -numEpochs 4 -hostFile `pwd`/hosts
sleep 5
echo "FCN5:"
python mxnetbm.py -log MMtestlogfcn5 -batchSize 512 -network fcn5 -devId 0,1,2,3 -numEpochs 4 -hostFile `pwd`/hosts
