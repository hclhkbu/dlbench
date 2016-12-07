-- '*' should be replaced by network name
th *.lua
th resnet.lua -depth 50 -nGPU 1 -deviceId 2 -nThreads 2 -batchSize 16 -shareGradInput true -nEpochs 4  -dataset imagenet -data ~/data/
