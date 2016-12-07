#nvprof --log-file profile2_gputrace.log --print-gpu-trace cntk configFile=ResNet_50.cntk configName=resnet
#nvprof --log-file profile512x2_16.log cntk configFile=ResNet_50.cntk configName=resnet epochSize=512 maxEpochs=2
nvprof --log-file profile3_gputrace.log --print-gpu-trace cntk configFile=ResNet_50.cntk configName=resnet epochSize=48 maxEpochs=1
#cntk configFile=ResNet_50.cntk configName=resnet epochSize=64 maxEpochs=16
rm Output/Models/*
