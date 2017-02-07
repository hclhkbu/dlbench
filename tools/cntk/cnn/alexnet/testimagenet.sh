mpirun -np 16 -machinefile cluster16 cntk configFile=alexnet_imagenet2.cntk deviceId=auto minibatchSize=256 maxEpochs=40 parallelTrain=true command=Train
#cntk configFile=alexnet_imagenet2.cntk deviceId=auto minibatchSize=256 command=Test
