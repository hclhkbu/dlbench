mpirun -np 12 -machinefile cluster12 cntk configFile=alexnet_imagenet2.cntk deviceId=auto minibatchSize=256 maxEpochs=20 parallelTrain=true command=Train
cntk configFile=alexnet_imagenet2.cntk deviceId=auto minibatchSize=256 command=Test
