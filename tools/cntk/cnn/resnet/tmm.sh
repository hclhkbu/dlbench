start=`date +%s.%N`
network=resnet
model_file=${network}.cntk
cp $network.cntk_template ${model_file}
sed -i -e "s|HOME|${HOME}|g" ${model_file}
batchSizeForCNTK=`awk "BEGIN {print ${minibatchSize}*${gpu_count} }"` 
mpirun -np ${gpu_count} -machinefile cluster${gpu_count} cntk configFile=resnet.cntk deviceId=auto minibatchSize=$batchSizeForCNTK maxEpochs=$maxEpochs parallelTrain=true distributedMBReading=false command=Train
cntk configFile=resnet.cntk command=Test
end=`date +%s.%N`
runtime=$( echo "$end - $start" | bc -l )
echo "MinibatchSize: ${minibatchSize}" 
echo "finished with execute time: ${runtime}" 
rm -rf Output/*
