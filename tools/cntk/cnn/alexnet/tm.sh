rm -rf Output/*
start=`date +%s.%N`
network=alexnet
model_file=${network}.cntk
cp $network.cntk_template ${model_file}
sed -i -e "s|HOME|${HOME}|g" ${model_file}
batchSizeForCNTK=`awk "BEGIN {print ${minibatchSize}*${gpu_count} }"` 
mpirun -n ${gpu_count} cntk configFile=alexnet.cntk deviceId=auto minibatchSize=$batchSizeForCNTK maxEpochs=$maxEpochs parallelTrain=true epochSize=0
end=`date +%s.%N`
runtime=$( echo "$end - $start" | bc -l )
echo "GPUCount: ${gpu_count}"
echo "MinibatchSize: ${minibatchSize}" 
echo "finished with execute time: ${runtime}" 
