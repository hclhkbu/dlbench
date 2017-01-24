rm -rf Output/*
start=`date +%s.%N`
network=alexnet
dataset="${dataset:-cifar10}"
model_file=${network}_${dataset}.cntk
if [ ${dataset} = 'cifar10' ]
then
    cp $network.cntk_template ${model_file}
    sed -i -e "s|HOME|${HOME}|g" ${model_file}
fi
batchSizeForCNTK=`awk "BEGIN {print ${minibatchSize}*${gpu_count} }"` 
mpirun -n ${gpu_count} cntk configFile=$model_file deviceId=auto minibatchSize=$batchSizeForCNTK maxEpochs=$maxEpochs parallelTrain=true epochSize=0 distributedMBReading=true 
end=`date +%s.%N`
runtime=$( echo "$end - $start" | bc -l )
echo "GPUCount: ${gpu_count}"
echo "MinibatchSize: ${minibatchSize}" 
echo "finished with execute time: ${runtime}" 
rm -rf Output/*
