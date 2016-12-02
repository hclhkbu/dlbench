#rm -rf Output/*
start=`date +%s.%N`
network=alexnet
dataset="${dataset:-cifar10}"
model_file=${network}_${dataset}.cntk
if [ ${dataset} = 'cifar10' ]
then
    cp $network.cntk_template ${model_file}
    sed -i -e "s|HOME|${HOME}|g" ${model_file}
fi
cntk configFile=${model_file} deviceId=$deviceId minibatchSize=$minibatchSize maxEpochs=$maxEpochs
end=`date +%s.%N`
runtime=$( echo "$end - $start" | bc -l )
echo "GPUCount: 1" 
echo "MinibatchSize: ${minibatchSize}" 
echo "finished with execute time: ${runtime}" 
#rm -rf Output/*
