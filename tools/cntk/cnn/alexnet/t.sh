rm -rf Output/*
start=`date +%s.%N`
network=alexnet
model_file=${network}.cntk
cp $network.cntk_template ${model_file}
sed -i -e "s|HOME|${HOME}|g" ${model_file}
cntk configFile=alexnet.cntk deviceId=$deviceId minibatchSize=$minibatchSize maxEpochs=$maxEpochs
end=`date +%s.%N`
runtime=$( echo "$end - $start" | bc -l )
echo "GPUCount: 1" 
echo "MinibatchSize: ${minibatchSize}" 
echo "finished with execute time: ${runtime}" 
