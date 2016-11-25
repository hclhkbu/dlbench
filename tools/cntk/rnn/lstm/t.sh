start=`date +%s.%N`
network=lstm
model_file=${network}.cntk
cp $network.cntk_template ${model_file}
sed -i -e "s|HOME|${HOME}|g" ${model_file}
cntk configFile=${network}.cntk deviceId=$deviceId minibatch=$minibatchSize maxEpochs=$maxEpochs
end=`date +%s.%N`
runtime=$( echo "$end - $start" | bc -l )
echo "GPUCount: 1" 
echo "MinibatchSize: ${minibatchSize}" 
echo "finished with execute time: ${runtime}" 
rm -rf Output/*
