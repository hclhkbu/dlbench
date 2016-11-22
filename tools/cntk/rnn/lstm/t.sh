rm -rf Output/*
start=`date +%s.%N`
cntk configFile=rnn.cntk deviceId=$deviceId minibatch=$minibatchSize maxEpochs=$maxEpochs
end=`date +%s.%N`
runtime=$( echo "$end - $start" | bc -l )
echo "MinibatchSize: ${minibatchSize}" 
echo "finished with execute time: ${runtime}" 
