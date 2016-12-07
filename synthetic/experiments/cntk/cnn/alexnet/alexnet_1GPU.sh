start=`date +%s.%N`
cntk configFile=AlexNet.cntk configName=AlexNet >1GPU.log 2>&1
end=`date +%s.%N`
runtime=$( echo "$end - $start" | bc -l )
echo "finished with execute time: ${runtime}" >>1GPU.log

