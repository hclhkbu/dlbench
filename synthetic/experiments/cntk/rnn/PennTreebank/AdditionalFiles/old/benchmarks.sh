rm ../Output/Models/rnn.dnn*
minibatch="${minibatch:-2048}"
device_id="${device_id:-0}"
tmplog=${minibatch}-${device_id}.log
benchmark_logfile=gpu${device_id}.bm

running_time=`date`

echo -e 'GPU:'${device_id}'\nNUM_THREADS (for CPU): '${OMP_NUM_THREADS}'\nNetwork: RNN\nEpochs: ''\nMinibatch: '${minibatch}'\nBenchmark Time: '${running_time}'\n_________________\n'>> ${benchmark_logfile}

cntk configFile=../Config/rnn.cntk deviceId=${device_id} minibatch=${minibatch} >& ${tmplog}
#cntk configFile=${network_name}.cntk configName=../Config/rnn.cntk deviceId=${device_id} minibatchSize=${minibatch} epochSize=${epoch_size} maxEpochs=${epochs} >& ${tmplog}
#tmp_total_time=`cat ${tmplog} | grep "Epoch\[ 1 of 2\]-Minibatch\[ " |awk '{print $14}' |cut -d's' -f 1`
tmp_total_time=`cat ${tmplog} | grep "Epoch\[ 1 of 2\]-Minibatch\[   1- 100]" |awk '{print $14}' |cut -d's' -f 1`
echo ${tmp_total_time}
linecount=100
time_in_second=`awk "BEGIN {print ${tmp_total_time}/${linecount}}"`
echo ${time_in_second}
echo -e ${tool}'\t\t\t'${time_in_second}>>${benchmark_logfile}

