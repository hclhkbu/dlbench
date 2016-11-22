rm ../Output/Models/rnn.dnn*
minibatch="${minibatch:-4096}"
device_id="${device_id:-0}"
tmplog=${minibatch}-${device_id}.log
epoch_size=`awk "BEGIN {print ${minibatch}*10}"`
lstm="${lstm:-32}"
benchmark_logfile=lstm${lstm}gpu${device_id}.bm
tool=cntk

running_time=`date`

echo -e 'GPU:'${device_id}'\nNUM_THREADS (for CPU): '${OMP_NUM_THREADS}'\nNetwork: RNN\nEpochs: ''\nMinibatch: '${minibatch}'\nBenchmark Time: '${running_time}'\n_________________\n'>> ${benchmark_logfile}

cntk configFile=../Config/rnn.cntk deviceId=${device_id} minibatch=${minibatch} epochSize=${epoch_size} maxEpochs=3 makeMode=false trainFile="ptb.train.${lstm}.ctf">& ${tmplog}
#tmp_total_time=`cat ${tmplog} | grep "Epoch\[ 1 of 2\]-Minibatch\[   1- 100]" |awk '{print $14}' |cut -d's' -f 1`
#tmp_total_time=`cat ${tmplog} | grep "Epoch\[ 2 of 3\]-Minibatch\[  19-  19" |awk '{print $15}' |cut -d's' -f 1`
tmp_total_time=`cat ${tmplog} | grep "Epoch\[ 2 of 3\]-Minibatch\[   8-   8" |awk '{print $15}' |cut -d's' -f 1`
linecount=100
#time_in_second=`awk "BEGIN {print ${tmp_total_time}/${linecount}}"`
time_in_second=${tmp_total_time}
echo ${time_in_second}
echo -e ${tool}'\t\t\t'${time_in_second}>>${benchmark_logfile}

