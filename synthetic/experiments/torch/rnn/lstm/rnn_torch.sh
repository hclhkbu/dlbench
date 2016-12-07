
network_type="${network_type:-rnn}"
network_name="${network_name:-lstm}"
tool=torch

minibatch="${minibatch:-64}"
iterations="${iterations:-2000}"
device_id="${device_id:-1}"
seqlen="${seqlen:-32}"
hiddensize="${hiddensize:-{256,256\}}"
thdevice_id=`awk "BEGIN {print ${device_id}+1}"`
if [ ${device_id} = -1 ]
then
    default="--lstm --startlr 1 --cutoff 5 --maxepoch 1"
else
    default="--cuda --lstm --startlr 1 --cutoff 5 --maxepoch 1 --device ${thdevice_id}"
fi

benchmark_logfile=${network_type}-${network_name}-gpu${device_id}.bm

echo -e 'GPU:'${device_id}'\nNetwork: '${network_name}'\nSeqlen: '${seqlen}'\nMinibatch: '${minibatch}'\nIterations: '${iterations}'\n_________________\n'>> ${benchmark_logfile}
echo -e 'ToolName\t\t\tAverageTime(s)'>>${benchmark_logfile}
tmplog=b${minibatch}-gpu${device_id}.log
th /home/comp/csshshi/Torch-Examples/rnn/examples/recurrent-language-model.lua --seqlen ${seqlen} --batchsize ${minibatch} --iters ${iterations} --hiddensize ${hiddensize} ${default} >& ${tmplog}
total_time=`cat ${tmplog} | grep "Time elapsed for " | awk '{print $6}'` 
time_in_second=`awk "BEGIN {print ${total_time}/${iterations}}"`

echo -e ${tool}'\t\t\t'${time_in_second}>>${benchmark_logfile}
echo -e '\n\n'>>${benchmark_logfile}
