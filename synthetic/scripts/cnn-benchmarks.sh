#!/bin/bash
# The benchmarks of all toolkits 

###########
# CNN
###########
#REPO_HOME=/home/comp/csshshi/repositories/dpBenchmark/synthetic
REPO_HOME=/home/ipdps/dpBenchmark/synthetic
current_path=$REPO_HOME/scripts
experiments_path=$REPO_HOME/experiments
log_path=$REPO_HOME/logs

retrain_step=1

# RetNet-50 or AlexNet
#############
minibatch="${minibatch:-16}"
iterations="${iterations:-8}"
epochs="${epochs:-4}"
network_type="${network_type:-cnn}"
network_name="${network_name:-alexnet}"
device_id="${device_id:-0}"

OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
FLAG="${FLAG:-syntheticSgbenchmark2}" # start from 1 
#cpu_name="E5-2630v3"
cpu_name="i7-3820"
device_name="${device_name:-K80}"
epoch_size=null
cuda="8.0"
cudnn="5.1"
cuda_driver="367.48"

running_time=`date +"%Y%m%d-%H:%M:%S"`
hostName=`hostname`


#tools=( "caffe" "cntk" "tensorflow" "torch" )
tools=( "cntk" "torch" )
benchmark_logfile=${current_path}/${network_type}-${network_name}-gpu${device_id}.bm
echo -e 'GPU:'${device_id}'\nNUM_THREADS (for CPU): '${OMP_NUM_THREADS}'\nNetwork: '${network_name}'\nEpochs: '${epochs}'\nMinibatch: '${minibatch}'\nIterations: '${iterations}'\nBenchmark Time: '${running_time}'\n_________________\n'>> ${benchmark_logfile}
echo -e 'ToolName\t\t\tAverageTime(s)'>>${benchmark_logfile}
for tool in "${tools[@]}" 
do
    tool_path=${experiments_path}/${tool}/${network_type}/${network_name}
    if [ ! -d "${tool_path}" ]
    then
        continue
    fi
    cd ${tool_path}
    total_time=0.0
    echo 'start to benchmark: '${tool}'...'
    #tmplog=${network_type}-${network_name}-gpu${device_id}.log
    device=gpu${device_id}
    if [ ${device_id} -gt -1 ]
    then
        device=gpu${device_id}
    else
        device=cpu${OMP_NUM_THREADS}
    fi
    tmplog=${tool}-${network_type}-${network_name}-${device}-${device_name}-b${minibatch}-${running_time}-${hostName}.log

    tmpresultlog=result-${network_type}-${network_name}-gpu${device_id}.bm
    echo -e 'GPU:'${device_id}'\nNetwork: '${network_name}'\nEpochs: '${epochs}'\nMinibatch: '${minibatch}'\nIterations: '${iterations}'\n_________________\n'>> ${tmpresultlog}
    #echo -e 'Epochs: '${epochs}'\nMinibatch: '${minibatch}'\nIterations: '${iterations}'\n_________________\n'> ${tmpresultlog}

    source ~/.bashrc 
    for step in `seq 1 ${retrain_step}`
    do
        time_in_second=0.0
        if [ ${tool} = "caffe" ] 
        then
            iters_for_caffe=`awk "BEGIN {print ${iterations}*${epochs}}"`
            rm _iter*
            if [ ${device_id} -gt -1 ]
            then
                echo 'caffe gpu'
                #caffe time -model=${network_name}-b${minibatch}.prototxt -iterations=${iters_for_caffe} -gpu=${device_id} >& ${tmplog} 
                caffe train -solver=${network_name}-b${minibatch}-solver-GPU.prototxt -gpu=${device_id} >& ${tmplog} 
            else
                echo 'caffe cpu'
                OPENBLAS_NUM_THREADS=${OMP_NUM_THREADS}
                #caffe time -model=${network_name}-b${minibatch}.prototxt -iterations=${iters_for_caffe} >& ${tmplog} 
                caffe train -solver=${network_name}-b${minibatch}-solver-CPU.prototxt >& ${tmplog} 
            fi
            #caffe time -model=${network_name}-b${minibatch}.prototxt -iterations=${iters_for_caffe} -gpu=${device_id} >& ${tmplog} 
            #ms=`grep -r "Average Forward-Backward" ${tmplog} |awk '{print $7}'`
            #time_in_second=`awk "BEGIN {print ${ms}*0.001}"`
            loopno=0
            calculateno=0
            tmp_total_time=0
            while IFS= read -r line
            do
                #echo 'line: '$line
                if [ "$loopno" -eq 0 ]
                then
                    #echo '0'
                    prelineno=`echo $line |awk -F : '{print $1}'`
                    precurrent_time=`echo $line |awk '{print $2}'`
                    precurrent_iteration=`echo $line |awk '{print $6}'|awk -F , '{print $1}'`
                else
                    #echo '1'
                    curlineno=`echo $line |awk -F : '{print $1}'`
                    curcurrent_time=`echo $line |awk '{print $2}'`
                    curcurrent_iteration=`echo $line |awk '{print $6}'|awk -F , '{print $1}'`
                    intervalline=`awk "BEGIN {print $curlineno-$prelineno}"`
                    if [ $intervalline -eq "3" -o "$network_name" = "resnet" ] # no data read
                    then
                        #echo 'pre: '$precurrent_iteration
                        #echo 'cur: '$curcurrent_iteration
                        itercount=`awk "BEGIN {print $curcurrent_iteration-$precurrent_iteration}"`
                        #echo 'itercount: '$itercount
                        
                        t1=`echo $precurrent_time | awk -F: '{ printf ("%.16g\n", ($1 * 3600) + ($2 * 60) + $3) }'`
                        t2=`echo $curcurrent_time | awk -F: '{ printf ("%.16g\n", ($1 * 3600) + ($2 * 60) + $3) }'`
                        #echo 't1, t2: '$t1','$t2
                        #interval=`awk '{ printf("%.16g\n", ($t2 - $t1)/$itercount) }'`
                        interval=`awk "BEGIN {print (${t2}-${t1})/${itercount}}"`
                        #interval=`awk '{ printf ("%.16g\n", ($t2 - $t1)) }'`
                        #echo 'interval: '$interval
                        calculateno=`awk "BEGIN {print ${calculateno}+1}"`
                        tmp_total_time=`awk "BEGIN {print ${tmp_total_time}+${interval}}"`
                    fi
                    prelineno=$curlineno
                    precurrent_time=$curcurrent_time
                    precurrent_iteration=$curcurrent_iteration
                fi
                loopno=`awk "BEGIN {print ${loopno}+1}"`
                #echo 'add tmp_total_time:'${tmp_total_time}
            done < <(grep -n "solver.cpp:228" ${tmplog}) 
            time_in_second=`awk "BEGIN {print ${tmp_total_time}/${calculateno}}"`
 
        elif [ ${tool} = "cntk" ]
        then
            rm -rf Output/*
            if [ ${device_id} -gt -1 ]
            then
                epoch_size=2048 #`awk "BEGIN {print ${minibatch}*128}"`
                iterations=`awk "BEGIN {print ${epoch_size}/${minibatch}}"`
            else
                epoch_size=`awk "BEGIN {print ${minibatch}*${iterations}}"`
                epoch_size=2048 #`awk "BEGIN {print ${minibatch}*128}"`
                iterations=`awk "BEGIN {print ${epoch_size}/${minibatch}}"`
            fi
            #epoch_size=`awk "BEGIN {print ${minibatch}*${iterations}}"`
            cntk configFile=${network_name}.cntk configName=${network_name} deviceId=${device_id} minibatchSize=${minibatch} epochSize=${epoch_size} maxEpochs=${epochs} >& ${tmplog}
            #time_in_second=`grep -r "Epoch\[ " tmp.log | cut -d'=' -f 6 | cut -d's' -f 1`
            tmp_total_time=0.0
            lineno=0
            linecount=`grep -r "Finished Epoch\[ " ${tmplog} | wc -l`
            #echo "line count: "${linecount}
            while IFS= read -r line
            do
                lineno=`awk "BEGIN {print ${lineno}+1}"`
                if [ "${linecount}" -gt 1 -a "${lineno}" -eq 1 ]
                then
                    continue
                fi
                #time_in_second=`echo ${line} | cut -d'=' -f 6 | cut -d's' -f 1`
                time_in_second=`echo ${line} | grep -o '[^=]*$' | cut -d's' -f 1` 
                #echo 'time_in_second:'${time_in_second}
                tmp_total_time=`awk "BEGIN {print ${tmp_total_time}+${time_in_second}}"`
                #echo 'add tmp_total_time:'${tmp_total_time}
            done < <(grep "Finished Epoch\[ " ${tmplog}) 

            if [ "${linecount}" -gt 1 ]
            then
                linecount=`awk "BEGIN {print ${linecount}-1}"`
            fi
            #echo 'tmp_total_time:'${tmp_total_time}
            time_in_second=`awk "BEGIN {print ${tmp_total_time}/${linecount}}"`
            # need to calculate each minibatch time
            time_in_second=`awk "BEGIN {print ${time_in_second}/${iterations}}"`
            rm -rf Output/*
        elif [ ${tool} = "tensorflow" ]
        then
            #. ../../export-local-var.sh
            source ~/tf11/bin/activate
            python ${network_name}bm.py -e ${epochs} -b ${minibatch} -i ${iterations} -d ${device_id} >& ${tmplog}
            time_in_second=`cat ${tmplog} | tail -n1 | awk '{print $8}'`
            deactivate
        elif [ ${tool} = "torch" ]
        then
            #. /usr/local/torch/bin/torch-activate
            th ${network_name}bm.lua -depth 50 -nGPU 1 -nThreads 2 -deviceId ${device_id} -batchSize ${minibatch} -shareGradInput true -nEpochs ${epochs} -nIterations ${iterations} -dataset imagenet -data ~/data/ >& ${tmplog}
            tmp_total_time=0.0
            lineno=0
            linecount=`grep -r "Epoch: " ${tmplog} | wc -l`
            #echo "line count: "${linecount}
            while IFS= read -r line
            do
                lineno=`awk "BEGIN {print ${lineno}+1}"`
                if [ "${linecount}" -gt 1 -a "${lineno}" -eq 1 ]
                then
                    continue
                fi
                #time_in_second=`echo ${line} | cut -d'=' -f 6 | cut -d's' -f 1`
                time_in_second=`echo ${line} | awk '{print $5}'` 
                #echo 'time_in_second:'${time_in_second}
                tmp_total_time=`awk "BEGIN {print ${tmp_total_time}+${time_in_second}}"`
                #echo 'add tmp_total_time:'${tmp_total_time}
            done < <(grep "Epoch: " ${tmplog}) 
            if [ "${linecount}" -gt 1 ]
            then
                linecount=`awk "BEGIN {print ${linecount}-1}"`
            fi
            time_in_second=`awk "BEGIN {print ${tmp_total_time}/${linecount}}"`
        fi 
        echo "train step"${step}": "${time_in_second} >> ${tmpresultlog}
        total_time=`awk "BEGIN {print ${total_time}+${time_in_second}}"`
    done
    echo -e '\n\n'>>${tmpresultlog}
    average_time=`awk "BEGIN {print ${total_time} / ${retrain_step}}"`
    echo ${tool}' finished, avevrage time: '${average_time}
    echo -e ${tool}'\t\t\t'${average_time}>>${benchmark_logfile}

    mv $tmplog $log_path/
    cd $current_path
    subargs="-a ${average_time}"
    if [ ${device_id} -eq -1 ]
    then
        device_name=$cpu_name
    fi

    args="-f ${FLAG} -n ${network_name} -b ${minibatch} -d ${device_name} -g 1 -c ${OMP_NUM_THREADS} -P ${cpu_name} -e ${epoch_size} -E ${epochs} -A unknown -l ${tmplog} -T ${tool} ${subargs}"
    python post_record.py ${args}
done
echo -e '\n\n'>>${benchmark_logfile}


