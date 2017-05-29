no_use_cnt=0
sleep 5
echo $no_use_cnt
while (($no_use_cnt < 4))
do
	idle_cnt=`nvidia-smi | grep 0% | wc -l`
	logTime=`date`
	if (($idle_cnt > 3)) 
	then
		echo "[$logTime] no_use_cnt++"
		no_use_cnt=$(($no_use_cnt + 1))
	else
		echo "[$logTime] no_use_cnt=0"
		no_use_cnt=0
	fi
	echo "no_use_cnt = $no_use_cnt"
	sleep 5
done

sleep 5

for ((i=10; i<=21; i++))
do
	echo Kill process in gpu$i;
	ssh gpu$i "./killMXnet.sh";
done
