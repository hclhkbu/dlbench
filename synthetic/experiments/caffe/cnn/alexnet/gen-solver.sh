solver_base=solver-base.prototxt
mode=${mode:-GPU}
for file in ./alexnet-b*.prototxt
do
    fname=${file##.*/}
    fnameprefix=`echo $fname | awk -F . '{print $1}'`
    networkname=`echo $fnameprefix | awk -F - '{print $1}'`
    minibatchname=`echo $fnameprefix | awk -F - '{print $2}'`
    minibatch=`echo $minibatchname| awk -F b '{print $2}'`
    solvername=$fnameprefix-solver-$mode.prototxt 
    echo $solvername
    cp $solver_base $solvername
    echo "net: \"$fname\"" >> $solvername
    echo "solver_mode: $mode" >> $solvername
done

