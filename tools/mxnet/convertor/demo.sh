echo "=======Demo======="
echo "You may need to check the config file and .miss file after converting to mxnet"
echo "Enter input Caffe config file path (requried):"
read caffePath
java caffe2mxnet/Caffe2mxnet -caffe $caffePath >> $caffePath\.miss
