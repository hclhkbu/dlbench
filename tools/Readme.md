# How to add new tools #
## Overview
-  Put your tools' scripts directory here and copy the benchmark scrip  common/xxxbm.py (rename it to \<your tool name\>bm.py) and test script testbm.sh in it.
-  Make sure \<your tool name>bm.py will take all those parameters pre-defined, you may ignore some of them as long as your scripts work. \<your tool name>bm.py servers as an entry and will be invoked by ../benchmark.py who only cares input parameters and output results of \<your tool name>bm.py.
-  All tests in common/testbm.sh should be passed before put new tools into use
-  Please create a readme file in your tool's directory including details about tool setup, data preparation, dependencies and environment etc.   

## \<Your tool name>bm.py explained   

### Input options   

| Input Argument |                                               Details                                              |                   Default value                  |
|:--------------:|:--------------------------------------------------------------------------------------------------:|:------------------------------------------------:|
|      -log      | Name of log file                                                                                   | You may set your own default value for debugging |
|   -batchSize   | Batch size for each GPU. If you are using n GPUs for a test, n*batchSize will be fed to the model. | Depend on memory size                            |
|    -network    | Name of network, values can be [fcn5 / alexnet / resnet / lstm]                                    | fcn5                                             |
|     -devId     | Training device: -1 for CPU; 0 for GPU 0; 0,2 for GPU0 and GPU2.                                   | None but training device must be specified              |
|   -numEpochs   | Number of epochs                                                                                   | 10                                               |
|   -epochSize   | Number of training samples for a epoch (not all tools need this parameter)                          | 50000                                            |
|    -hostFile   | Path to the host file if you are running on multiple machines                                      | None*                                            |
|    -cpuCount   | If devId=-1, you need to specify how may CPUs will be used for training                            | 1                                                |
|    -gpuCount   | If devId != -1, you need to specify how many GPUs will be used                                     | None                                             |
|       -lr      | Learning rate                                                                                      | None                                             |
|    -netType    | Network type, values can be [fc / cnn / rnn]                                                       | None                                             |
|     -debug     | Boolean value, true for debugging this script.                                                     | False                                            |
###Output
-  \<your tool name>bm.py should print out the running result which will be taken by benchmark.py and post to the server, and the format of the result is:
```
-t $totalTimeInSecond -a $averageMiniBatchTimeInSecond -I $lossValueOfEachEpoch
```
Example of *$lossValueOfEachEpoch* (There are 4 epochs' item, and splitted by `,`, and 3 values in each item splitted by `:` represents epoch number, accuracy and cross entropy respectively.):
```
0:-:2.32620807,1:-:2.49505453,2:-:2.30122152,3:-:2.30028142
```
###Benchmark procedure    
#### 1. Build cmd.   
- In order to make this framework be competible with different types of deep learinig tools written in different languages, <tool>bm.py is only an interface that standardize the input and output. You need to use arguments above to determine the variable cmd, and it will be executed in a subshell by calling `os.system(cmd)` during which a log file must be genrated containing necessary information for post processing. Some tools will generate a log file automatically, if not redirect all stdout and stderr to the log file. The name of log file ends with ".log". Here are some examples of cmd:   

Caffe: `caffe train -solver=fcn5-b1024-GPU-solver1.prototxt -gpu=0 >& /root/dlbench/tools/caffe/fc/debug.log`   

Mxnet: `cd fc; python train_mnist.py --lr 0.05 --batch-size 4096 --num-epochs 40 --num-examples 60000 --gpus 1 --kv-store device >& mxnet_debug.log`

Torch: `THC_CACHING_ALLOCATOR=1 CUDA_VISIBLE_DEVICES=1 THC_CACHING_ALLOCATOR=1 th Main.lua -LR 0.05 -dataset MNIST -network ffn5 -batchSize 342 -epoch 40 -logfile torch_debug.log -save torch_debug.log`    

#### 2. Execute cmd    
- Normally you don't need to change anything in this part. Cmd will be executed and total running time will be measured as well.   

#### 3. Post processing    
- There're three pieces of information need to be printed out:   
**Total runing time** `t`: xxxbm.py handles it already   
**Average batch time** `avgBatch`: average training time for one batch of date. Note that if there are more than one GPU, the batch size is n\*args.batchSize since the input argument args.batchSize is for each GPU core. If new tool doesn't measure the batch batch time directly, you need to convert other metrics into seconds/batch here.   
**Batch info** `info`: grep test accuracy\* and cross entropy for each epoch from the logfile and form them as \<epoch number>:\<test accuracy>:\<cross entropy>,\<epoch number+1>:\<test accuracy>:\<cross entropy>,...   
Then just print out the result: `print "-t " + str(t) + " -a " + str(avgBatch) + " -I " + info`   
and save the original log file under \<project root path>/logs/    
*There's no need to test the model after each epoch for now, but we still leave the space for future use.
