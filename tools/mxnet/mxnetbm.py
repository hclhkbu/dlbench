import argparse
import os, sys
import time
import subprocess

# Parse arguments
current_time = time.ctime()
parser = argparse.ArgumentParser(description='Python script benchmarking mxnet')
parser.add_argument('-log', type=str, default=('mxnet_' + current_time + '.log').replace(" ", "_"),
        help='Name of log file, default= mxnet_ + current time + .log') 
parser.add_argument('-batchSize', type=str, default='64', help='Batch size for each GPU, default = 64')
parser.add_argument('-network', type=str, default='fcn5', help='name of network[fcn5 | alexnet | resnet | lstm32 | lstm64]')
parser.add_argument('-devId', type=str, help='CPU: -1, GPU:0,1,2,3(Multiple gpu supported)')
parser.add_argument('-numEpochs', type=str, default='10', help='number of epochs, default=10')
parser.add_argument('-epochSize', type=str, default='50000', help='number of training data per epoch')
parser.add_argument('-numThreads', type=str, default='8', help='number of Threads, default=8')
parser.add_argument('-hostFile', type=str, help='path to running hosts(config in host file) for multiple machine training.')
parser.add_argument('-gpuCount', type=str, help='number of gpus in used')
parser.add_argument('-cpuCount', type=str, default='1', help='number of cpus in used for cpu version')
parser.add_argument('-lr', type=str, help='learning rate')
parser.add_argument('-netType', type=str, help='network type')
parser.add_argument('-debug', type=bool, default=False, help='debug mode or benchmark mode')

args = parser.parse_args()
if args.debug: print("==================Debug test " + args.network + "====================")
if args.debug: print( "args: " + str(args))

# Set system variable
os.environ['OMP_NUM_THREADS'] = args.cpuCount 
os.environ['OPENBLAS_NUM_THREADS'] = args.cpuCount 

# Build cmd
exePath = ""
cmd = "cd "
pyscript=""
network = args.network
numSamples = args.epochSize
if network == "fcn5":
    exePath = "fc/"
    cmd += "fc; "
    pyscript = "python train_mnist.py --lr " + args.lr
elif network == "alexnet" or network == "resnet":
    exePath = "cnn/"
    cmd += "cnn; "
    pyscript = "python train_cifar10_" + network + ".py --lr " + args.lr
elif "lstm" in network:
    exePath = "rnn/"
    cmd += "rnn; "
    pyscript = "python train_rnn.py --lr " + args.lr
    if "64" in network:
        pyscript += " --sequence-lens 64"
    else:
        pyscript += " --sequence-lens 32"
else:
    print("Unknown network type " + network + ", supported ones: fcn, alexnet, resnet, lstm32, lstm64")
    sys.exit(-1)

numNodes = 1 if args.hostFile is None else int(subprocess.check_output("cat " + args.hostFile + " | wc -l", shell=True).strip())
if args.hostFile is not None:
    cmd += "PS_VERBOSE=0 nohup python ../multi-nodes-support/launch.py --launcher ssh -n " + str(numNodes) + " -s 1 -H " + args.hostFile + " " + pyscript
    cmd += " --kv-store dist_sync" + " --num-nodes " + str(numNodes) + " " 
    if network == "resnet":  cmd += "--optimizer SGD "
    os.system("sh monitorkill.sh >& kill.log &")
else:
    cmd += pyscript

nGPU = len(args.devId.split(','))
batchSize = int(args.batchSize)*nGPU
if "alexnet" in network and batchSize == 2048: batchSize = 2049 #alexnet bug, 2048 will cause error
numEpochs = args.numEpochs
cmd += " --batch-size " + str(batchSize) + " --num-epochs " + numEpochs + " --num-examples " + numSamples

devId = args.devId
if devId is not None:
    if "-" not in devId:
        cmd += " --gpus " + devId 
	if "--kv-store" not in cmd:
            cmd += " --kv-store device"
    elif "-1" == devId:
        pass
        #os.environ["MXNET_CPU_WORKER_NTHREADS"] = args.numThreads
        #print os.environ["MXNET_CPU_WORKER_NTHREADS"]
    else:
        print("invalid devId!")
        sys.exit(-1)

else:
    print("Device not set, please set device by adding -devId <-1 or 0,1,2,3>. See help for more")
    sys.exit(-2)


logfile = args.log
if ".log" not in logfile:
    logfile += ".log"

cmd += " >& " + logfile
t = time.time()
if args.debug: print "cmd: " +  cmd
os.system(cmd)
t = time.time() - t
if args.hostFile is not None:
	if args.debug: print "kill monitorkill"
	os.system("kill -9 `ps auw | grep pengfei | grep monitorkill | awk '{print $2}'`")
	os.system("kill -9 `ps auw | grep pengfei | grep sleep | awk '{print $2}'`")
if args.debug: print("Time diff: " + str(t))
logPath = exePath + logfile
catLog = "cat " + logPath
with open(logPath, "a") as logFile:
    logFile.write("Total time: " + str(t) + "\n")
    logFile.write("cmd: " + cmd + "\n")
if args.debug: os.system(catLog);
os.system("cp " + logPath + " ../../logs")
totalEpochTime = subprocess.check_output( catLog + " | grep Time | cut -d\'=\' -f2 | paste -sd+ - | bc", shell=True)
numEpoch = subprocess.check_output(catLog + " | grep Time | cut -d\'=\' -f2 | wc -l", shell=True)
if args.debug: print "totalEpochTime: " + totalEpochTime
if args.debug: print "numEpoch: " + numEpoch
avgEpoch = 0
if int(numEpoch) != 0:
    avgEpoch = float(totalEpochTime)/float(numEpoch)
if args.debug: print "avgEpch: " + str(avgEpoch)
if "lstm" in network:
    numSamples = int(subprocess.check_output(catLog + " | grep \"len of data train\" | cut -d' ' -f5", shell=True))
avgBatch = (avgEpoch/int(numSamples))*float(batchSize)
if args.debug: print("Avg Batch: " + str(avgBatch))

if "lstm" not in network:
    valAccuracy = subprocess.check_output(catLog + "| grep Validation-a | cut -d'=' -f2", shell=True).strip().split('\n')
    if args.debug: print "valAccuracy: " + str(valAccuracy)
    trainCE = subprocess.check_output(catLog + "| grep Train-cross-entropy | cut -d'=' -f2", shell=True).strip().split('\n')
    if args.debug: print "trainCE: " + str(trainCE)
    
    info = ""
    for i in range(len(trainCE)/numNodes):
	valAcu = 0
	train_ce = 0
	for j in range(numNodes):
	    #valAcu += float(valAccuracy[i*numNodes + j])
	    train_ce += float(trainCE[i*numNodes + j])
        if i != 0:
            info += ","
        #info += str(i) + ":" + str(valAcu/numNodes) + ":" + str(train_ce/numNodes) 
        info += str(i) + ":" + "-" + ":" + str(train_ce/numNodes) 
    if args.debug: print "info: " + info 
    print "-t " + str(t) + " -a " + str(avgBatch) + " -I " + info
else:
    valPerplexity = subprocess.check_output(catLog + " | grep Val | cut -d'=' -f2", shell=True).strip().split('\n')
    trainPerplexity = subprocess.check_output(catLog + " | grep Speed | cut -d'=' -f2", shell=True).strip().split('\n')
    info = " -I "
    for i in range(len(valPerplexity)):
        if i != 0:
            info += ","
        info += str(i) + ":" + valPerplexity[i] + ":" + trainPerplexity[i]
    if args.debug: print "info:" + info
    print "-t " + str(t) + " -a " + str(avgBatch) + info

if args.debug: print("==================Debug test " + args.network + " End ====================")
