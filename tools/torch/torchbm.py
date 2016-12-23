import argparse
import os, sys
import time
import subprocess

# Parse arguments
current_time = time.ctime()
parser = argparse.ArgumentParser(description='Python script benchmarking torch')
parser.add_argument('-log', type=str, default=('torch_' + current_time + '.log').replace(" ", "_"),
        help='Name of log file, default= torch_ + current time + .log') 
parser.add_argument('-batchSize', type=str, default='64', help='Batch size in each GPU, default = 64')
parser.add_argument('-network', type=str, default='fcn5', help='name of network[fcn5 | alexnet | resnet | lstm32 | lstm64]')
parser.add_argument('-devId', type=str, help='CPU: -1, GPU: 0,1,2,3 (Multiple gpu supported)')
parser.add_argument('-numEpochs', type=str, default='10', help='number of epochs, default=10')
parser.add_argument('-epochSize', type=str, help='number of training data per epoch')
parser.add_argument('-numThreads', type=str, default='8', help='number of Threads, default=8')
parser.add_argument('-hostFile', type=str, help='path to running hosts(config in host file) for multiple machine training.')
parser.add_argument('-gpuCount', type=str, default='1', help='number of gpus in used')
parser.add_argument('-cpuCount', type=str, default='1', help='number of cpus in used for cpu version')
parser.add_argument('-lr', type=str, default='0.01', help='learning rate')
parser.add_argument('-netType', type=str, help='network type')
parser.add_argument('-debug', type=bool, default=False, help='debug mode')

args = parser.parse_args()
if args.debug: print("args: " + str(args))

# Set system variable
#os.environ['OMP_NUM_THREADS'] = args.cpuCount 
#os.environ['OPENBLAS_NUM_THREADS'] = args.cpuCount 

# Build cmd
cmd = "THC_CACHING_ALLOCATOR=1 th Main.lua "
network = args.network
numSamples = 0
if network == "fcn5":
    cmd += "-LR " + args.lr +" -dataset MNIST -network ffn5"
    numSamples = args.epochSize
elif network == "alexnet" or network == "resnet":
    if args.devId == '-1':
        cmd += " -LR " + args.lr + " -network " + network 
    else:
        cmd += "-network " + network + " -LR " + args.lr + " "
    numSamples = args.epochSize
elif "lstm" in network:
	if args.devId is not None:
		if "-" not in args.devId:
			cmd = "THC_CACHING_ALLOCATOR=1 CUDA_VISIBLE_DEVICES=" + args.devId  + " th rnn/recurrent-language-model.lua --cuda " 
		else:
			cmd = "OMP_NUM_THREADS=%s OPENBLAS_NUM_THREADS=%s th rnn/recurrent-language-model.lua --lstm --startlr 1 " % (args.cpuCount, args.cpuCount)
	else:
		print("Device not set, please set device by adding -devId <-1 or 0,1,2,3>. See help for more")
		sys.exit(-2)
	if "64" in network:
		cmd += " --seqlen 64 "
	else:
		cmd += " --seqlen 32 "
	cmd += "--lstm --hiddensize '{256,256}' --startlr " + args.lr + " --minlr " + args.lr + " " 
	cmd += "--batchsize " + args.batchSize + " --maxepoch " + args.numEpochs
	logfile = args.log
	if ".log" not in logfile:
		logfile += ".log"
	cmd += " >& " + logfile
	if args.debug: print "cmd: " + cmd
	t = time.time()
	os.system(cmd)
	t = time.time() - t
	if args.debug: print "total time: " + str(t)
	with open(logfile, "a") as logFile:
		logFile.write("Total time: " + str(t) + "\n")
		logFile.write(cmd + "\n")
	os.system("cp " + logfile + " ../../logs")
	catLog = "cat " + logfile
	totalEpochBatchTime = subprocess.check_output( catLog + " | grep Speed | cut -d':' -f2 | paste -sd+ - | bc", shell=True).strip()
	numEpoch = subprocess.check_output(catLog + " | grep Speed | cut -d':' -f2 | wc -l", shell=True).strip()
	avgBatch = float(totalEpochBatchTime)/float(numEpoch)
	avgBatch = avgBatch/1000.0
	if args.debug: print("Avg Batch: " + str(avgBatch))
	trainPPL = subprocess.check_output(catLog + "|grep \"Training PPL\" | cut -d':' -f2", shell=True).replace("	"," ").strip().split("\n")
	valPPL = subprocess.check_output(catLog + "|grep \"Validation PPL\" | cut -d':' -f2", shell=True).replace("	"," ").strip().split("\n")
	if args.debug: print "trainPPL: " + trainPPL
	if args.debug: print "valPPL: " + valPPL
	info = " -I "
	for i in range(int(numEpoch)):
		if i != 0:
			info += ","
		info += str(i) + ":" + valPPL[i].strip() + ":" + trainPPL[i].strip()
	print " -t " + str(t) + " -a " + str(avgBatch) + info
	with open(logfile, "a") as logFile:
		logFile.write("Total time: " + str(t) + "\n")
	os.system("cp " + logfile + " ../../logs")
	sys.exit(0)
else:
    print("Unknown network type " + network + ", supported ones: fcn5, alexnet, resnet, lstm32, lstm64")
    sys.exit(-1)


devId = args.devId
nGPU = int(args.gpuCount) 
if devId is not None:
    if "-" not in devId:
        if nGPU > 1:
            cmd += " -nGPU " + str(nGPU)
        cmd = "CUDA_VISIBLE_DEVICES=" + devId + " " + cmd
    #elif "-1" == devId and (network == "ffn" or network=="fcn5"):
    elif "-1" == devId:
        cmd += "_cpu -nGPU 0 -type float -threads " + args.cpuCount
    else:
        print("Only CNN is not supported on CPU in torch")
        sys.exit(-2)
else:
    print("Device not set, please set device by adding -devId <-1 or 0,1,2,3>. See help for more")
    sys.exit(-2)

if args.devId == '-1':
    batchSize = args.batchSize
else:
    batchSize = int(args.batchSize)*nGPU
numEpochs = args.numEpochs
cmd += " -batchSize " + str(batchSize) + " -epoch " + numEpochs

logfile = args.log
if ".log" not in logfile:
    logfile += ".log"

cmd += ' -logfile ' + logfile + ' -save ' + logfile + " >& display.tmp"
cmd = 'THC_CACHING_ALLOCATOR=1 ' + cmd
if args.debug: print("cmd:" + cmd)
t = time.time()
os.system(cmd)
t = time.time() - t
with open(logfile, "a") as logFile:
	logFile.write("Total time: " + str(t) + "\n")
	logFile.write(cmd + "\n")
os.system("cp " + logfile + " ../../logs")
if args.debug: print("Time diff: " + str(t))
os.system("rm display.tmp")
getInfo = 'cat ' + logfile + ' | grep Info'
totalEpochTime = subprocess.check_output( getInfo + " | grep time | cut -d' ' -f6 | cut -d':' -f2 | paste -sd+ - | bc", shell=True)
numEpoch = subprocess.check_output(getInfo + " | grep time | cut -d' ' -f6 | cut -d':' -f2 | wc -l", shell=True)
if args.debug: print "totalEpochTime: " + totalEpochTime
if args.debug: print "numEpoch: " + numEpoch
avgEpoch = 0
if int(numEpoch) != 0:
    avgEpoch = float(totalEpochTime)/float(numEpoch)
avgBatch = (avgEpoch/int(numSamples))*float(batchSize)
if args.debug: print("Avg Batch: " + str(avgBatch))

valAccuracy = subprocess.check_output(getInfo + "| grep accu | cut -d' ' -f7 | cut -d':' -f2", shell=True).strip().split('\n')
if args.debug: print "valAccuracy: " + valAccuracy
trainCE = subprocess.check_output(getInfo + "| grep Loss | cut -d' ' -f7 | cut -d':' -f2", shell=True).strip().split('\n')
if args.debug: print "trainCE: " + trainCE

info = ""
for i in range(len(valAccuracy)):
    if i != 0:
        info += ","
    info += str(i) + ":" + valAccuracy[i] + ":" +trainCE[i]
if args.debug: print "info: " + info 
print "-t " + str(t) + " -a " + str(avgBatch) + " -I " + info
