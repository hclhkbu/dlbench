import argparse
import os,sys
import time
import subprocess

# Parse arguments
current_time = time.ctime()
parser = argparse.ArgumentParser(description='Python script benchmarking mxnet')
parser.add_argument('-log', type=str, default=('mxnet_' + current_time + '.log').replace(" ", "_"),
        help='Name of log file, default= mxnet_ + current time + .log') 
parser.add_argument('-batchSize', type=str, default='64', help='Batch size for each GPU, default = 64')
parser.add_argument('-network', type=str, default='fcn5', help='name of network[fcn5 | fcn8 | alexnet | resnet | lstm32 | lstm64]')
parser.add_argument('-devId', type=str, help='CPU: -1, GPU:0,1,2,3(Multiple gpu supported)')
parser.add_argument('-numEpochs', type=str, default='10', help='number of epochs, default=10')
parser.add_argument('-epochSize', type=str, default='50000', help='number of training data per epoch')
parser.add_argument('-numThreads', type=str, default='8', help='number of Threads, used only if -devId is set to -1, default=8')
parser.add_argument('-netType', type=str, help='network type(experiment dir)')
parser.add_argument('-debug', type=bool, default=False, help='Debug mode or benchmark mode')

args = parser.parse_args()
if args.debug: print(args)

# Build cmd
exePath = ""
if args.netType is None:
	print "You must set the network type!"
	sys.exit()
else:
	exePath = args.netType + "/" + args.network + "/"
	if os.path.exists(exePath) is not True: exePath = args.netType + "/"
os.chdir(exePath)
if args.debug:  print "Working dirctory: " +str(os.getcwd())
cmd = ""
pyscript=""
network = args.network
numSamples = args.epochSize
logFile = ""
if args.netType == 'fc':
    	tmpnetwork = 'fcn5'
        if args.network == 'ffn26752l6':
            tmpnetwork = 'fcn8'
    	pyscript = "python train_fake_data.py --network " + tmpnetwork
elif args.netType == 'cnn':
	pyscript = "python train_imagenet.py "
elif args.netType == 'lstm32':
	pyscript = "python train_rnn.py --sequence-lens 32 --num-hidden 256 --num-embed 256 --num-lstm-layer 2 "
elif args.netType == 'lstm64':
	pyscript = "python train_rnn.py --sequence-lens 64 --num-hidden 256 --num-embed 256 --num-lstm-layer 2 "
else:
	print("Unknow network type " + args.network + ". Try with --help")

if args.devId is None:
	print("You must set a running device!")
	sys.exit(-1)
elif args.devId == "-1":
	pyscript = "export OMP_NUM_THREADS=" + args.numThreads + "; export OPENBLAS_NUM_THREADS=" + args.numThreads + "; export MXNET_CPU_WORKER_NTHREADS=" + args.numThreads + ";" + pyscript
else:
	pyscript += " --gpus " + args.devId
	#if len(args.gpus.split(',')) > 1: pyscript += " --kv-store device "

cmd += pyscript + " --num-epochs " + args.numEpochs + " --batch-size " + args.batchSize + " --num-examples " + numSamples
if ".log" not in args.log:
	args.log += ".log"
cmd += " >& " + args.log 
logFile = args.log

if args.debug: print cmd 
t = time.time()
os.system(cmd)
t = time.time() - t
if args.debug: print("Time difference: " + str(t))
samplePerSec = subprocess.check_output("cat " + logFile + "| grep Speed | cut -d']' -f4 | cut -d':' -f2 | cut -d' ' -f2", shell=True).strip().split('\n')
totalSpeed = 0
for s in samplePerSec:
	totalSpeed += float(s)
avgBatch = float(args.batchSize)/(totalSpeed/len(samplePerSec))
if args.debug: print "Average batch: " + str(avgBatch)
with open(logFile, "a") as l:
	l.write("Total time: " + str(t) + "\n")
	l.write("cmd: " + cmd + "\n")
	l.write("AverageBatch " + str(avgBatch) + "\n")
