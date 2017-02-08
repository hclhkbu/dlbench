import argparse
import os, sys
import time
import subprocess

# Parse arguments, don't change input args
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
parser.add_argument('-lr', type=str, help='learning rate')
parser.add_argument('-netType', type=str, help='network type')
parser.add_argument('-debug', type=bool, default=False, help='Debug mode or not')
args = parser.parse_args()
if args.debug is True: print(args)


# Build cmd for benchmark
cmd = ''


# Execute cmd 
if args.debug is True: print(cmd)
t = time.time()
os.system(cmd)
t = time.time() - t
if args.debug is True: print("Time diff: " + str(t))


#Save log file
logPath = '' 
with open(logPath, "a") as logFile:
    logFile.write("Total time: " + str(t) + "\n")
    logFile.write("cmd: " + cmd + "\n")
os.system("cp " + logPath + " ../../logs")


# Parse log file and extract benchmark info
avgBatch = (avgEpoch/int(numSamples))*float(batchSize)
if args.debug is True: print("Avg Batch: " + str(avgBatch))
info = ''
print "-t " + str(t) + " -a " + str(avgBatch) + " -I " + info
