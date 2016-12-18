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
parser.add_argument('-cpuCount', type=str, default='1', help='number of cpus in used for cpu version')
parser.add_argument('-netType', type=str, help='network type')
args = parser.parse_args()
#print(args)

# Set system variable
os.environ['OMP_NUM_THREADS'] = args.cpuCount 
os.environ['OPENBLAS_NUM_THREADS'] = args.cpuCount 

# Build cmd for benchmark
root_path = os.path.dirname(os.path.abspath(__file__))
tool_path = root_path + "/" + args.netType
if os.path.exists(tool_path + "/" + args.network):
	tool_path = tool_path + "/" + args.network
os.chdir(tool_path)
gencmd = 'num_epochs=%s gpu_count=%s learning_rate=%s batch_size=%s ./gen-%s.sh' % (args.numEpochs, args.gpuCount, args.lr, args.batchSize, args.network)
#os.system("./gen-" + args.network + ".sh")
os.system(gencmd)
log_file = args.log
if ".log" not in log_file:
	log_file += ".log"
log_path = os.getcwd() + "/" + log_file
if args.devId == '-1':
    cmd = 'OMP_NUM_THREADS=' + args.cpuCount + ' OPENBLAS_NUM_THREADS='+args.cpuCount + ' caffe train -solver=' + args.network + '-b' + args.batchSize + '-CPU-solver.prototxt'
else:
    cmd = 'caffe train -solver=' + args.network + '-b' + args.batchSize + '-GPU-solver' + args.gpuCount + ".prototxt -gpu=" + args.devId
cmd += ' >& ' + log_path

## Execute cmd 
t = time.time()
os.system(cmd)
t = time.time() - t
os.system("rm _iter*")
## Parse log file and extract benchmark info
os.chdir(root_path)
print(subprocess.check_output("python ../common/extract_info.py -f " + log_path + " -t caffe", shell=True))


#Save log file
with open(log_path, "a") as logFile:
    logFile.write("Total time: " + str(t) + "\n")
    logFile.write("cmd: " + cmd + "\n")
os.system("cp " + log_path + " ../../logs")


