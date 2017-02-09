import argparse
import sys,os,time
import subprocess


# Parse arguments
parser = argparse.ArgumentParser(description='Benchmark deep learning tools')
parser.add_argument('-config', type=str, help='Path to the config file')
parser.add_argument('-post', type=bool, default=False, help='Post to our server. You should keep it False')
parser.add_argument('-debug', type=bool, default=False, help='Debug benchmark.py')

args = parser.parse_args()

if args.debug: print "[DEBUG] args:" + str(args)

# Parse config file
config_experiments = False
experiments = ''
host_file = None
flag = ''
tools = ''
cpu_name = ''
device_name = ''
cuda_driver = ''
cudnn = ''
cuda = ''
cpu_count = '1'
if args.config is not None:
	with open(args.config) as f:
		content = f.readlines()
	#print content
	for line in content:
		line = line.split('#')[0].replace('\t','').replace('\n','')
		if len(line) < 1 or "None" in line:
			continue 
		if not config_experiments:	
			if "flag:" in line:
				flag = line.split(':')[1]
			elif "tools:" in line:
				tools = line.split(':')[1].split(',')
			elif "{" in line:
				config_experiments = True
			elif "host_file:" in line:
				host_file = line.split(':')[1]
			elif "cpu_name:" in line:
				cpu_name = line.split(':')[1]
			elif "device_name:" in line:
				device_name = line.split(':')[1]
			elif "cuda_driver:" in line:
				cuda_driver = line.split(':')[1]
			elif "cudnn:" in line:
				cudnn = line.split(':')[1]
			elif "cuda:" in line:
				cuda = line.split(':')[1]
			elif "cpu_count:" in line:
				cpu_count = line.split(':')[1]
		else:
			if "}" in line:
				config_experiments = False
				experiments = experiments[:len(experiments)-1].replace('\t','').replace(' ','').split(':')
			else:
				experiments += line + ':'
else:
	print("Please add -config <path to your config file>")
	sys.exit(0)

post_flags = " -f " + flag + " -d " + device_name + " -c " + cpu_count + " -P " + cpu_name + " -A unknown" + " -r " + cuda_driver + " -C " + cuda + " -D " + cudnn

if args.debug:
	print "[DEBUG] Defalut post flags:" + str(post_flags)
	print "[DEBUG] Tool(s):" + str(tools)
	print "[DEBUG] Experiments:" + str(experiments)

# Benchmark each tool
root_path = os.path.dirname(os.path.abspath(__file__))
host_name = subprocess.check_output("hostname", shell=True).strip().split('\n')[0]
if os.path.exists(root_path + "/logs/") is not True:
	os.system("rm -rf logs")
	print "Creating log directory... " + root_path + "/logs/"
	os.system("mkdir logs")

if args.debug:
	print "[DEBUG] Benchmark running on: " + host_name
	print "[DEBUG] Root path:" + root_path

for tool in tools:
	work_dir = root_path + "/tools/" + tool
	for experiment in experiments:
		os.chdir(work_dir)
		exp_args = experiment.split(";")
		print "\n-------Benchmarking " + tool + " " + exp_args[1] + "-------"
		log_file = tool + "-" + exp_args[0] + "-" + exp_args[1] + "-" +"gpu"+ exp_args[2] + "-" + device_name + "-" +"b"+ exp_args[4] + "-" 
		log_file += time.ctime()+ "-" + host_name + ".log"
		log_file = log_file.replace(" ","_")
		bm_script = "python " + tool + "bm.py" 
		bm_script += " -netType "+exp_args[0]+" -log "+log_file+" -batchSize "+exp_args[4]+" -network "+exp_args[1]+" -lr "+exp_args[7] + " -cpuCount " + cpu_count
		bm_script += " -devId " + exp_args[2] + " -numEpochs " + exp_args[5] + " -epochSize " + exp_args[6] + " -gpuCount " + exp_args[3]
		if host_file is not None and len(host_file) > 4:
			bm_script += " -hostFile " + host_file
		print bm_script
		try:
			result_args = subprocess.check_output(bm_script, shell=True).strip().split('\n')[0]
		except Exception as e:
			print "Benchmark failed with " + bm_script 
			os.system("cat " + root_path + "/logs/" + log_file)
			continue
		post_flags += " " +  result_args + " -b " + exp_args[4] + " -g " + exp_args[3] + " -e " + exp_args[6] + " -E " + exp_args[5] 
		post_flags += " -l " + log_file + " -T " + tool + " -n " + exp_args[1] 
		os.chdir(root_path)
		if args.post is True:
			post_script = "python post_record.py " + post_flags
			print post_script
			print(subprocess.check_output(post_script, shell=True).strip().split('\n')[0])
			post_flags = " -f " + flag + " -d " + device_name + " -c " + cpu_count + " -P " + cpu_name + " -A unknown" + " -r " + cuda_driver + " -C " + cuda + " -D " + cudnn
			post_script = ''
		else:
			print "Result:"
			print result_args
		print "Done!"

