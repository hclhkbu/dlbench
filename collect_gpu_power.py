import argparse
import os
import subprocess
import time
import numpy as np


def benchmark_is_runing(running_cmd):
    cmd = 'ps aux| grep \"%s\" | grep -v grep' % running_cmd
    try:
        result = subprocess.check_output(cmd, shell=True)
        if len(result) > 0:
            return True
        else:
            return False
    except:
        return False
    return False 


def start_collecting_gpu_power(running_cmd, log_file):
    cmd = 'nvidia-smi'
    log = open(log_file, "w")
    time.sleep(1)
    while benchmark_is_runing(running_cmd):
        result = subprocess.check_output(cmd, shell=True)
        log.write(result)
        time.sleep(1)
    log.close()


def get_average_gpu_power_and_mem(gpu_name, log_file):
    log = open(log_file, 'r')
    content = log.readlines()
    powers = []
    mems = []
    for index, line in enumerate(content):
        if line.find(gpu_name[len(gpu_name)-3:]) > 0:
            if index == len(content) - 1:
                break
            valid_line = content[index+1].lstrip()
            items = valid_line.split(' ')
            for item in items:
                if item.find('W') > 0:
                    power = float(item.split('W')[0])
                    break
            for item in items:
                if item.find('MiB') > 0:
                    memory = float(item.split('MiB')[0])
                    break

            powers.append(power)
            mems.append(memory)
    log.close()
    return np.mean(powers[2:len(powers)-1]), np.mean(mems[2:len(mems)-1]) 


if __name__ == '__main__':
    #running_cmd = 'python testing.py'
    #start_collecting_gpu_power(running_cmd, 'debug.log')
    power, mem = get_average_gpu_power_and_mem('GTX980', 'logs/power_mxnet-fc-fcn5-TitanX_Pascal-devId1-c1-b4096-Thu_Jul__6_10:39:09_2017-hpclgpu.log')
    print power, mem
