#!/usr/bin/env python

import settings
import argparse
import time
import requests
import json

from pymongo import MongoClient

mongo_client = MongoClient("mongodb://%s:%s/" % (settings.MONGO_HOST, settings.MONGO_PORT))
db = mongo_client[settings.MONGO_DBNAME]
if settings.MONGO_AUTH_USER_NAME:
    auth = db.authenticate(settings.MONGO_AUTH_USER_NAME, settings.MONGO_AUTH_PASSWORD)

def post_record(**args):
    """
    Args: flag, network, batch_size, device_name, gpu_count, 
          cpu_count, epoch_size, epoch, total_time, average_time, 
          tool_name, avg_mem, epoch_info, log_file, cuda,
          cudnn, cuda_driver
    """
    create_time = int(time.time() * 1000)
    args['create_time'] = create_time
    print 'args: ', args
    data = json.dumps(args) 
    ret = requests.post(settings.RESOURCE_URI, {'data': data})
    print ret
    #db['record'].insert(args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Post experiments record tool')
    parser.add_argument('-f', '--flag', help='Flag for each group of experiments')
    parser.add_argument('-n', '--network', help='Name of deep neural network, e.g., FCN5, AlexNet')
    parser.add_argument('-b', '--batch_size', help='Mini-batch size')
    parser.add_argument('-d', '--device_name', help='Device name, e.g., GTX1080, GTX980, K80')
    parser.add_argument('-g', '--gpu_count', help='The number of GPU used')
    parser.add_argument('-c', '--cpu_count', help='The number of CPU used')
    parser.add_argument('-P', '--cpu_name', help='The name of CPU used')
    parser.add_argument('-e', '--epoch_size', help='Epoch size')
    parser.add_argument('-E', '--epoch', help='The number of epoch')
    parser.add_argument('-t', '--total_time', help='The total time used (in second)')
    parser.add_argument('-a', '--average_time', help='The average time for each mini-batch (in second)')
    parser.add_argument('-T', '--tool_name', help='The tool name')
    parser.add_argument('-A', '--average_mem', help='Avg memory (in Megabytes')
    parser.add_argument('-I', '--epoch_info', help='Epoch infomation')
    parser.add_argument('-l', '--log_file', help='The full path of the log file')
    parser.add_argument('-C', '--cuda', help='The version of CUDA', default='8.0')
    parser.add_argument('-D', '--cudnn', help='The version of cuDNN', default='5.1')
    parser.add_argument('-r', '--cuda_driver', help='The version of cuda driver', default='367.48')
    parser.add_argument('-v', '--experiment_version', help='The version of running', default='v8')
    p = parser.parse_args()
    object_id = post_record(flag=p.flag, network=p.network, batch_size=p.batch_size, device_name=p.device_name,
                gpu_count=p.gpu_count, cpu_count=p.cpu_count, cpu_name=p.cpu_name, epoch_size=p.epoch_size, epoch=p.epoch,
                total_time=p.total_time, average_time=p.average_time, tool_name=p.tool_name, avg_mem=p.average_mem, 
                epoch_info=p.epoch_info, log_file=p.log_file, cuda=p.cuda, cudnn=p.cudnn, cuda_driver=p.cuda_driver)
    #object_id = post_record(flag='test', network='network')
    print 'post finished, object_id: ', object_id

