#!/bin/bash

# for cuda 7.5
export CUDA_PATH=/usr/local/cuda-7.5

export LD_LIBRARY_PATH=
export PATH=$CUDA_PATH/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_PATH/lib64:$LD_LIBRARY_PATH
export PATH=/usr/local/openmpi/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/openmpi/lib:$LD_LIBRARY_PATH
#export PATH=/home/dl/dsstne/amazon-dsstne/src/amazon/dsstne/bin:$PATH
#export PATH=/home/dl/dsstne/amazon-dsstne/src/amazon/dsstne2/bin:$PATH # specify to GTX 980
export PATH=/home/dl/dsstne/amazon-dsstne/src/amazon/dsstne0/bin:$PATH # specify to GTX 1080

#export PATH=/home/dl/downloads/cntk/cntk/bin:$PATH # downloaded cntk banary which is within cuda 7.5
export PATH=/home/dl/downloads/cntk-src/cntk/build/release/bin:$PATH # self-compiled with cuda 8.0
#export LD_LIBRARY_PATH=/home/dl/downloads/cntk/cntk/lib:/home/dl/downloads/cntk/cntk/dependencies/lib:$LD_LIBRARY_PATH
export ACML_FMA=0

# for tensorflow which supports cuDNN-4.0 only
export LD_LIBRARY_PATH=/usr/local/cudnn-4.0/cuda/lib64:$LD_LIBRARY_PATH

export PATH=/home/dl/caffe/build/tools:$PATH
export PYTHONPATH=/home/dl/caffe/python:$PYTHONPATH


