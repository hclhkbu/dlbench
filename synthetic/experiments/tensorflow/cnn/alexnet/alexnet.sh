# for cuda 8.0
export CUDA_PATH=/usr/local/cuda

export PATH=$CUDA_PATH/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_PATH/lib64:$LD_LIBRARY_PATH
export PATH=/usr/local/openmpi/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/openmpi/lib:$LD_LIBRARY_PATH

# for tensorflow which supports cuDNN-4.0 only
export LD_LIBRARY_PATH=/usr/local/cudnn-4.0/cuda/lib64:$LD_LIBRARY_PATH

python alexnet.py
