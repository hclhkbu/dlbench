## Instruction of Caffe benchmarking ##
Please be noted that you should successfully install Caffe binary file.

### 1. Data preparation ###
- Create directory: caffe to store all the experiment data in the home directory. `$mkdir -p ~/data/caffe`
- MNIST dataset: In the directory of the source code of Caffe, execute: 
```
	$./data/minst/get_mnist.sh
	$./examples/mnist/create_mnist.sh
	$mkdir ~/data/caffe/mnist
	$mv ./examples/mnist/*_lmdb ~/data/caffe/mnist/
```
- Cifar10 dataset
```
	$./data/cifar10/get_cifar10.sh
	$./examples/cifar10/create_cifar10.sh
	$mkdir ~/data/caffe/cifar10
	$mv ./examples/cifar10/*_lmdb ~/data/caffe/cifar10/
```
### 2. Dependency packages ###
- OpenBLAS==0.2.18
- OpenCV=3.1.0

### 3. Example ###
	
	python caffebm.py -log test.log -batchSize 1024 -netType fc -network fcn5 -devId 0 -numEpochs 40 -numThreads 1 -gpuCount 1 -lr 0.5 -cpuCount 
