# CPU Version
minibatch=16    iterations=3    epochs=2    device_id=-1     network_name=alexnet       OMP_NUM_THREADS=1 ./cnn-benchmarks.sh
minibatch=16    iterations=3    epochs=2    device_id=-1     network_name=alexnet       OMP_NUM_THREADS=2 ./cnn-benchmarks.sh
minibatch=16    iterations=3    epochs=2    device_id=-1     network_name=alexnet       OMP_NUM_THREADS=4 ./cnn-benchmarks.sh
minibatch=16    iterations=3    epochs=2    device_id=-1     network_name=alexnet       OMP_NUM_THREADS=8 ./cnn-benchmarks.sh
#minibatch=16    iterations=3    epochs=2    device_id=-1     network_name=alexnet       OMP_NUM_THREADS=16 ./cnn-benchmarks.sh
#minibatch=16    iterations=3    epochs=2    device_id=-1     network_name=alexnet       OMP_NUM_THREADS=32 ./cnn-benchmarks.sh
#
minibatch=16    iterations=2    epochs=3    device_id=-1     network_name=resnet        OMP_NUM_THREADS=1 ./cnn-benchmarks.sh
minibatch=16    iterations=2    epochs=3    device_id=-1     network_name=resnet        OMP_NUM_THREADS=2 ./cnn-benchmarks.sh
minibatch=16    iterations=2    epochs=3    device_id=-1     network_name=resnet        OMP_NUM_THREADS=4 ./cnn-benchmarks.sh
minibatch=16    iterations=2    epochs=3    device_id=-1     network_name=resnet        OMP_NUM_THREADS=8 ./cnn-benchmarks.sh
#minibatch=16    iterations=2    epochs=3    device_id=-1     network_name=resnet        OMP_NUM_THREADS=16 ./cnn-benchmarks.sh
#minibatch=16    iterations=2    epochs=3    device_id=-1     network_name=resnet        OMP_NUM_THREADS=32 ./cnn-benchmarks.sh

minibatch=64    iterations=8    epochs=4     device_id=-1   network_name=ffn26752       OMP_NUM_THREADS=1  ./fc-benchmarks.sh
minibatch=64    iterations=8    epochs=4     device_id=-1   network_name=ffn26752       OMP_NUM_THREADS=2  ./fc-benchmarks.sh
minibatch=64    iterations=8    epochs=4     device_id=-1   network_name=ffn26752       OMP_NUM_THREADS=4  ./fc-benchmarks.sh
minibatch=64    iterations=8    epochs=4     device_id=-1   network_name=ffn26752       OMP_NUM_THREADS=8  ./fc-benchmarks.sh
#minibatch=64    iterations=8    epochs=4    device_id=-1     network_name=ffn26752 	OMP_NUM_THREADS=16 ./fc-benchmarks.sh
#minibatch=64    iterations=8    epochs=4    device_id=-1     network_name=ffn26752 	OMP_NUM_THREADS=32 ./fc-benchmarks.sh
#
minibatch=64    iterations=8    epochs=4     device_id=-1   network_name=ffn26752l6     OMP_NUM_THREADS=1  ./fc-benchmarks.sh
minibatch=64    iterations=8    epochs=4     device_id=-1   network_name=ffn26752l6     OMP_NUM_THREADS=2  ./fc-benchmarks.sh
minibatch=64    iterations=8    epochs=4     device_id=-1   network_name=ffn26752l6     OMP_NUM_THREADS=4  ./fc-benchmarks.sh
minibatch=64    iterations=8    epochs=4     device_id=-1   network_name=ffn26752l6     OMP_NUM_THREADS=8  ./fc-benchmarks.sh
#minibatch=64    iterations=8    epochs=4    device_id=-1     network_name=ffn26752l6 	OMP_NUM_THREADS=16 ./fc-benchmarks.sh
#minibatch=64    iterations=8    epochs=4    device_id=-1     network_name=ffn26752l6 	OMP_NUM_THREADS=32 ./fc-benchmarks.sh
#
