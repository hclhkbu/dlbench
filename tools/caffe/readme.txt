caffe time -model=*.prototxt -gpu=0

nvprof --log-file profile3_gputrace.log --print-gpu-trace caffe time -model=resnet50.prototxt -gpu=0 -iterations=3
