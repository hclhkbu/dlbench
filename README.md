# dpBenchmark
A benchmark framework for measuring different deep learning tools

##TODO:
### Bugs and errors:
  1. Caffe won't take incoming parameters, they (learning rate, epoch size, number of epochs, etc) are pre-defined in the scripts.  
  2. CNTK didn't pass the test for rnn. Can run, can't parse the log
  3. Each tool script should be able to create its own runtime-need directory
  
### Optimization:
  1. Code redundancy
      Each xxxbm.py contains the same parameter parser, which is annoying when trying to modify the flags
      
