CUDA_VISIBLE_DEVICES=3 th recurrent-language-model.lua --progress --cuda --lstm --seqlen 32 --hiddensize '{256,256}' --batchsize 64 --startlr 0.1 --minlr 0.1 --maxepoch 40
