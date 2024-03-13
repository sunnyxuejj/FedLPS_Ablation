#!/bin/bash
cd ../../

python RCR.py --dataset 'mnist' \
--nclass 2 \
--nsamples 100 \
--nusers 100 \
--frac 0.1 \
--local_ep 5 \
--local_bs 32 \
--lr 0.1 \
--model_split_mode 'dynamic'
