#!/bin/bash
cd ../../

python diff_Spa.py --dataset 'reddit' \
--nusers 100 \
--frac 0.1 \
--local_ep 1 \
--local_bs 5 \
--momentum 0.5 \
--clip 1 \
--lr 8 \
--model_split_mode 'fix' \
--mask_random True \
--mask_rate 0.4
