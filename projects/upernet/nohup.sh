#!/bin/bash
CUDA_VISIBLE_DEVICES=1 PORT=12445 ./projects/upernet/dist_train.sh projects/upernet/configs/upernet_transnext_tiny_512x512_160k_ade20k_ss.py 1\
 --resume-from work_dirs/upernet_transnext_tiny_512x512_160k_ade20k_ss/latest.pth &