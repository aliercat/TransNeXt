#!/bin/bash
CUDA_VISIBLE_DEVICES=1 projects/upernet/dist_train.sh projects/upernet/configs/upernet_transnext_tiny_512x512_160k_ade20k_ss.py 1 &