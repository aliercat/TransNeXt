#!/bin/bash
CUDA_VISIBLE_DEVICES=1 PORT=12345 ./projects/upernet/dist_train.sh ./configs/upernet_transnext_small_512x512_160k_ade20k_ms.py 1 &