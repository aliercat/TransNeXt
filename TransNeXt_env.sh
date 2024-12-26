#!/bin/bash

conda create --name TransNeXt python=3.8 -y

conda activate TransNeXt

pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118

pip install --upgrade pip

pip install timm==0.5.4
pip install mmcv-full==1.7.1
pip install mmsegmentation==0.30.0

# pip install IPython
# pip install cityscapesscripts