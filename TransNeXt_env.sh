#!/bin/bash

conda create --name TransNeXt python=3.8 -y

conda activate TransNeXt

pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118

pip install --upgrade pip

pip install timm==0.5.4
# 1.
# conda install -c conda-forge gcc=11.1.0
# conda install -c conda-forge gxx=11.1.0
# pip install ninja
# cp /usr/include/crypt.h ~/anaconda3/envs/TransNeXt/include/python3.8/
# 这句放shell里面,设置环境变量CFLAGS，将其传递给编译器，给连接器参数sysroot，用于指定系统根目录，用于隔离编译环境（所以上一句可以不用吗？）
# export CFLAGS='-Wl,--sysroot=/media/ssd_2t/home/jzy/anaconda3/envs/TransNeXt/x86_64-conda-linux-gnu/sysroot  
# pip install mmcv-full==1.7.1
# 编译时注释掉cpp_extension.py中的警告行貌似可行？（重要提示）641的CUDA：12.2,然后和运行时11.8配合不了？
# 一种解决方式是在用户目录安装cuda11.8，然后我按照上面的思路来安装的mmcv-full
# 2. 注释掉cpp_extension.py代码警告部分？可行？等待验证-- 能编译通过，有没有问题我就不知道了，先用着
pip install mmcv-full==1.7.1
pip install -r requirements.txt
# pip install -U openmim
# mim install mmcv==1.7.1
pip install mmsegmentation==0.30.0
pip install yapf==0.40.1
# pip install IPython
# pip install cityscapesscripts