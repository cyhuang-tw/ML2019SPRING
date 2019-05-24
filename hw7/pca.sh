#!/bin/bash

wget -O mean_face.npy https://www.dropbox.com/s/6zzts0r6jfs3ip4/mean_face.npy?dl=1
wget -O eigen_face.npy https://www.dropbox.com/s/jibwlyogg9dcdxw/eigen_face.npy?dl=1
python3 pca.py --pre_computed --input_dir=$1 --input_img=$2 --output_img=$3
