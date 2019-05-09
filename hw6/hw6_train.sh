#!/bin/bash

python3 train_w2v.py --input_file $1 $3 --dict_file=$4
python3 train.py --data_file=$1 --label_file=$2 --dict_file=$4
