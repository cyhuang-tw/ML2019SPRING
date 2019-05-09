#!/bin/bash

wget -O model_1.pth https://www.dropbox.com/s/te2lqlup66qxo8l/model_1.pth?dl=1
wget -O model_2.pth https://www.dropbox.com/s/a51w2iv00n3gmdf/model_2.pth?dl=1
wget -O model_3.pth https://www.dropbox.com/s/fiojvpz3uqrqgxh/model_3.pth?dl=1
python3 test.py --test_file=$1 --dict_file=$2 --output_file=$3 --model_file model_1.pth model_2.pth model_3.pth
