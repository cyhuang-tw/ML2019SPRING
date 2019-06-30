#!/bin/bash

python3 gen_json.py --json_file=$2 --NC_file=$4
python3 main.py --dict_file=$1 --TD_file=$3 --query_file=$5 --output_file=$6
