import os
import sys
import argparse
import json
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()

parser.add_argument('--NC_file', default='./news_data_1/NC_1.csv')
parser.add_argument('--json_file', default='./url2content.json')
parser.add_argument('--output_dir', default='./')

def main(NC_file, json_file, output_dir):
    NC_data = pd.read_csv(NC_file).values
    with open(json_file, 'r') as f:
        url_dict = json.load(f)
    idx_dict = {}
    for row in NC_data:
        news_index = row[0]
        news_url = row[1]
        idx_dict[news_index] = url_dict[news_url]

    with open(os.path.join(output_dir, 'idx2content.json'), 'w') as f:
        json.dump(idx_dict, f)

if __name__ == '__main__':
    args = parser.parse_args()
    main(args.NC_file, args.json_file, args.output_dir)