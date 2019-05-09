import os
import sys
import argparse
import jieba
from itertools import groupby
import numpy as np
from gensim import models
from gensim.models import word2vec
from util import read_file

parser = argparse.ArgumentParser()

parser.add_argument('--dict_file', default='./dict.txt.big')
parser.add_argument('--input_file', default=['./train_x.csv'], nargs='+')
parser.add_argument('--output_file', default='./w2v.model')



def segment(text_list):
    seg_list = []
    for i, text in enumerate(text_list):
        seg_text = jieba.lcut(text, cut_all=False)
        seg_text = [word for word, count in groupby(seg_text)]
        seg_list.append(seg_text)
    return seg_list

def main(input_file, output_file):
    text_list = []

    for text_file in input_file:
        tmp_text = read_file(text_file)
        text = segment(tmp_text)
        text_list += text

    text_list = text_list + [['<UNK>'] * 5] + [['<PAD>'] * 5]

    text_list = [list(filter(str.strip, text)) for text in text_list]

    model = word2vec.Word2Vec(text_list, size=256, workers=8, min_count=5, iter=10)

    model.save(output_file)

if __name__ == '__main__':
    args = parser.parse_args()
    dict_file = args.dict_file
    input_file = args.input_file
    output_file = args.output_file
    jieba.load_userdict(dict_file)
    main(input_file, output_file)