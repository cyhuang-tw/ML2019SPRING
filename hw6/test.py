import os
import sys
import csv
import argparse
from itertools import groupby
import numpy as np
import torch
from torch.utils.data import DataLoader
import jieba
from gensim import models
from gensim.models import word2vec
from util import read_file, collate_fn, TextDataset, TextNet

parser = argparse.ArgumentParser()

parser.add_argument('--dict_file', default='./dict.txt.big')
parser.add_argument('--model_file', default=['./model.pth'], nargs='+')
parser.add_argument('--w2v_model_file', default='./w2v.model')
parser.add_argument('--test_file', default='./test_x.csv')
parser.add_argument('--output_file', default='./result.csv')
parser.add_argument('--max_length', default=100)

def gen_file(y_test, output_file):
    f = open(output_file, "w")
    w = csv.writer(f)
    title = ['id', 'label']
    w.writerow(title) 
    for i in range(y_test.shape[0]):
        content = [str(i), str(y_test[i])[1]]
        w.writerow(content)
    f.close()

def segment(text_list):
    seg_list = []
    for i, text in enumerate(text_list):
        seg_text = jieba.lcut(text, cut_all=False)
        seg_text = [word for word, count in groupby(seg_text)]
        seg_list.append(seg_text)
    return seg_list

def word2index(text_list, word_dict):
    index_tensor = torch.zeros(len(text_list), len(text_list[0]), dtype=torch.long)
    for i, text in enumerate(text_list):
        index = [word_dict[x] if x in word_dict else word_dict['<UNK>'] for x in text]
        index_tensor[i, :] = torch.tensor(index)
    return index_tensor

def main(model_file, w2v_model_file, test_file, output_file, max_length):
    test_text = read_file(test_file)
    test_text = segment(test_text)
    test_text = [list(filter(str.strip, text_list)) for text_list in test_text] # Remove " " element

    w2v_model = word2vec.Word2Vec.load(w2v_model_file)
    weights = torch.FloatTensor(w2v_model.wv.vectors)
    word_dict = {word : w2v_model.wv.vocab[word].index for word in w2v_model.wv.vocab}

    test_set = TextDataset(test_text, np.arange(len(test_text)), max_length) # random sequence for label parameter
    test_loader = DataLoader(test_set, batch_size=128, shuffle=False, collate_fn=collate_fn)

    model_list = []

    for file in model_file:
        model = torch.load(file)
        model.eval()
        model_list.append(model.cuda())

    ans = np.zeros((0, 1),dtype=np.int)

    for i, batch in enumerate(test_loader):
        text, _, length, number = batch
        text = word2index(text, word_dict).cuda()
        length = length.cuda()

        output = torch.zeros(len(number), 2).cuda()
        
        for i in range(len(model_list)):
            output += model_list[i](text, length)

        label = np.argmax(output.cpu().data.numpy(), axis=1)
        _, label = map(list, zip(*sorted(zip(number, label.tolist()))))
        label = np.array(label).reshape(-1, 1)
        ans = np.concatenate((ans, label), axis=0)
        torch.cuda.empty_cache()
    gen_file(ans, output_file)


if __name__ == '__main__':
    args = parser.parse_args()
    dict_file = args.dict_file
    model_file = args.model_file
    w2v_model_file = args.w2v_model_file
    test_file = args.test_file
    output_file = args.output_file
    max_length = args.max_length
    jieba.load_userdict(dict_file)
    main(model_file, w2v_model_file, test_file, output_file, max_length)
