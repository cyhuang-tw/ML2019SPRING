import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
from itertools import groupby
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import jieba
from gensim import models
from gensim.models import word2vec
from util import read_file, collate_fn, TextDataset, TextNet

parser = argparse.ArgumentParser()

parser.add_argument('--dict_file', default='./dict.txt.big')
parser.add_argument('--data_file', default='./train_x.csv')
parser.add_argument('--label_file', default='./train_y.csv')
parser.add_argument('--max_length', default=100)
parser.add_argument('--w2v_model_file', default='./w2v.model')

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

def main(data_file, label_file, w2v_model_file, max_length):
    train_text, train_label, val_text, val_label = read_file(data_file, label_file)

    train_text = segment(train_text)
    train_text = [list(filter(str.strip, text_list)) for text_list in train_text] # Remove " " element

    val_text = segment(val_text)
    val_text = [list(filter(str.strip, text_list)) for text_list in val_text] # Remove " " element

    w2v_model = word2vec.Word2Vec.load(w2v_model_file)
    weights = torch.FloatTensor(w2v_model.wv.vectors)
    word_dict = {word : w2v_model.wv.vocab[word].index for word in w2v_model.wv.vocab}

    train_set = TextDataset(train_text, train_label, max_length)
    val_set = TextDataset(val_text, val_label, max_length)

    train_loader = DataLoader(train_set, batch_size=128, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_set, batch_size=128, shuffle=True, collate_fn=collate_fn)

    model = TextNet(weights, max_length).cuda()

    best_acc = 0.0
    num_epoch = 10

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss = nn.CrossEntropyLoss()

    for epoch in range(num_epoch):
        epoch_start_time = time.time()
        train_acc = 0.0
        train_loss = 0.0
        val_acc = 0.0
        val_loss = 0.0

        model.train()
        for i, batch in enumerate(train_loader):
            text, label, length, number = batch
            text = word2index(text, word_dict).cuda()
            length = length.cuda()
            label = label.cuda()

            optimizer.zero_grad()
            output = model(text, length)
            batch_loss = loss(output, label)

            batch_loss.backward()
            optimizer.step()

            train_acc += np.sum(np.argmax(output.cpu().data.numpy(), axis=1).reshape(-1, 1) == label.cpu().numpy().reshape(-1, 1))
            train_loss += batch_loss.item()

            torch.cuda.empty_cache()

            progress = ('=' * int(float(i)/len(train_loader)*40)).ljust(40)
            print ('[%03d/%03d] %2.2f sec(s) | %s |' % (epoch + 1, num_epoch, \
            (time.time() - epoch_start_time), progress), end='\r', flush=True)
        train_loss = train_loss / len(train_loader)
        train_acc = train_acc / len(train_set)

        model.eval()
        for i, batch in enumerate(val_loader):
            text, label, length, number = batch

            text = word2index(text, word_dict).cuda()
            length = length.cuda()
            label = label.cuda()

            output = model(text, length)
            batch_loss = loss(output, label)
            val_acc += np.sum(np.argmax(output.cpu().data.numpy(), axis=1).reshape(-1, 1) == label.cpu().numpy().reshape(-1, 1))
            val_loss += batch_loss.item()

            torch.cuda.empty_cache()

        val_loss = val_loss / len(val_loader)
        val_acc = val_acc / len(val_set)
        print('[%03d/%03d] %2.2f sec(s) Train Acc: %3.6f Loss: %3.6f | Val Acc: %3.6f Loss: %3.6f' % \
            (epoch + 1, num_epoch, time.time() - epoch_start_time, \
            train_acc, train_loss, val_acc, val_loss))

        if val_acc > best_acc:
            torch.save(model, 'model.pth')
            best_acc = val_acc
            print('Model Saved!')

if __name__ == '__main__':
    args = parser.parse_args()
    dict_file = args.dict_file
    data_file = args.data_file
    label_file = args.label_file
    max_length = args.max_length
    w2v_model_file = args.w2v_model_file
    jieba.load_userdict(dict_file)

    main(data_file, label_file, w2v_model_file, max_length)