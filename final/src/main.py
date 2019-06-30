import os
import sys
import argparse
import itertools
import tqdm
import csv
import json
import jieba
import numpy as np
import pandas as pd
from gensim.summarization import bm25
from sklearn.feature_extraction.text import TfidfVectorizer


parser = argparse.ArgumentParser()

parser.add_argument('--dict_file', default='./dict.txt.big')
parser.add_argument('--json_file', default='./idx2content.json')
parser.add_argument('--query_file', default='./news_data_1/QS_1.csv')
parser.add_argument('--TD_file', default='./news_data_1/TD.csv')
parser.add_argument('--output_file', default='./result.csv')

def load_data(raw_data):
    data = {}
    for news_index in tqdm.tqdm(raw_data):
        text = raw_data[news_index].replace('\n', '')
        text_list = list(jieba.cut(text, cut_all=False))
        data[news_index] = text_list
    return data

def tokenize(raw_data):
    data = []
    for text in tqdm.tqdm(raw_data):
        text = text.replace('\n', '')
        text_list = list(jieba.cut(text, cut_all=False))
        data.append(text_list)
    return data

def gen_output(result_list, output_file):
    for i, result in enumerate(result_list):
        index_list = []
        for index in result:
            index_list.append('news_{:0>6d}'.format(index + 1))
        result_list[i] = index_list

    with open(output_file, 'w') as f:
        w = csv.writer(f)
        title = ['Query_Index']
        for i in range(300):
            s = ('%03d') % (i + 1)
            title.append('Rank_' + str(s))
        w.writerow(title)
        for i, result in enumerate(result_list):
            w.writerow(['q_{:0>2d}'.format(i + 1)] + result)

def clean_text(query):
    remove_list = ['\n', '(', ')', '（', '）', '、', '。', '，', '！', '!', '「', '」', '《', '》', '『', '』', '？', '?', ':', '：']
    for char in remove_list:
        query = query.replace(char, '')
    return query

def get_index(result):
    index_list = []
    for news_index in result:
        index = int(news_index.split('_')[-1]) - 1
        index_list.append(index)
    return index_list

def get_TD_data(file_name):
    data_dict = {}
    data = pd.read_csv(file_name).values
    for row in data:
        query = row[0]
        if query not in data_dict:
            data_dict[query] = [[], [], [], []]
        news_index = row[1]
        rank = int(row[2])
        data_dict[query][3 - rank].append(news_index)

    for query in data_dict.keys():
        data_dict[query] = [list(itertools.chain.from_iterable(data_dict[query][:3])), data_dict[query][3]]
    return data_dict

def compute_similarity(query, key):
    query = list(jieba.cut(clean_text(query), cut_all=False))
    key = list(jieba.cut(clean_text(key), cut_all=False))

    similarity = len([x for x in key if x in query]) / len(query)

    if ('同意' in query and '反對' in key) or ('反對' in query and '同意' in key):
        opposite = True
    else:
        opposite = False
    return similarity, opposite

def main(json_file, query_file, TD_file, output_file):
    with open(json_file, 'r') as f:
        raw_data = json.load(f)

    data_list = []
    index_list = sorted(raw_data.keys())
    
    for news_index in index_list:
        news_data = raw_data[news_index]
        news_data = clean_text(news_data)
        data_list.append(news_data)

    print('Tokenize')

    data_list = tokenize(data_list)
    index_list = range(len(data_list))

    query_list = list((pd.read_csv(query_file).values)[:, 1])

    TD_dict = get_TD_data(TD_file)

    print('BM25')
    bm25_model = bm25.BM25(data_list)

    result_list = []
    for i, query in tqdm.tqdm(enumerate(query_list)):
        query = clean_text(query)
        query = list(jieba.cut(query, cut_all=False))
        scores = bm25_model.get_scores(query)
        rank = sorted(zip(index_list, scores), reverse=True, key=lambda x: x[1])
        rank, _ = map(list, zip(*rank))
        
        result = rank[:500]
        result_list.append(result)

    print('PRF')
    corpus = []
    for data in data_list:
        corpus.append(' '.join(data))

    del data_list
    vectorizer = TfidfVectorizer()
    doc_features = vectorizer.fit_transform(corpus)
    query_feature_list = []

    for i, query in enumerate(query_list):
        query = clean_text(query)
        query = ' '.join(list(jieba.cut(query, cut_all=False)))
        query_feature = np.array(vectorizer.transform([query]).toarray().tolist()).reshape(1, -1)
        query_feature_list.append(query_feature)

    max_iter = 3

    for iter_index in range(max_iter):
        print('Iteration {}'.format(iter_index + 1))
        for i, query_feature in tqdm.tqdm(enumerate(query_feature_list)):
            index_list = result_list[i]
            top_feature = np.zeros(query_feature.shape, dtype=np.float32)
            weight = 0.8
            for count, index in enumerate(index_list):
                if count >= 5:
                    break
                top_feature += weight * np.array(doc_features[index].toarray().tolist()).reshape(1, -1)
                weight -= 0.05
            final_feature = query_feature + top_feature
            score_list = []
            for index in index_list:
                current_feature = np.array(doc_features[index].toarray().tolist()).reshape(1, -1)
                s = np.linalg.norm(current_feature) * np.linalg.norm(final_feature)
                score = float(np.inner(current_feature, final_feature)) / s
                score_list.append(score)
            result, _ = map(list, zip(*sorted(zip(index_list, score_list), reverse=True, key=lambda k: k[1])))
            result_list[i] = result
    
    for i, result in enumerate(result_list):
        TD_assist = False
        for key in TD_dict:
            sim, opposite = compute_similarity(query_list[i], key)
            if sim >= 0.85:
                if opposite:
                    ref = []
                    exception = get_index(TD_dict[key][0])
                else:
                    ref = get_index(TD_dict[key][0])
                    exception = get_index(TD_dict[key][1])
                for index in result:
                    if len(ref) == 300:
                        break
                    if (index not in ref) and (index not in exception):
                        ref.append(index)
                TD_assist = True
                break
        if TD_assist:
            result_list[i] = ref[:300]
        else:
            result_list[i] = result[:300]
    gen_output(result_list, output_file)

if __name__ == '__main__':
    args = parser.parse_args()
    jieba.load_userdict(args.dict_file)
    main(args.json_file, args.query_file, args.TD_file, args.output_file)
