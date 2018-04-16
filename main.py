#!/usr/bin/env python
# encoding: utf-8

"""
@version: python3.6
@author: 'zenRRan'
@license: Apache Licence 
@contact: zenrran@qq.com
@software: PyCharm
@file: main.py
@time: 2018/4/16 10:53
"""


from read import Reader
from config import Configer
from Alphabet import Alphabet
from collections import OrderedDict
import torch.nn.functional as F
import torch.optim as optim
from train import train
from model import biRNN
from Common import *
import numpy as np

if __name__ == '__main__':
    config_path = './config.cfg'
    config = Configer(config_path)
    print('reading ',config.train_path)
    train_reader = Reader(config.train_path, max_len=config.max_len)
    print('done')
    print('reading ', config.test_path)
    test_reader = Reader(config.test_path, max_len=config.max_len)
    print('done')
    train_texts, train_labels = train_reader.getData()
    test_texts, test_labels = test_reader.getData()
    print('train:', len(train_texts))
    print('test:', len(test_texts))
    # print(train_texts[:10])
    # print(train_labels[:10])

    text_alpha = Alphabet()
    label_alpha = Alphabet()
    text_dic = OrderedDict()
    label_dic = OrderedDict()

    print('create Alphabet...')
    for (text, label) in zip(train_texts, train_labels):
        for l in label:
            if l in label_dic:
                label_dic[l] += 1
            else:
                label_dic[l] = 1
        for word in text:
            if word in text_dic:
                text_dic[word] += 1
            else:
                text_dic[word] = 1
    print('done')
    text_dic[PADDING] = COUNTS
    text_dic[UNKNOWN] = COUNTS
    text_alpha.initial(text_dic)
    label_alpha.initial(label_dic)
    print('text_alpha: ', text_alpha.m_size)
    print('label alpha: ', label_alpha.m_size)
    '''
        obtain the ids seqs
    '''
    label_id_list = seq2id(train_labels, label_alpha)
    text_id_list = seq2id(train_texts, text_alpha)

    model = biRNN(config)
    if config.adam:
        optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    elif config.SGD:
        optim.SGD(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    train(model, )














