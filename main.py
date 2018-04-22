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
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from model import biRNN
from CRF import CRF
from Common import *
import numpy as np
import time
import math
import numpy as np
import random
torch.manual_seed(23)
np.random.seed(23)
random.seed(23)

def to_scalar(vec):
    return vec.view(-1).data.tolist()[0]
def as_minutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def time_since(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (as_minutes(s), as_minutes(rs))

if __name__ == '__main__':
    start_time = time.time()
    print('begin time:', as_minutes(time.time()-start_time))
    config_path = './config.cfg'
    config = Configer(config_path)
    print('reading ',config.train_path)
    train_reader = Reader(config.train_path, max_len=config.max_len)
    print('done, using time:', as_minutes(time.time()-start_time))

    print('reading ', config.test_path)
    test_reader = Reader(config.test_path, max_len=config.max_len)
    print('done, using time:', as_minutes(time.time()-start_time))
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
    text_dic[PADDING] = COUNTS
    text_dic[UNKNOWN] = COUNTS
    label_dic[START] = COUNTS
    text_alpha.initial(text_dic)
    label_alpha.initial(label_dic)
    print('text_alpha: ', text_alpha.m_size)
    print('label alpha: ', label_alpha.m_size)
    print('done, using time:', as_minutes(time.time()-start_time))

    '''
        obtain the ids seqs
    '''
    print('seq2id...')
    label_id_list = seq2id(train_labels, label_alpha)
    text_id_list = seq2id(train_texts, text_alpha)
    test_label_id_list = seq2id(test_labels, label_alpha)
    test_text_id_list = seq2id(test_texts, text_alpha)
    print('done, using time:', time.time()-start_time)
    '''
        init model
    '''
    print('init model...')
    model = biRNN(config, text_alpha.m_size, label_alpha.m_size)
    crf = CRF(config, label_alpha)
    print('done, using time:', as_minutes(time.time()-start_time))

    '''
        init optimizer
    '''
    print('init optimizer...')
    optimizer = None
    print('parameters:')
    if config.adam:
        print('optimizer:Adam')
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    elif config.SGD:
        print('optimizer:SGD')
        optimizer = optim.SGD(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    print('done, using time:', as_minutes(time.time()-start_time))

    # print(text_id_list[:10])
    print('lr:', config.learning_rate)
    print('weight_decay:',config.weight_decay)
    print('train_batch_size:', config.train_batch_size)
    print('test_batch_size:', config.test_batch_size)

    print('create train and test data...')
    print(text_id_list)
    train_data, sent_num, batch_num = create_training_data(text_id_list, label_id_list, config)
    test_data, test_sent_num, test_batch_num = create_training_data(test_text_id_list, test_label_id_list, config)
    print('batch size:', batch_num)
    print('done, using time:', as_minutes(time.time()-start_time))

    # print(train_data[:10])
    # print('-----------------------')
    # print(model)

    '''
        begining to train...
    '''
    print('begin training in ', as_minutes(time.time()-start_time))
    for Epoch in range(config.step):
        epoch_loss = 0
        for batch in create_batch(train_data, config.train_batch_size):
            model.train()
            model.zero_grad()
            optimizer.zero_grad()

            train_texts, train_labels = prepared_one_batch(batch)
            train_texts = Variable(torch.LongTensor(train_texts))
            train_labels = Variable(torch.LongTensor(train_labels))
            emit_scores = model(train_texts)
            loss = crf(emit_scores, train_labels)

            loss.backward()
            optimizer.step()
            # print(loss)
            epoch_loss += to_scalar(loss)
            # input()
        # print(sent_num)
        if Epoch == 0:
            continue
        print('Epoch is {}, average loss is {} ({})'.format(Epoch, (epoch_loss / sent_num),
                                                              time_since(start_time, Epoch/config.step)))
        print('Test...')

    def eval(texts, labels):
        model.eval()
        model.zero_grad()
        for batch in create_batch(test_data, config.test_batch_size):
            test_texts, test_labels = prepared_one_batch(batch)
            test_texts = Variable(torch.LongTensor(test_texts))
            test_labels = Variable(torch.LongTensor(test_labels))
            emit_scores = model(test_texts)
            prediction = ''














