#!/usr/bin/env python
# encoding: utf-8

"""
@version: python3.6
@author: 'zenRRan'
@license: Apache Licence 
@contact: zenrran@qq.com
@software: PyCharm
@file: Common.py
@time: 2018/4/4 9:15
"""

import numpy as np
import random

random.seed(23)

PADDING = 'PADDING'
UNKNOWN = 'UNKNOWN'
START = '<start>'
COUNTS = 10

def seq2id(seqs, word_alpha, padding=False):
    id_list = []
    max_len = 0

    if padding:
        for seq in seqs:
            max_len = max(max_len, len(seq))
    for seq in seqs:
        id = []
        for word in seq:
            degit = word_alpha.from_string(word)
            if degit >= 0:
                id.append(degit)
            else:
                id.append(UNKNOWN)
        if padding:
            for _ in range(max_len - len(seq)):
                id.append(PADDING)
        id_list.append(id)
    return id_list

def label2id(labels, label_alpha):
    id_list = []
    for label in labels:
        id = label_alpha.from_string(label)
        if id != -1:
            id_list.append(id)
        else:
            print("Wrong: label2id id = -1!")
            return []

    return id_list

def create_batch(data, batch_size):
    data_size = len(data)
    # print(data[:10])
    # data_ids = sorted(range(data_size), key=lambda src_id: len(data[src_id][0]))
    # data = [data[src_id] for src_id in data_ids]

    new_data = []
    for each_data in data:
        new_data.extend(each_data)
    data = new_data

    batched_data = []
    instances = []
    last_length = 0
    for instance in data:
        cur_length = len(instance[0])
        if cur_length > 0 and cur_length != last_length and len(instances) > 0:
            batched_data.append(instances)
            instances = []
        instances.append(instance)
        last_length = cur_length
        if len(instances) == batch_size:
            batched_data.append(instances)
            instances = []
    if len(instances) > 0:
        batched_data.append(instances)
    random.shuffle(batched_data)
    for batch in batched_data:
        yield batch

def create_training_data(texts, labels, config):
    max_length = 0
    if config.max_len == -1:
        for text in texts:
            max_length = max(max_length, len(text))
    else:
        max_length = config.max_len
    train_data = []
    sent_num = 0
    for _ in range(max_length):
        train_data.append([])
    for (text, label) in zip(texts, labels):
        sent_num += 1
        ids = len(text) - 1
        train_data[ids].append((text, label))
    data_size = len(texts)
    batch_num = 0
    for idx in range(len(train_data)):
        train_size = len(train_data[idx])
        batch_num += int(np.ceil(train_size / config.train_batch_size))
    return train_data, sent_num, batch_num

def prepared_one_batch(batch):
    train_texts = []
    train_labels = []
    # print(batch)
    for idx, instance in enumerate(batch):
        # print(idx)
        # print(instance)
        train_texts.append(instance[0])
        train_labels.append(instance[1])
    return train_texts, train_labels









































