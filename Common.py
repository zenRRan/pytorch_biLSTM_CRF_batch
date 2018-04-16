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

PADDING = 'PADDING'
UNKNOWN = 'UNKNOWN'
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
    data_ids = sorted(range(data_size), key=lambda src_id: len(data[src_id]))
    data = [data[src_id] for src_id in data_ids]

    batched_data = []
    instances = []
    last_length = 0
    for instance in data:
        cur_length = len(instance)
        if cur_length > 0 and cur_length != last_length and len(instance) > 0:
            batched_data.append(instances)
        instances.append(instance)
        last_length = cur_length
        if len(instances) > batch_size:
            batched_data.append(instances)
            instances = []
    if len(instances) > 0:
        batched_data.append(instances)

    for batch in batched_data:
        yield batch

def create_training_data(texts, labels, config):
    max_length = 0
    if config.max_length == -1:
        for text in texts:
            max_length = max(max_length, len(text))
    else:
        max_length = config.max_length
    train_data = []
    for _ in range(max_length):
        train_data.append([])
    for (text, label) in zip(texts, labels):
        ids = len(text)
        train_data[ids].append((text, label))
    data_size = len(texts)
    batch_num = 0
    for idx in range(train_data):
        train_size = len(train_data[idx])
        batch_num += int(np.ceil(train_size / config.train_batch_size))
    return train_data, batch_num











































