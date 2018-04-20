#!/usr/bin/env python
# encoding: utf-8

"""
@version: python3.6
@author: 'zenRRan'
@license: Apache Licence 
@contact: zenrran@qq.com
@software: PyCharm
@file: CRF.py
@time: 2018/4/18 14:52
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class CRF(nn.Module):
    def __init__(self, config, label_alpha):
        super(CRF, self).__init__()
        self.hidden_size =  config.hidden_size
        self.use_cuda = config.use_cuda
        self.label_num = label_alpha.m_size
        self.transition = torch.zeros(self.label_num, self.label_num)
        if self.use_cuda:
            self.transition = self.transition.cuda()
        self.T = nn.Parameter(self.transition)

    def forward(self, emit_scores, labels):
        gold_scores = self.calc_sentences_scores(emit_scores, labels)
        encoder_scores = self.crf_encoder(emit_scores)
        return encoder_scores - gold_scores

    def calc_sentences_scores(self, emit_scores, labels):
        '''
        :param emit_scores: Variable(sent_len, batch_size, label_num)
        :param labels: Variable(batch_size, label_num)
        :return:
        '''
        sent_size = emit_scores.size(0)
        batch_size = emit_scores.size(1)

        labels_group = [[label[id]*self.label_num + label[id+1] for id in range(sent_size-1)] for label in labels.data.tolist()]
        labels_group = Variable(torch.LongTensor(labels_group))
        if self.use_cuda: labels_group = labels_group.cuda()
        # print('labels:', labels)

        gold_begin_emits = torch.gather(labels, 1, Variable(torch.LongTensor(batch_size*[[0]])))
        gold_begin_emits_broadcast = gold_begin_emits.view(1, batch_size, 1).expand(sent_size, batch_size, 1)
        gold_begin_emits = torch.gather(emit_scores, 2, gold_begin_emits_broadcast)
        begin_emit_scores = torch.sum(gold_begin_emits)

        batch_word_nums = batch_size * sent_size
        emit_scores_boardcast = emit_scores.view(batch_word_nums, -1).unsqueeze(1).view(batch_word_nums, 1, self.label_num)\
                                .expand(batch_word_nums, self.label_num, self.label_num)
        trans_scores_boardcast = self.T.unsqueeze(0).view(1, self.label_num, self.label_num).expand(batch_word_nums,\
                                                                                        self.label_num, self.label_num)
        scores = emit_scores_boardcast + trans_scores_boardcast

        labels_group = labels_group.transpose(0,1).contiguous()
        print(labels_group.size())
        a = scores.view(sent_size, batch_size, self.label_num, self.label_num)\
                                   .view(sent_size, batch_size, -1)
        print('a.size():', a.size())
        b = labels_group.view(sent_size, batch_size).unsqueeze(2).view(sent_size,batch_size,1)
        score_total = torch.gather(a, 2, b)
        print(score_total.size())

        input()
        # return batch_scores + begin_emit_scores
    def crf_encoder(self, emit_scores):
        pass





