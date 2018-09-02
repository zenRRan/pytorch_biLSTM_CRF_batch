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
from Common import *
import torch.nn.functional as F
from torch.autograd import Variable

def log_sum_exp(scores, label_num):
    """
        params:
            scores: variable (batch_size, label_num, label_num)
            label_num:
        return:
            variable (batch_size, label_num)
        """
    batch_size = scores.size(0)
    max_scores, max_indexs = torch.max(scores, dim=1)
    # print('max_scores.size():', max_scores.size())
    max_scores_boardcast = max_scores.view(batch_size, label_num).unsqueeze(1).view(batch_size, 1, label_num).\
                            expand(batch_size, label_num, label_num)
    return max_scores.view(batch_size, label_num) + torch.log(torch.sum(torch.exp(scores - max_scores_boardcast), 1)).\
                                                    view(batch_size, label_num)

def log_sum_exp_low_dim(scores):
    """
            params:
                scores: variable (batch_size, label_num)
            return:
                variable (batch_size, label_num)
            """
    batch_size = scores.size(0)
    label_num = scores.size(1)
    max_score, max_indexs = torch.max(scores, dim=1)
    max_score_boardcast = max_score.unsqueeze(1).expand(batch_size, label_num)
    return max_score + torch.log(torch.sum(torch.exp(scores - max_score_boardcast), 1))

class CRF(nn.Module):
    def __init__(self, config, label_alpha):
        super(CRF, self).__init__()
        self.hidden_size =  config.hidden_size
        self.use_cuda = config.use_cuda
        self.label_num = label_alpha.m_size
        self.transition = torch.zeros(self.label_num, self.label_num)
        self.label_alpha = label_alpha
        if self.use_cuda:
            self.transition = self.transition.cuda()
        self.T = nn.Parameter(self.transition)

    def forward(self, emit_scores, labels):
        gold_scores = self.calc_sentences_scores(emit_scores, labels)
        encoder_scores = self.crf_encoder(emit_scores)
        # print('---')
        # print(gold_scores.size())
        # print(encoder_scores.size())
        return encoder_scores - gold_scores

    def calc_sentences_scores(self, emit_scores, labels):
        '''
        :param emit_scores: Variable(sent_len, batch_size, label_num)
        :param labels: Variable(batch_size, label_num)
        :return:
        '''
        sent_size = emit_scores.size(0)
        batch_size = emit_scores.size(1)

        labels = list(map(lambda t: [self.label_alpha.string2id[START]] + list(t), labels.data.tolist()))

        labels_group = [[label[id]*self.label_num + label[id+1] for id in range(sent_size)] for label in labels]
        labels_group = Variable(torch.LongTensor(labels_group))
        if self.use_cuda: labels_group = labels_group.cuda()
        # print('labels:', labels)

        # gold_begin_emits = torch.gather(Variable(torch.LongTensor(labels)), 1, Variable(torch.LongTensor(batch_size*[[0]])))
        # gold_begin_emits_broadcast = gold_begin_emits.view(1, batch_size, 1).expand(sent_size, batch_size, 1)
        # gold_begin_emits = torch.gather(emit_scores, 2, gold_begin_emits_broadcast)
        # begin_emit_scores = torch.sum(gold_begin_emits)

        batch_word_nums = batch_size * sent_size
        emit_scores_boardcast = emit_scores.view(batch_word_nums, -1).unsqueeze(1).view(batch_word_nums, 1, self.label_num)\
                                .expand(batch_word_nums, self.label_num, self.label_num)
        trans_scores_boardcast = self.T.unsqueeze(0).view(1, self.label_num, self.label_num).expand(batch_word_nums,\
                                                                                        self.label_num, self.label_num)
        scores = emit_scores_boardcast + trans_scores_boardcast

        labels_group = labels_group.transpose(0, 1).contiguous()
        # print(labels_group.size())
        # print('a.size():', a.size())
        score_total = torch.gather(scores.view(sent_size, batch_size, self.label_num, self.label_num)\
                                   .view(sent_size, batch_size, -1), 2,
                                   labels_group.view(sent_size, batch_size).unsqueeze(2).view(sent_size, batch_size, 1))
        # print('score_total:', score_total)
        batch_scores = torch.sum(score_total)
        # print('batch_scores:', batch_scores)

        return batch_scores
    def crf_encoder(self, emit_scores):
        """
           params:
               emit_scores: variable (seq_length, batch_size, label_nums)
           """
        sent_size = emit_scores.size(0)
        batch_size = emit_scores.size(1)
        # print('sent_size:',sent_size)
        # print('batch_size:', batch_size)
        forward_scores = emit_scores[0]
        # print('forward_scores.size():', forward_scores.size())

        for id in range(1, sent_size):
            emit_scores_boardcast = emit_scores[id].view(batch_size, self.label_num).unsqueeze(1)\
                                    .view(batch_size, 1, self.label_num).expand(batch_size, self.label_num, self.label_num)
            trans_scores_boardcast = self.T.view(self.label_num, self.label_num).unsqueeze(0)\
                                    .expand(batch_size, self.label_num, self.label_num)
            forward_scores_boardcast = forward_scores.view(batch_size, self.label_num).unsqueeze(2).expand(batch_size,
                                                                                                          self.label_num,
                                                                                                          self.label_num)
            scores = emit_scores_boardcast + trans_scores_boardcast + forward_scores_boardcast
            scores = log_sum_exp(scores, self.label_num)
            # print('forward_scores.size():', forward_scores_boardcast.size())
            forward_scores = scores
        total_score = log_sum_exp_low_dim(forward_scores)
        # print('total_score:', total_score.sum().size())
        return total_score.sum()

    def biterbi_decoder(self, emit_scores, masks):
        '''
        :param emit_scores: Variable(sent_size, batch, label_num)
        :return:
        '''
        sent_size = emit_scores.size(0)
        batch_size = emit_scores.size(1)

        ##prepare...
        emit_scores_broadcast = emit_scores[0].view(batch_size, self.label_num)
        trans_scores_broadcast = self.T[self.label_alpha.string2id[START], :].unsqueeze(0).expend(batch_size, self.label_num)
        forward_scores = emit_scores_broadcast + trans_scores_broadcast
        ### forward_scores: Variable(batch_size, label_num)

        ##calculate the back path
        back_path = []

        for idx in range(1, sent_size):
            emit_scores_broadcast = emit_scores[idx].view(batch_size, self.label_num).unsqueeze(1).view(batch_size, 1, self.label_num).expend(batch_size, self.label_num, self.label_num)
            trans_scores_broadcast = self.T.view(self.label_num, self.label_num).unsqueeze(0).view(1, self.label_num, self.label_num).expend(batch_size, self.label_num, self.label_num)
            forward_scores_broadcast = forward_scores.view(batch_size, self.label_num).unsqueeze(2).view(batch_size, self.label_num, 1).expend(batch_size, self.label_num, self.label_num).clone()
            scores = emit_scores_broadcast + trans_scores_broadcast + forward_scores
            ##scores:Variable(batch_size, self.label_num, self.label_num)

            max_scores, max_indexs = torch.max(scores, dim=1)

            ##max_indexs:Variable(batch_size, self.label_num)
            back_path.append(max_indexs.data.tolist())

        back_path.append([[self.label_alpha.string2id[PADDING]] * self.label_num for _ in range(batch_size)])
        back_path = Variable(torch.LongTensor(back_path)).transpose(0, 1)

        if self.use_cuda:
            back_path = back_path.cuda()

        ## calculate end transition scores
        end_trans_broadcast = self.T[:, self.label_alpha.string2id[PADDING]].unsqueeze(0).expend(batch_size, self.label_num)
        forward_scores_broadcast = forward_scores.view(batch_size, self.label_num)

        ends_scores = end_trans_broadcast + forward_scores_broadcast

        max_scores, max_ends_indexs = torch.max(ends_scores, dim=1)
        ## max_ends_indexs:Variable(batch_size)

        ##calculate predict path
        ends_max_indexs_broadcast = max_ends_indexs.unsqueeze(1).expend(batch_size, self.label_num)
        batch_length = torch.sum(masks, dim=0).long().unsqueeze(1)
        ends_position = batch_length.expend(batch_size, self.label_num)

        back_path.scatter_(1, ends_position.view(batch_size, self.label_num).unsqueeze(1), ends_max_indexs_broadcast.contiguous().view(batch_size, self.label_num).unsqueeze(1))
        ## batch_path: Variable(batch_size, seq_length, self.label_num)

        back_path = back_path.transpose(0, 1)
        ## batch_path: Variable(batch_size, seq_length, self.label_num)

        decode_path = Variable(torch.zeros(sent_size, batch_size))
        decode_path[-1] = max_ends_indexs

        for idx in range(sent_size-2, -1, -1):
            max_ends_indexs = torch.gather(back_path[idx], 1, max_ends_indexs.unsqueeze(1).view(batch_size, 1))
            decode_path[idx] = max_ends_indexs

        return decode_path






