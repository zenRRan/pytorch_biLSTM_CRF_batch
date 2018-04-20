#!/usr/bin/env python
# encoding: utf-8

"""
@version: python3.6
@author: 'zenRRan'
@license: Apache Licence 
@contact: zenrran@qq.com
@software: PyCharm
@file: model.py
@time: 2018/4/16 10:54
"""

import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn

class biRNN(nn.Module):
    def __init__(self, config, text_vocab_size, label_size):
        super(biRNN, self).__init__()
        self.embed_dim = config.embed_dim
        self.hidden_size = config.hidden_size
        self.hidden_layer = config.hidden_layer
        self.vocab_size = text_vocab_size
        self.label_size = label_size
        self.use_cuda = config.use_cuda
        self.embedding = nn.Embedding(self.vocab_size, self.embed_dim)
        if config.GRU:
            self.rnn = nn.GRU(self.embed_dim,
                              hidden_size=self.hidden_size // 2,
                              num_layers=self.hidden_layer,
                              dropout=config.dropout,
                              bidirectional=config.bidirectional)
        elif config.LSTM:
            self.rnn = nn.LSTM(self.embed_dim,
                               hidden_size=self.hidden_size // 2,
                               num_layers=self.hidden_layer,
                               dropout=config.dropout,
                               bidirectional=config.bidirectional)
        self.dropout = nn.Dropout(config.dropout)
        self.linear = nn.Linear(self.hidden_size, label_size, bias=True)

    def forward(self, input):
        batch_size = input.size(0)
        # print('batch_size:', batch_size)
        input = self.embedding(input)    #torch.Size([batch_size, sent_len, 64])
        # print(input.size())
        input = self.dropout(input)      #torch.Size([batch_size, sent_len, 64])
        # print(input.size())
        input = torch.transpose(input, 0, 1)
        init_hidden = self.init_hidden(batch_size)
        output, hidden = self.rnn(input)    #torch.Size([sent_len, batch_size, 64])
        # print('#', output.size())
        # output = output.squeeze(0)          #torch.Size([sent_len, batch_size, 64])
        # output = torch.transpose(output, 0, 1)
        # output = F.relu(output)
        output = self.linear(output)
        # print('##', output.size())
        return output

    def init_hidden(self, batch):
        if self.use_cuda:
            return (Variable(torch.randn(2, batch, self.hidden_size // 2)).cuda(),
                    Variable(torch.randn(2, batch, self.hidden_size // 2)).cuda())
        else:
            return (Variable(torch.randn(2, batch, self.hidden_size // 2)),
                    Variable(torch.randn(2, batch, self.hidden_size // 2)))


















