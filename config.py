#!/usr/bin/env python
# encoding: utf-8

"""
@version: python3.6
@author: 'zenRRan'
@license: Apache Licence 
@contact: zenrran@qq.com
@software: PyCharm
@file: config.py
@time: 2018/4/16 10:53
"""

from configparser import ConfigParser

class Configer:
    def __init__(self, path):
        self.config = ConfigParser()
        self.config.read(path)


    @property
    def train_path(self):
        return self.config.get('path', 'train_path')

    @property
    def test_path(self):
        return self.config.get('path', 'test_path')

    @property
    def max_len(self):
        return self.config.getint('data', 'max_len')

    @property
    def learning_rate(self):
        return self.config.getfloat('data', 'learning_rate')

    @property
    def adam(self):
        return self.config.getboolean('data', 'adam')

    @property
    def SGD(self):
        return self.config.getboolean('data', 'SGD')

    @property
    def train_batch_size(self):
        return self.config.getint('data', 'train_batch_size')

    @property
    def test_batch_size(self):
        return self.config.getint('data', 'test_batch_size')

    @property
    def weight_decay(self):
        return self.config.getfloat('data', 'weight_decay')
