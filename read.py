#!/usr/bin/env python
# encoding: utf-8

"""
@version: python3.6
@author: 'zenRRan'
@license: Apache Licence 
@contact: zenrran@qq.com
@software: PyCharm
@file: read.py
@time: 2018/4/16 10:54
"""

class Reader:
    def __init__(self, path, max_len=-1):
        self.text = []
        self.label = []
        with open(path, 'r', encoding='utf8') as f:
            lines = f.readlines()
            text_inst = []
            label_inst = []
            for line in lines:
                line = line.strip().split()
                if line == '' or len(line) == 0:
                    if len(text_inst) < max_len or max_len == -1:
                        self.text.append(text_inst)
                        self.label.append(label_inst)
                        text_inst = []
                        label_inst = []
                    else:
                        text_inst = []
                        label_inst = []
                    continue
                text_inst.append(line[0])
                label_inst.append(line[2])

    def getData(self):
        return self.text, self.label