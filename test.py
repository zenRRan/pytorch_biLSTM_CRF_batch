#!/usr/bin/env python
# encoding: utf-8

"""
@version: python3.6
@author: 'zenRRan'
@license: Apache Licence 
@contact: zenrran@qq.com
@software: PyCharm
@file: test.py
@time: 2018/4/16 15:48
"""

import numpy as np
import torch

a = torch.FloatTensor([[[1,1,3]], [[1,2,0]]])
b = [1, 2, 3, 4]
# print(a[1, :])
print(a.size())
print(torch.max(a, dim=2))






