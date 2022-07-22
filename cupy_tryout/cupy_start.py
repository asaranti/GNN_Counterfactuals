"""
    CuPy test

    :author: Anna Saranti
    :copyright: © 2021 HCI-KDD (ex-AI) group
    :date: 2022-07-14
"""

import os

import cupy as cp

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = 'cuda:0'

x = cp.arange(6).reshape(2, 3).astype('f')
print(x)

