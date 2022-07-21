"""
    CuPy example

    :author: Anna Saranti
    :copyright: © 2021 HCI-KDD (ex-AI) group
    :date: 2022-04-27
"""

import numpy as np
import cupy as cp


x_cpu = np.array([1, 2, 3])
l2_cpu = np.linalg.norm(x_cpu)

x_gpu = cp.array([1, 2, 3])
l2_gpu = cp.linalg.norm(x_gpu)


