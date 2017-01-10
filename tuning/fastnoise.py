#!/usr/bin/env python
from __future__ import print_function

from collections import OrderedDict
import numpy as np
from kernel_tuner import tune_kernel, run_kernel
from scipy.misc import imread

from context import get_kernel_path, get_testdata_path



def tune_fastnoise():

    with open(get_kernel_path()+'fastnoisefilter.cu', 'r') as f:
        kernel_string = f.read()

    image = imread(get_testdata_path() + "test.jpg", mode="F")

    problem_size = image.shape
    height = np.int32(image.shape[0])
    width = np.int32(image.shape[1])

    output = np.zeros(problem_size, dtype=np.float32)

    args = [height, width, output, image]

    tune_params = OrderedDict()
    tune_params["block_size_x"] = [32*i for i in range(1,33)]
    tune_params["block_size_y"] = [2**i for i in range(6)]

    kernels = ["normalized_gradient", "gradient",
               "convolveHorizontally", "convolveVertically", "normalize"]
    for k in kernels:
        tune_kernel(k, kernel_string, problem_size, args, tune_params)





if __name__ == "__main__":
    tune_fastnoise()
