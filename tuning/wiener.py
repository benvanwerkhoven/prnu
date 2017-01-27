#!/usr/bin/env python
from __future__ import print_function

from collections import OrderedDict
import numpy as np
from kernel_tuner import tune_kernel, run_kernel
from scipy.misc import imread

from context import get_kernel_path, get_testdata_path



def tune_wiener():

    with open(get_kernel_path()+'wienerfilter.cu', 'r') as f:
        kernel_string = f.read()

    image = imread(get_testdata_path() + "test.jpg", mode="F")

    height = np.int32(image.shape[0])
    width = np.int32(image.shape[1])
    problem_size = (width, height)

    output = np.zeros(problem_size, dtype=np.float32)

    args = [height, width, output, image]

    tune_params = OrderedDict()
    tune_params["block_size_x"] = [32*i for i in range(1,33)]
    tune_params["block_size_y"] = [2**i for i in range(6)]

    #first the naive kernel
    tune_kernel("computeVarianceEstimates", kernel_string, problem_size, args, tune_params, grid_div_y=["block_size_y"])

    #more sophisticated kernel
    tune_params["reuse_computation"] = [0,1]
    tune_kernel("computeVarianceEstimates_shared", kernel_string, problem_size, args, tune_params, grid_div_y=["block_size_y"])


def tune_variance_zero_mean():

    with open(get_kernel_path()+'wienerfilter.cu', 'r') as f:
        kernel_string = f.read()

    image = imread(get_testdata_path() + "test.jpg", mode="F")

    height = np.int32(image.shape[0])
    width = np.int32(image.shape[1])
    size = np.int32(height*width)

    tune_params = OrderedDict()
    tune_params["block_size_x"] = [2**i for i in range(5,11)]
    tune_params["num_blocks"] = [2**i for i in range(5,11)]

    max_blocks = max(tune_params["num_blocks"])
    output = np.zeros(max_blocks, dtype=np.float32)

    args = [size, output, image]
    problem_size = ("num_blocks", 1)

    tune_kernel("computeVarianceZeroMean", kernel_string, problem_size, args, tune_params, grid_div_x=[], verbose=True)


if __name__ == "__main__":
    tune_wiener()
    tune_variance_zero_mean()
