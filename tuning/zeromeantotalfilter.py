#!/usr/bin/env python
from __future__ import print_function

from collections import OrderedDict
import numpy as np
from kernel_tuner import tune_kernel, run_kernel
from scipy.misc import imread

from context import get_kernel_path, get_testdata_path


def tune_zeromean():

    with open(get_kernel_path()+'zeromeantotalfilter.cu', 'r') as f:
        kernel_string = f.read()

    kernel_name = "computeMeanVertically"

    image = imread(get_testdata_path() + "test.jpg", mode="F")
    height = np.int32(image.shape[0])
    width = np.int32(image.shape[1])

    #only one row of thread-blocks is to be created
    problem_size = (width, 1)

    args = [height, width, image]

    tune_params = OrderedDict()
    tune_params["block_size_x"] = [32*i for i in range(1,9)]
    tune_params["block_size_y"] = [2**i for i in range(6)]

    grid_div_x = ["block_size_x"]
    grid_div_y = None

    tune_kernel("computeMeanVertically", kernel_string,
        problem_size, args, tune_params,
        grid_div_y=grid_div_y, grid_div_x=grid_div_x)



if __name__ == "__main__":
    tune_zeromean()
