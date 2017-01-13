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

    image = imread(get_testdata_path() + "test.jpg", mode="F")
    height = np.int32(image.shape[0])
    width = np.int32(image.shape[1])

    #tune_vertical(kernel_string, image, height, width)
    #tune_horizontal(kernel_string, image, height, width)
    tune_transpose(kernel_string, image, height, width)



def tune_vertical(kernel_string, image, height, width):
    args = [height, width, image]

    #only one row of thread-blocks is to be created
    problem_size = (width, 1)
    grid_div_x = ["block_size_x"]
    grid_div_y = []

    tune_params = OrderedDict()
    tune_params["block_size_x"] = [32*i for i in range(1,9)]
    tune_params["block_size_y"] = [2**i for i in range(6)]

    tune_kernel("computeMeanVertically", kernel_string, problem_size, args, tune_params,
        grid_div_y=grid_div_y, grid_div_x=grid_div_x)

def tune_horizontal(kernel_string, image, height, width):
    args = [height, width, image]

    #use only one column of thread blocks
    problem_size = (1, height)
    grid_div_x = []
    grid_div_y = ["block_size_y"]

    tune_params = OrderedDict()
    tune_params["block_size_x"] = [2**i for i in range(11)]
    tune_params["block_size_y"] = [2**i for i in range(6)]

    tune_kernel("computeMeanHorizontally", kernel_string, problem_size, args, tune_params,
        grid_div_y=grid_div_y, grid_div_x=grid_div_x)

def tune_transpose(kernel_string, image, height, width):
    output = np.zeros((width,height), dtype=np.float32)
    args = [height, width, output, image]

    #tune the transpose kernel
    problem_size = (width, height)
    grid_div_x = ["block_size_x"]
    grid_div_y = ["block_size_y"]

    tune_params = OrderedDict()
    tune_params["block_size_x"] = [32*i for i in range(1,9)]
    tune_params["block_size_y"] = [2**i for i in range(6)]

    tune_kernel("transpose", kernel_string, problem_size, args, tune_params,
        grid_div_y=grid_div_y, grid_div_x=grid_div_x)



if __name__ == "__main__":
    tune_zeromean()


