from __future__ import print_function

from collections import OrderedDict
import numpy as np
from kernel_tuner import tune_kernel, run_kernel
from scipy.misc import imread

from context import get_kernel_path, get_testdata_path


def test_wiener():

    with open(get_kernel_path()+'wienerfilter.cu', 'r') as f:
        kernel_string = f.read()

    image = imread(get_testdata_path() + "test.jpg", mode="F")

    height = np.int32(image.shape[0])
    width = np.int32(image.shape[1])
    problem_size = (width, height)

    output = np.zeros(problem_size, dtype=np.float32)

    args = [height, width, output, image]

    params = OrderedDict()
    params["block_size_x"] = 32
    params["block_size_y"] = 8
    params["reuse_computation"] = 1

    answer = run_kernel("computeVarianceEstimates",
        kernel_string, problem_size, args, params, grid_div_y=["block_size_y"])

    reference = run_kernel("computeVarianceEstimates_naive",
        kernel_string, problem_size, args, params, grid_div_y=["block_size_y"])

    assert np.allclose(answer[2], reference[2], atol=1e-6)



def test_variance_zero_mean():

    with open(get_kernel_path()+'wienerfilter.cu', 'r') as f:
        kernel_string = f.read()

    image = imread(get_testdata_path() + "test.jpg", mode="F")

    height = np.int32(image.shape[0])
    width = np.int32(image.shape[1])
    size = np.int32(height*width)

    params = OrderedDict()
    params["block_size_x"] = 512
    params["num_blocks"] = 64

    num_blocks = params["num_blocks"]
    output = np.zeros(num_blocks, dtype=np.float32)

    args = [size, output, image]
    problem_size = ("num_blocks", 1)

    answer = run_kernel("computeVarianceZeroMean", kernel_string, problem_size, args, params, grid_div_x=[])

    print("answer:")
    ans = np.sum(answer[1])
    print(ans, answer[1])
    print("reference:")
    reference = np.sum(image*image)
    print(reference)

    assert np.isclose(ans, reference, atol=1e-6)
