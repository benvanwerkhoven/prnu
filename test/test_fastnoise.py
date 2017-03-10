from __future__ import print_function

from collections import OrderedDict
import numpy as np
from nose.tools import nottest
from kernel_tuner import tune_kernel, run_kernel
from scipy.misc import imread

from context import get_kernel_path, get_testdata_path

@nottest
def fastnoise_method_ben(image):
    d = np.gradient(image)
    norm = np.sqrt( (d[0]*d[0]) + (d[1]*d[1]) )
    scale = 1.0 / (1.0 + norm)
    reference = (d[0] * scale) + (d[1] * scale)
    gradient = reference[0] + reference[1]

    reference = np.gradient(image)
    reference = reference[0] + reference[1]
    return reference

@nottest
def fastnoise_method_gisolf(image):
    d = np.gradient(image)
    norm = np.sqrt( (d[0]*d[0]) + (d[1]*d[1]) )
    scale = 1.0 / (1.0 + norm)
    dxs = d[0] * scale
    dys = d[1] * scale

    reference = np.gradient(dxs, axis=0) + np.gradient(dys, axis=1)
    return reference




def test_fastnoise():

    with open(get_kernel_path()+'fastnoisefilter.cu', 'r') as f:
        kernel_string = f.read()

    image = imread(get_testdata_path() + "test.jpg", mode="F")

    height = np.int32(image.shape[0])
    width = np.int32(image.shape[1])
    problem_size = (width, height)

    output1 = np.zeros_like(image)
    output2 = np.zeros_like(image)
    output3 = np.zeros_like(image)

    args = [height, width, output1, output2, image]

    params = OrderedDict()
    params["block_size_x"] = 32
    params["block_size_y"] = 16

    d = np.gradient(image)
    norm = np.sqrt( (d[0]*d[0]) + (d[1]*d[1]) )
    scale = 1.0 / (1.0 + norm)
    dys = d[0] * scale
    dxs = d[1] * scale

    answer = run_kernel("normalized_gradient",
        kernel_string, problem_size, args, params)

    assert np.allclose(answer[2], dxs, atol=1e-6)
    assert np.allclose(answer[3], dys, atol=1e-6)

    args = [height, width, output3, dxs, dys]
    answer = run_kernel("gradient",
        kernel_string, problem_size, args, params)

    reference = np.gradient(dys, axis=0) + np.gradient(dxs, axis=1)

    assert np.allclose(answer[2], reference, atol=1e-6)


