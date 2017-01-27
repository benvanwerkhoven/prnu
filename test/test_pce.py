from __future__ import print_function

from collections import OrderedDict
import numpy as np
from kernel_tuner import tune_kernel, run_kernel
from scipy.misc import imread

from context import get_kernel_path, get_testdata_path


def test_complex_and_flip2():

    with open(get_kernel_path()+'peaktocorrelationenergy.cu', 'r') as f:
        kernel_string = f.read()

    image = imread(get_testdata_path() + "test_small.jpg", mode="F")

    height = np.int32(image.shape[0])
    width = np.int32(image.shape[1])
    problem_size = (width, height)

    output = np.zeros((height, width,2), dtype=np.float32)

    args = [height, width, output, output, image, image]

    params = OrderedDict()
    params["block_size_x"] = 32
    params["block_size_y"] = 16

    answer = run_kernel("toComplexAndFlip2",
        kernel_string, problem_size, args, params,
        grid_div_y=["block_size_y"], grid_div_x=["block_size_x"])

    output1 = answer[2].reshape(height, width, 2)
    output1 = output1[:,:,0] + 1j * output[:,:,1]
    reference1 = image + 1j * np.zeros((height, width), dtype=np.float32)
    assert np.allclose(output1, reference1, atol=1e-6)

    reference2 = image.flatten()[::-1].reshape(height, width)
    reference2 = reference2
    output2 = answer[3].reshape(height, width, 2)
    assert np.allclose(output2[:,:,0], reference2, atol=1e-6)
    assert np.allclose(output2[:,:,1], np.zeros((height,width), dtype=np.float32), atol=1e-6)




def test_find_peak():

    with open(get_kernel_path()+'peaktocorrelationenergy.cu', 'r') as f:
        kernel_string = f.read()

    image = imread(get_testdata_path() + "test_small.jpg", mode="F")

    height = np.int32(image.shape[0])
    width = np.int32(image.shape[1])
    problem_size = (width, height)

    #generate some bogus crosscorr data
    crosscorr = np.random.randn(height, width, 2).astype(np.float32)

    #compute reference in Python
    peak_index = np.argmax(np.absolute(crosscorr[:,:,0]))
    peak_value = np.absolute(crosscorr[:,:,0].flatten()[peak_index])

    params = {"block_size_x": 512, "num_blocks": 64}
    problem_size = ("num_blocks", 1)
    num_blocks = np.int32(params["num_blocks"])

    peakvals = np.zeros((num_blocks), dtype=np.float32)
    peakindx = np.zeros((num_blocks), dtype=np.int32)
    loc = np.zeros((1), dtype=np.int32)
    val = np.zeros((1), dtype=np.float32)

    args = [height, width, peakvals, peakindx, crosscorr]
    output1 = run_kernel("findPeak",
        kernel_string, problem_size, args, params, grid_div_x=[])

    peakvals = output1[2]
    peakindx = output1[3]

    args = [loc, val, peakindx, peakvals, num_blocks]
    output2 = run_kernel("maxlocFloats",
        kernel_string, (1,1), args, params, grid_div_x=[])

    loc = output2[0][0]
    val = output2[1][0]

    print("answer")
    print("loc=", loc, "val=", val)

    print("reference")
    print("loc=", peak_index, "val=", peak_value)

    assert loc == peak_index
    assert np.isclose(val, peak_value, atol=1e-6)

