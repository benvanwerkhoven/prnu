#!/usr/bin/env python
from __future__ import print_function

from collections import OrderedDict
import numpy as np
from numpy.fft import fft2, ifft2
from kernel_tuner import tune_kernel, run_kernel
from scipy.misc import imread

from context import get_kernel_path, get_testdata_path

from matplotlib import pyplot

def fastnoise(image):
    gradient = np.gradient(image)
    norm = np.sqrt((gradient[0]**2) + (gradient[1]**2))
    scale = np.ones_like(norm) / (1.0 + norm)
    step1 = gradient[0] * scale + gradient[1] * scale
    gradient2 = np.gradient(step1)
    return gradient2[0] + gradient2[1]

def tune_pce():

    with open(get_kernel_path()+'peaktocorrelationenergy.cu', 'r') as f:
        kernel_string = f.read()

    image = imread(get_testdata_path() + "Pentax_OptioA40_0_30731.JPG", mode="F")
    image = fastnoise(image)

    image2 = imread(get_testdata_path() + "Pentax_OptioA40_0_30757.JPG", mode="F")
    image2 = fastnoise(image2)

    height = np.int32(image.shape[0])
    width = np.int32(image.shape[1])

    image_freq, image2_freq = tune_complex_and_flip(kernel_string, height, width, image, image2)

    crosscorr = tune_crosscorr(kernel_string, height, width, image_freq, image2_freq)

    loc, val = tune_find_peak(kernel_string, height, width, crosscorr)

    energy = tune_energy(kernel_string, height, width, crosscorr, loc)

    pce_score = (val[0] * val[0]) / energy
    print("Finished tuning PCE, pce_score=", pce_score)


def tune_complex_and_flip(kernel_string, height, width, image, image2):
    """step 1 convert to complex data structure and flip pattern"""
    problem_size = (width, height)
    tune_params = OrderedDict()
    tune_params["block_size_x"] = [32*i for i in range(1,33)]
    tune_params["block_size_y"] = [2**i for i in range(6)]

    image_freq = np.zeros((height,width,2), dtype=np.float32)
    image2_freq = np.zeros((height,width,2), dtype=np.float32)

    args = [height, width, image_freq, image2_freq, image, image2]
    params = {"block_size_x": 32, "block_size_y": 16}
    output = run_kernel("toComplexAndFlip2",
        kernel_string, problem_size, args, params, grid_div_y=["block_size_y"])

    tune_kernel("toComplexAndFlip2", kernel_string, problem_size,
        args, tune_params, grid_div_y=["block_size_y"])

    return output[2], output[3]


def tune_crosscorr(kernel_string, height, width, image_freq, image2_freq):
    """step 2 Fourier transforms and cross correlation"""
    problem_size = (width, height)
    tune_params = OrderedDict()
    tune_params["block_size_x"] = [32*i for i in range(1,33)]
    tune_params["block_size_y"] = [2**i for i in range(6)]

    image_freq = image_freq.reshape(height,width,2)
    image_freq = image_freq[:,:,0] + 1j * image_freq[:,:,1]
    image_freq = fft2(image_freq).astype(np.complex64)

    image2_freq = image2_freq.reshape(height,width,2)
    image2_freq = image2_freq[:,:,0] + 1j * image2_freq[:,:,1]
    image2_freq = fft2(image2_freq).astype(np.complex64)

    crosscorr = np.zeros((height,width,2), dtype=np.float32)

    args = [height, width, crosscorr, image_freq, image2_freq]
    params = {"block_size_x": 32, "block_size_y": 16}
    output = run_kernel("computeCrossCorr",
        kernel_string, problem_size, args, params, grid_div_y=["block_size_y"])

    tune_kernel("computeCrossCorr",
        kernel_string, problem_size, args, tune_params, grid_div_y=["block_size_y"])

    crosscorr = output[2].reshape(height,width,2)
    crosscorr_invert = crosscorr[:,:,0] + 1j * crosscorr[:,:,1]
    crosscorr_invert = ifft2(crosscorr_invert)

    crosscorr[:,:,0] = crosscorr_invert.real
    crosscorr[:,:,1] = crosscorr_invert.imag

    return crosscorr


def tune_find_peak(kernel_string, height, width, crosscorr):
    """step 3 find peak"""
    tune_params = OrderedDict()
    tune_params["block_size_x"] = [2**i for i in range(5,11)]
    tune_params["num_blocks"] = [2**i for i in range(5,11)]
    max_blocks = max(tune_params["num_blocks"])

    params = {"block_size_x": 512, "num_blocks": 64}
    num_blocks = np.int32(params["num_blocks"])
    problem_size = ("num_blocks", 1)

    peakvals = np.zeros((max_blocks), dtype=np.float32)
    peakindx = np.zeros((max_blocks), dtype=np.int32)
    loc = np.zeros((1), dtype=np.int32)
    val = np.zeros((1), dtype=np.float32)

    args = [height, width, peakvals, peakindx, crosscorr]
    output1 = run_kernel("findPeak",
        kernel_string, problem_size, args, params, grid_div_x=[])

    tune_kernel("findPeak",
        kernel_string, problem_size, args, tune_params, grid_div_x=[])

    peakvals = output1[2]
    peakindx = output1[3]

    args = [loc, val, peakindx, peakvals, num_blocks]
    output2 = run_kernel("maxlocFloats",
        kernel_string, (1,1), args, params, grid_div_x=[])

    loc = output2[0]
    val = output2[1]
    return loc, val


def tune_energy(kernel_string, height, width, crosscorr, loc):
    """step 4 compute energy"""

    tune_params = OrderedDict()
    tune_params["block_size_x"] = [2**i for i in range(5,11)]
    tune_params["num_blocks"] = [2**i for i in range(5,11)]
    max_blocks = max(tune_params["num_blocks"])

    params = {"block_size_x": 512, "num_blocks": 64}
    num_blocks = np.int32(params["num_blocks"])
    problem_size = ("num_blocks", 1)

    energy_part = np.zeros((max_blocks), dtype=np.float64)
    args = [height, width, energy_part, loc, crosscorr]

    output3 = run_kernel("computeEnergy",
        kernel_string, problem_size, args, params, grid_div_x=[])

    tune_kernel("computeEnergy",
        kernel_string, problem_size, args, tune_params, grid_div_x=[])

    energy_part = output3[2]
    energy = np.zeros((1), dtype=np.float64)

    args = [energy, energy_part, num_blocks]
    output4 = run_kernel("sumDoubles",
        kernel_string, (1,1), args, params, grid_div_x=[])

    energy = output4[0]
    return energy


if __name__ == "__main__":
    tune_pce()
