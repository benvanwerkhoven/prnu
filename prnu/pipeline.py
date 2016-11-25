#!/usr/bin/env python
import numpy
import kernel_tuner

from scipy import misc
#from matplotlib import pyplot

#image = misc.imread("../test_small.jpg", "r")
image = misc.imread("../test_small.jpg", mode='RGB')

misc.imshow(image)

print (image.shape)







exit()


kernel_names = []
""" pipeline overview

-- fastnoisefilter --
    zeromem dxs
    zeromem dys
    convolveHorizontally dxs
    convolveVertically dys
    normalize
    zeromem input
    convolveHorizontally input
    convolveVertically input

-- zeromeantotalfilter --
    computeMeanVertically
    transpose
    computeMeanVertically
    transpose

-- wienerfilter
    tocomplex
    fft
    computeSquaredMagnitudes
    computeVarianceEstimates
    computeVarianceZeroMean
    scaleWithVariances
    ifft
    normalizeToReal

"""





with open('fastnoisefilter.cu', 'r') as f:
    fastnoise_string = f.read()
with open('zeromeantotalfilter.cu', 'r') as f:
    zeromean_string = f.read()
with open('wienerfilter.cu', 'r') as f:
    wiener_string = f.read()





height = numpy.int32(image.shape[0])
width = numpy.int32(image.shape[1])

problem_size = (width, 1)

image = numpy.random.randn(height*width).astype(numpy.float32)

args = [height, width, image]

tune_params = dict()
tune_params["block_size_x"] = [32*i for i in range(1,9)]
tune_params["block_size_y"] = [2**i for i in range(6)]

grid_div_x = ["block_size_x"]
grid_div_y = None

kernel_tuner.tune_kernel(kernel_name, kernel_string,
    problem_size, args, tune_params,
    grid_div_y=grid_div_y, grid_div_x=grid_div_x)






