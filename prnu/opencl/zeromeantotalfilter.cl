/*
* Copyright 2015 Netherlands eScience Center, VU University Amsterdam, and Netherlands Forensic Institute
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/

/**
 * This file contains CUDA kernels for applying a zero-mean total
 * filter to a PRNU pattern, as proposed by:
 * M. Chen et al. "Determining image origin and integrity using sensor
 * noise", IEEE Trans. Inf. Forensics Secur. 3 (2008) 74-90.
 *
 * The Zero Mean filter ensures that even and uneven subsets of columns
 * and rows in a checkerboard pattern become zero to remove any linear
 * patterns in the input.
 *
 * To apply the complete filter:
 *  computeMeanVertically(h, w, input);
 *  transpose(h, w, input);
 *  computeMeanVertically(h, w, input);
 *  transpose(h, w, input);
 *
 * @author Ben van Werkhoven <b.vanwerkhoven@esciencecenter.nl>
 * @version 0.1
 */

//function interfaces to prevent C++ garbling the kernel names
extern "C" {
    __kernel void computeMeanVertically(int h, int w, __global float* input);
    __kernel void transpose(int h, int w, __global float* output, __global float* input);
    __kernel void computeMeanHorizontally(int h, int w, __global float* input);
}


/*
 * This function applies the Zero Mean filter vertically.
 *
 * Setup this kernel as follows:
 * get_num_groups(0) = ceil ( w / (block_size_x) )
 * get_num_groups(1) = 1
 *
 * block_size_x (block_size_y) = multiple of 32
 * block_size_y (block_size_y) = power of 2
 */
//#define block_size_y 1
//#define block_size_y 256
__kernel void computeMeanVertically(int h, int w, __global float* input) {
    int j = get_local_id(0) + get_group_id(0) * block_size_x;
    int ti = get_local_id(1);
    int tj = get_local_id(0);

    if (j < w) {
        float sumEven = 0.0f;
        float sumOdd = 0.0f;

        //iterate over vertical domain
        for (int i = 2*ti; i < h-1; i += 2*block_size_y) {
            sumEven += input[i*w+j];
            sumOdd += input[(i+1)*w+j];
        }
        if (ti == 0 && h & 1) { //if h is odd
            sumEven += input[(h-1)*w+j];
        }

        //write local sums into shared memory
        __local float shEven[block_size_y][block_size_x];
        __local float shOdd[block_size_y][block_size_x];

        shEven[ti][tj] = sumEven;
        shOdd[ti][tj] = sumOdd;
        barrier(CLK_LOCAL_MEM_FENCE);

        //reduce local sums
        for (unsigned int s=block_size_y/2; s>0; s>>=1) {
            if (ti < s) {
                shEven[ti][tj] += shEven[ti + s][tj];
                shOdd[ti][tj] += shOdd[ti + s][tj];
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }

        //compute means
        float meanEven = shEven[0][tj] / ((h + 1) / 2);
        float meanOdd = shOdd[0][tj] / (h / 2);

        //iterate over vertical domain
        for (int i = 2*ti; i < h-1; i += 2*block_size_y) {
            input[i*w+j] -= meanEven;
            input[(i+1)*w+j] -= meanOdd;
        }
        if (ti == 0 && h & 1) { //if h is odd
            input[(h-1)*w+j] -= meanEven;
        }
    }
}
__kernel void computeMeanVertically_naive(int h, int w, __global float* input) {
    int j = get_local_id(0) + get_group_id(0) * block_size_x;

    if (j < w) {
        float sumEven = 0.0f;
        float sumOdd = 0.0f;

        //iterate over vertical domain
        for (int i = 0; i < h-1; i += 2) {
            sumEven += input[i*w+j];
            sumOdd += input[(i+1)*w+j];
        }
        if (h & 1) { //if h is odd
            sumEven += input[(h-1)*w+j];
        }

        //compute means
        float meanEven = sumEven / ((h + 1) / 2);
        float meanOdd = sumOdd / (h / 2);

        //iterate over vertical domain
        for (int i = 0; i < h-1; i += 2) {
            input[i*w+j] -= meanEven;
            input[(i+1)*w+j] -= meanOdd;
        }
        if (h & 1) { //if h is odd
            input[(h-1)*w+j] -= meanEven;
        }
    }
}



/*
 * Naive transpose kernel
 *
 * get_num_groups(0) = w / block_size_x  (ceiled)
 * get_num_groups(1) = h / block_size_y  (ceiled)
 */
__kernel void transpose(int h, int w, __global float* output, __global float* input) {
    int i = get_local_id(1) + get_group_id(1) * block_size_y;
    int j = get_local_id(0) + get_group_id(0) * block_size_x;

    if (j < w && i < h) {
        output[j*h+i] = input[i*w+j];
    }
}






/**
 * Applies an in place zero mean filtering operation to each row in an image.
 * First two mean values are computed, one for even and one for odd elements,
 * for each row in the image. Then, the corresponding mean value is subtracted
 * from each pixel value in the image.
 *
 * block_size_x power of 2
 * block_size_y any
 */
__kernel void computeMeanHorizontally(int h, int w, __global float *input) {
    int i = get_local_id(1) + get_group_id(1) * block_size_y;
    int tj = get_local_id(0);

    if (i < h) {

        float sumEven = 0.0f;
        float sumOdd = 0.0f;
        for (int j = 2*tj; j < w - 1; j += 2*block_size_x) {
            sumEven += input[i*w+j];
            sumOdd += input[i*w+j + 1];
        }
        if (tj == 0 && w & 1) {    // if w is odd 
            sumEven += input[i*w+(w-1)];
        }

        #if block_size_x > 1
        int ti = get_local_id(1);
        //write local sums into shared memory
        __local float shEven[block_size_y][block_size_x];
        __local float shOdd[block_size_y][block_size_x];
        shEven[ti][tj] = sumEven;
        shOdd[ti][tj] = sumOdd;
        barrier(CLK_LOCAL_MEM_FENCE);

        //reduce local sums
        for (unsigned int s=block_size_x/2; s>0; s>>=1) {
            if (tj < s) {
                shEven[ti][tj] += shEven[ti][tj + s];
                shOdd[ti][tj] += shOdd[ti][tj + s];
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        sumEven = shEven[ti][0];
        sumOdd = shOdd[ti][0];
        #endif

        float meanEven = sumEven / ((w + 1) / 2);
        float meanOdd = sumOdd / (w / 2);

        for (int j = 2*tj; j < w - 1; j += 2*block_size_x) {
            input[i*w+j] -= meanEven;
            input[i*w+j + 1] -= meanOdd;
        }
        if (tj == 0 && w & 1) {    // if w is odd 
            input[i*w+(w-1)] -= meanEven;
        }
    }
}


