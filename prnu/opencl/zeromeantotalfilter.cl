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
#ifndef block_size_x
#define block_size_x 32
#endif

#ifndef block_size_y
#define block_size_y 32
#endif

//function interfaces to prevent C++ garbling the kernel names
extern "C" {
	;
	;
}


/*
 * This function applies the Zero Mean filter vertically.
 *
 * Setup this kernel as follows:
 * gridDim.x = ceil ( w / (block_size_x) )
 * gridDim.y = 1
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
		barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

		//reduce local sums
		for (unsigned int s=block_size_y/2; s>0; s>>=1) {
			if (ti < s) {
				shEven[ti][tj] += shEven[ti + s][tj];
				shOdd[ti][tj] += shOdd[ti + s][tj];
			}
			barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
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
 * gridDim.x = w / block_size_x  (ceiled)
 * gridDim.y = h / block_size_y  (ceiled)
 */
__kernel void transpose(int h, int w, __global float* output, __global float* input) {
	int i = get_local_id(1) + get_group_id(1) * block_size_y;
	int j = get_local_id(0) + get_group_id(0) * block_size_x;

	if (j < w && i < h) {
		output[j*h+i] = input[i*w+j];
	}
}






























