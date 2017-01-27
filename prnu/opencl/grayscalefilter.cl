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
 * This file contains the CUDA kernel for converting an image into
 * a grayscale array of floats. Scaling factors used are:
 * 0.299 r + 0.587 g + 0.114 b
 *
 * @author Ben van Werkhoven <b.vanwerkhoven@esciencecenter.nl>
 * @version 0.1
 */
extern "C" {
//    __kernel void grayscale(int h, int w, __global float* output, __global uchar3* input);
    __kernel void grayscale(int h, int w, __global float* output, __global char* input);
}

/*
 * Naive grayscale kernel
 *
 * Bytes go in, floats come out, alpha is ignored
 *
 * get_num_groups(0) = w / block_size_x  (ceiled)
 * get_num_groups(1) = h / block_size_y  (ceiled)
 */
//__kernel void grayscale(int h, int w, __global float* output, __global uchar3* input) {
__kernel void grayscale(int h, int w, __global float* output, __global char* input) {
	int i = get_local_id(1) + get_group_id(1) * block_size_y;
	int j = get_local_id(0) + get_group_id(0) * block_size_x;
	
    uchar3 *c3_input = (uchar3 *)input;

	if (j < w && i < h) {

		uchar3 c = c3_input[i*w+j];

//          float b = (float) input[(i*w+j) * 3 + 0] & 0xFFFF;
//          float g = (float) input[(i*w+j) * 3 + 1] & 0xFFFF;
//          float r = (float) input[(i*w+j) * 3 + 2] & 0xFFFF;
		output[i*w+j] = 0.299f*c.z + 0.587f*c.y + 0.114f*c.x;




	}
}




