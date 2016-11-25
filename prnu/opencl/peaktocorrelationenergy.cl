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
 * This file contains CUDA kernels for comparing two PRNU noise patterns
 * using Peak To Correlation Energy.
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
    ;
    ;
    ;
    ;
    ;


    ;
    ;
}


/**
 * Simple helper kernel to convert an array of real values to an array of complex values
 */
__kernel void toComplex(int h, int w, __global float* x, __global float *input_x) {
    int i = get_local_id(1) + get_group_id(1) * block_size_y;
    int j = get_local_id(0) + get_group_id(0) * block_size_x;

    if (i < h && j < w) {
        x[i * w * 2 + 2 * j] = input_x[i * w + j];
        x[i * w * 2 + 2 * j + 1] = 0.0f;
    }
}

/**
 * Simple helper kernel to convert an array of real values to a flipped array of complex values
 */
__kernel void toComplexAndFlip(int h, int w, __global float *y, __global float* input_y) {
    int i = get_local_id(1) + get_group_id(1) * block_size_y;
    int j = get_local_id(0) + get_group_id(0) * block_size_x;

    if (i < h && j < w) {
        //y is flipped vertically and horizontally
        int yi = h - i -1;
        int yj = w - j -1;
        y[yi* w * 2 + 2 * yj] = input_y[i * w + j];
        y[yi* w * 2 + 2 * yj + 1] = 0.0f;
    }
}

/**
 * Two-in-one kernel that puts x and y to Complex, but flips y
 */
__kernel void toComplexAndFlip2(int h, int w, __global float* x, __global float *y, __global float *input_x, __global float* input_y) {
    int i = get_local_id(1) + get_group_id(1) * block_size_y;
    int j = get_local_id(0) + get_group_id(0) * block_size_x;

    if (i < h && j < w) {
        x[i * w * 2 + 2 * j] = input_x[i * w + j];
        x[i * w * 2 + 2 * j + 1] = 0.0f;

        //y is flipped vertically and horizontally
        int yi = h - i -1;
        int yj = w - j -1;
        y[yi* w * 2 + 2 * yj] = input_y[i * w + j];
        y[yi* w * 2 + 2 * yj + 1] = 0.0f;

    }
}



/*
 * This method computes a cross correlation in frequency space
 */
__kernel void computeCrossCorr(int h, int w, __global float *c, __global float *x, __global float *y) {
    int i = get_local_id(1) + get_group_id(1) * block_size_y;
    int j = get_local_id(0) + get_group_id(0) * block_size_x;

    if (i < h && j < w) {
        float xRe = x[i*w*2+j*2];
        float xIm = x[i*w*2+j*2+1];
        float yRe = y[i*w*2+j*2];
        float yIm = y[i*w*2+j*2+1];
        c[i*w*2+j*2] = (xRe * yRe) - (xIm * yIm);
        c[i*w*2+j*2+1] = (xRe * yIm) + (xIm * yRe);
    }
}


/* 
 * This method searches for the peak value in a cross correlated signal and outputs the index
 * input is assumed to be a complex array of which only the real component contains values that
 * contribute to the peak
 *
 * Thread block size should be power of two because of the reduction.
 * The implementation currently assumes only one thread block is used for the entire input array
 * 
 * In case of multiple thread blocks initialize output to zero and use atomic add or another kernel
 */
#ifndef block_size_x
#define block_size_x 1024      //has to be a power of two because of reduce
#endif
__kernel void findPeak(int h, int w, __global float *peakValue, __global float *peakValues, __global int *peakIndex, __global float *input) {

    int x = get_group_id(0) * block_size_x + get_local_id(0);
    int ti = get_local_id(0);
    int step_size = get_num_groups(0) * block_size_x;
    int n = h*w;
    __local float shmax[block_size_x];
    __local int shind[block_size_x];

    //compute thread-local sums
    float max = -1.0f;
    float val = 0.0f;
    int index = -1;
    for (int i=x; i < n; i+=step_size) {
        val = fabs(input[i*2]); //input is a complex array, only using real value 
        if (val > max) {
            max = val;
            index = i;
        }
    }
        
    //store local sums in shared memory
    shmax[ti] = max;
    shind[ti] = index;
    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
        
    //reduce local sums
    for (unsigned int s=block_size_x/2; s>0; s>>=1) {
        if (ti < s) {
            float v1 = shmax[ti];
            float v2 = shmax[ti + s];
            if (v1 < v2) {
                shmax[ti] = v2;
                shind[ti] = shind[ti + s];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
    }
        
    //write result
    if (ti == 0) {
        peakValues[get_group_id(0)] = shmax[0];
        peakIndex[get_group_id(0)] = shind[0];
        if (get_group_id(0) == 0) {
            peakValue[0] = input[n*2-2]; //instead of using real peak use last real value
        }
    }

}


/* 
 * This method computes the energy of the signal minus an area around the peak
 *
 * input is assumed to be a complex array of which only the real component
 * contains values that contribute to the energy
 *
 * Thread block size should be power of two because of the reduction.
 * The implementation currently assumes only one thread block is used for the entire input array
 * 
 * In case of multiple thread blocks run kernel twice, with 1 thread block the second time
 */
#define SQUARE_SIZE 11
#define RADIUS 5
__kernel void computeEnergy(int h, int w, __global double *energy, __global int *peakIndex, __global float *input) {

    int x = get_group_id(0) * block_size_x + get_local_id(0);
    int ti = get_local_id(0);
    int step_size = get_num_groups(0) * block_size_x;
    int n = h*w;
    __local double shmem[block_size_x];

    int peak_i = peakIndex[0];
    int peak_y = peak_i / w;
    int peak_x = peak_i - (peak_y * w);

    if (ti < n) {

        //compute thread-local sums
        double sum = 0.0f;
        for (int i=x; i < n; i+=step_size) {
            int row = i / w;
            int col = i - (row*w);

            //exclude area around the peak from sum
            int peakrow = (row > peak_y - RADIUS && row < peak_y + RADIUS);
            int peakcol = (col > peak_x - RADIUS && col < peak_x + RADIUS);
            if (peakrow && peakcol) {
                continue;
            } else {
                double val = input[row*w*2+col*2];
                sum += val * val;
            }
        }
        
        //store local sums in shared memory
        shmem[ti] = sum;
        barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
        
        //reduce local sums
        for (unsigned int s=block_size_x/2; s>0; s>>=1) {
            if (ti < s) {
                shmem[ti] += shmem[ti + s];
            }
            barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
        }
        
        //write result
        if (ti == 0) {
            //use 1 thread block or multiple kernel calls
            energy[get_group_id(0)] = shmem[0] / (double)((w*h) - (SQUARE_SIZE * SQUARE_SIZE));

            //use atomics in case of multiple threads and single kernel call
            //don't forget to zero output by the host
            //double l_energy = shmem[0] / (double)(w*h);
            //my_atomicAdd(energy, l_energy);
        }

    }
}


 double my_atomicAdd(double* address, double val)
{
    unsigned long long int* address_as_ull =
                              (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val +
                               __longlong_as_double(assumed)));

    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);

    return __longlong_as_double(old);
}



/*
 * Simple CUDA Helper function to reduce the output of a
 * reduction kernel with multiple thread blocks to a single value
 * 
 * This function performs a sum of an array of doubles
 *
 * This function is to be called with only a single thread block
 */
__kernel void sumDoubles(__global double *output, __global double *input, int n) {
    int ti = get_local_id(0);
    __local double shmem[block_size_x];

    //compute thread-local sums
    double sum = 0.0;
    for (int i=ti; i < n; i+=block_size_x) {
        sum += input[i];
    }
        
    //store local sums in shared memory
    shmem[ti] = sum;
    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
        
    //reduce local sums
    for (unsigned int s=block_size_x/2; s>0; s>>=1) {
        if (ti < s) {
            shmem[ti] += shmem[ti + s];
        }
        barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
    }
        
    //write result
    if (ti == 0) {
        output[0] = shmem[0];
    }
}


/*
 * Simple CUDA helper functions to reduce the output of a reducing kernel with multiple
 * thread blocks to a single value
 *
 * This function performs a reduction for the max and the location of the max
 *
 * This function is to be called with only one thread block
 */
__kernel void maxlocFloats(__global int *output_loc, __global float *output_float, __global int *input_loc, __global float *input_float, int n) {

    int ti = get_local_id(0);
    __local float shmax[block_size_x];
    __local int shind[block_size_x];

    //compute thread-local variables
    float max = -1.0f;
    float val = 0.0f;
    int loc = -1;
    for (int i=ti; i < n; i+=block_size_x) {
         val = input_float[i];
         if (val > max) {
             max = val;
             loc = input_loc[i];
         }
    }
        
    //store local variables in shared memory
    shmax[ti] = max;
    shind[ti] = loc;
    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
        
    //reduce local variables
    for (unsigned int s=block_size_x/2; s>0; s>>=1) {
        if (ti < s) {
            float v1 = shmax[ti];
            float v2 = shmax[ti + s];
            if (v1 < v2) {
                shmax[ti] = v2;
                shind[ti] = shind[ti + s];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
    }
        
    //write result
    if (ti == 0) {
        output_float[0] = shmax[0]; 
        output_loc[0] = shind[0]; 
    }

}

