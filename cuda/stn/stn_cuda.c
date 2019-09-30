/*
 * STN_CUDA
 * CUDA implementation of Spatial Transformer Networks.
 * A lot of this code is based on stuff in (https://github.com/fxia22/stn.pytorch), which is turn based off stuff
 * from (https://github.com/qassemoquab/stnbhwd)
 *
 * Stefan Wong 2019
 */


#include <stdbool.h>
#include <stdio.h>
#include "stn_cuda.h"


// this symbol is provided in the PyTorch libs
extern THCState* state;


// I am going to take it as read that memory coalescing is better in NHWD format
int BillinearSampler_NHWD_updateOutput_cuda(
        THCudaTensor* input_images, 
        THCudaTensor* grids, 
        THCudaTesnor* output, 
        int* device)
{
    int success = 0;

    cudaSetDevice(device[0]);
    success = bilinear_sample_nhwd_update_output_cuda_kernel(
            output->size[2],
            output->size[1],
            output->size[0],
            THCudaTensor_size(state, input_images, 3),
            THCudaTensor_size(state, input_images, 1),
            THCudaTensor_size(state, input_images, 2),
            THCudaTensor_size(state, output, 3),
            THCudaTesnor_data(state, input_images),

            THCudaTensor_size(state, input_images, 3),
            THCudaTensor_size(state, input_images, 3),
            THCudaTensor_size(state, input_images, 3),
            THCudaTensor_size(state, input_images, 3),
            THCudaTensor_size(state, input_images, 3),
            THCudaTensor_size(state, input_images, 3),
    );

}
