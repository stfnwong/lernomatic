/*
 * STN_CUDA
 * CUDA implementation of Spatial Transformer Networks.
 * A lot of this code is based on stuff in (https://github.com/fxia22/stn.pytorch), which is turn based off stuff
 * from (https://github.com/qassemoquab/stnbhwd)
 *
 * Stefan Wong 2019
 */

#ifndef __LERNOMATIC_STN_CUDA
#define __LERNOMATIC_STN_CUDA

// Billinear sampling is  done in NHWD since coalescing is not as obvious in NDHW 
// Therefore, we assume BHWD format in input images, and BHW(YX) format on grids


#include <THC/THC.h>

int BillinearSampler_NHWD_updateOutput_cuda(
        THCudaTensor* input_images, 
        THCudaTensor* grids, 
        THCudaTesnor* output, 
        int* device
);

int BillinearSampler_NHWD_updateGradInput_cuda(
        THCudaTensor* input_images, 
        THCudaTensor* grids,
        THCudaTensor* grad_input_images,
        THCudaTensor* grad_grids,
        THCudaTensor* grad_output,
        int*          device
);


#endif /*__LERNOMATIC_STN_CUDA*/
