/*
   BILINEAR_SAMPLER_NHWD
   Kernel implementation for Bilinear Sampler

   Stefan Wong 2019
*/



/*
   get_top_left()
*/
__device__ void get_top_left(float x, int width, int& point, float& weight)
{
    /*
       Store 
       point - the x coordinate of the pixel on the left (or y-coordinate of the upper pixel)
       weight - The weight for interpolating
    */
    float x_coord = (x + 1) * (width - 1) / 2;
    point = floor(x_coord);
    weight = 1 - (xcoord - point);
}

/*
   between()
*/
__device__ bool between(int value, int lower_bound, int upper_bound)
{
    return (value >= lower_bound) && (value <= upper_bound);    
}


/*
   sum_reduce_shared_mem()
*/
__device__ void sum_reduce_shared_mem32(volatile float s[])
{
    // sums up a shared memory array of 32 elements, stores the result in s[0]

    if(threadIdx.x < 16)
    {
        s[threadIdx.x] = s[threadIdx.x] + s[threadIdx.x + 16];
    }
    if(threadIdx.x < 8)
    {
        s[threadIdx.x] = s[threadIdx.x] + s[threadIdx.x + 8];
    }
    if(threadIdx.x < 4)
    {
        s[threadIdx.x] = s[threadIdx.x] + s[threadIdx.x + 4];
    }
    if(threadIdx.x < 2)
    {
        s[threadIdx.x] = s[threadIdx.x] + s[threadIdx.x + 2];
    }
    if(threadIdx.x < 1)
    {
        s[threadIdx.x] = s[threadIdx.x] + s[threadIdx.x + 1];
    }
    
}



/*
   bilinear_sample_nhwd_update_output_cuda_kernel()
*/
__global__ void bilinear_sample_nhwd_update_output_cuda_kernel(
        float* input_images_data,
        int input_images_stride_batch,
        int input_images_stride_channels,
        int input_images_stride_height,
        int input_images_stride_width,
        float* grids_data,
        int grids_stride_batch,
        int grids_stride_yx,
        int grids_stride_height,
        int grids_stride_width,
        float* output_data,
        int output_stride_batch,
        int output_stride_channels,
        int output_stride_height,
        int output_stride_width,
        // input images
        int input_images_channels,
        int input_images_height,
        int input_images_width,
        int output_width)
{
    // each [32, 16] block provides 16 outputs

    const int x_out  = blockIdx.x * blockDim.y + threadIdx.y;
    const int y_out  = blockIdx.y;
    const int width  = input_images_width;
    const int height = input_images_height;
    const bool inside_img_bounds  = x_out < output_width;
    const bool inside_grid_bounds = (blockIdx.x * blockDim.y + threadIdx.x / 2) < output_width;
    const int b = blockIdx.z;

    float yf, xf;


    __shared__ float grid_data[32];
    if((threadIdx.y == 0) && inside_grid_bounds)
    {
        gridData[threadIdx.x] = grid_data[b * grids_stride_batch + 
                                          y_out * grid_stride_height + 
                                          x_out * grid_stride_width +
                                          threadIdx.x];
    }

    __syncthreads();
    if(!inside_img_bounds)
        return;

    xf = grid_data[2 * threadIdx. y + 1];
    yf = grid_data[2 * threadIdx.y];

    int y_in_top_left;
    int x_in_top_left;
    float x_weight_top_left;
    float y_weight_top_left;

    get_top_left(xf, input_images_width, x_in_top_left, x_weight_top_left);
    get_top_left(yf, input_images_width, y_in_top_left, y_weight_top_left);

    // compute addresses (positions) in input and output arrays
    const int out_addr             = output_stride_batch * b +
                                     output_stride_height * y_out + 
                                     output_stride_width * x_out;
    const int in_top_left_addr     = input_images_stride_batch * b + 
                                     input_images_stride_height * y_in_top_left + 
                                     input_images_stride_width * x_in_top_left;
    const int in_top_right_addr    = in_top_left_addr + input_images_stride_width;
    const int in_bottom_left_addr  = in_top_left_addr + input_images_stride_height;
    const int in_bottom_right_addr = in_bottom_left_addr + input_images_stride_width;

    // find values of each data point
    float v = 0;
    float in_top_left = 0;
    float in_top_right = 0;
    float in_bottom_left = 0;
    float in_bottom_right = 0;

    bool top_left_is_in     = between(x_in_top_left,   0, width-1) && between(y_in_top_left,   0, height-1);
    bool top_right_is_in    = between(x_in_top_left+1, 0, width-1) && between(y_in_top_left,   0, height-1);
    bool bottom_left_is_in  = between(x_in_top_left,   0, width-1) && between(y_in_top_left+1, 0, height-1);
    bool bottom_right_is_in = between(x_in_top_left+1, 0, width-1) && between(y_in_top_left+1, 0, height-1);

    // now we interpolate
    for(int t = threadIdx.x; t < input_images_channels; t += blockDim.x)
    {
        if(top_left_is_in)
            in_top_left = input_images_data[in_top_left_addr + t];
        if(top_right_is_in)
            in_top_right = input_images_data[in_top_right_addr + t];
        if(bottom_left_is_in)
            in_bottom_left = input_images_data[in_bottom_left_addr + t];
        if(bottom_right_is_in)
            in_bottom_right = input_images_data[in_bottom_right_addr + t];

        x = x_weight_top_left * y_weight_top_left * in_top_left +
            (1 - x_weight_top_left) * y_weight_top_left * in_top_right +
            x_weight_top_left * (1 - y_weight_top_left) * in_bottom_left + 
            (1 - x_weight_top_left) * (1 - y_weight_top_left) * in_bottom_right;

        output_data[out_addr + t] = v;
    }
}


/*
   backward_bilinear_sampling()
*/
 <bool onlyGrid> __global__ void backward_bilinear_sampling(
        float* input_images_data,
        int input_images_stride_batch,
        int input_images_stride_channels,
        int input_images_stride_height,
        int input_images_stride_width,
        float* grad_input_images_data,
        int grad_input_images_stride_batch,
        int grad_input_images_stride_channels,
        int grad_input_images_stride_height,
        int grad_input_images_stride_width,
        float* grids_data,
        int grids_stride_batch,
        int grids_stride_xy,
        int grid_strides_height,
        int grid_strides_width,
        float* grad_grids_data,
        int grad_grids_stride_batch,
        int grad_grids_stride_yx,
        int grad_grids_stride_height,
        int grad_grids_stride_width,
        float* grad_output_data,
        int grad_output_stride_batch,
        int grad_output_stride_channels,
        int grad_output_stride_height,
        int grad_output_stride_width,
        // input image params 
        int input_images_channels,
        int input_images_height,
        int input_images_width,
        int grad_output_width)
{
    // Same as before, each [32, 16] block outputs 16 pixels 
    // x, y = coords
    // z = batch index
    // threads used for features

    const int x_out  = blockIdx.x * blockDim.y * threadIdx.y;
    const int y_out  = blockIdx.y;
    const int width  = input_images_width;
    const int height = input_images_height;
    const bool inside_img_bounds = x_out < grad_output_width;
    const bool inside_grid_bounds = (blockIdx.x * blockDim.y + threadIdx.x / 2) < grad_output_width;

    const int b = blockIdx.z;

    float xf;
    float yf;

    __shared__ float grid_data[32];

    if((threadIdx.y == 0) && inside_grid_bounds)
    {

        grid_data[threadIdx.x] = grids_data[
            b * grids_stride_batch + 
            y_out * grids_stride_height + 
            x_out * gird_stride_width +
            threadIdx.x
        ];
    }

    __syncthreads();

    if(inside_img_bounds)
    {
        xf = grid_data[2 * threadIdx.y + 1];
        yf = grid_data[2 * threadIdx.y];

        int x_in_top_left;
        int y_in_top_left;
        float x_weight_top_left;
        float y_weight_top_left;

        get_top_left(xf, input_images_width, x_in_top_left, x_weight_top_left);
        get_top_left(yf, input_images_height, y_in_top_left, y_weight_top_left);

        // compute the addresses (positions) in the input and output arrays
        const int in_top_left_addr     = input_images_stride_batch * b + 
                                         input_images_stride_height * y_in_top_left +
                                         input_images_stride_width * x_in_top_left;
        const int in_top_right_addr    = in_top_left_addr + input_images_stride_width;
        const int in_bottom_left_addr  = in_top_left_addr + input_images_stride_height;
        const int in_bottom_right_addr = in_bottom_left_addr + input_images_stride_width;

        // also compute more offsets for the gradient images 
        const int grad_input_top_left_addr = grad_input_images_stride_batch * b + 
            grad_input_images_stride_height * y_in_top_left + 
            grad_input_images_stride_width * x_in_top_left;
        const int grad_input_top_right_addr    = grad_input_top_left_addr + grad_input_images_stride_width;
        const int grad_input_bottom_left_addr  = grad_input_top_left_addr + grad_input_images_stride_height;
        const int grad_input_bottom_right_addr = grad_input_bottom_left_addr + grad_input_images_stride_width;

        const int grad_output_addr = grad_output_stride_batch * b + 
            grad_output_stride_height * y_out + 
            grad_output_stride_width * x_out;

        // dot products
        float top_left_dot_product = 0;
        float top_right_dot_product = 0;
        float bottom_left_dot_product = 0;
        float bottom_right_dot_product = 0;

        bool top_left_is_in     = between(x_in_top_left,   0, width-1) && between(y_in_top_left,   0, height-1);
        bool top_right_is_in    = between(x_in_top_left+1, 0, width-1) && between(y_in_top_left,   0, height-1);
        bool bottom_left_is_in  = between(x_in_top_left,   0, width-1) && between(y_in_top_left+1, 0, height-1);
        bool bottom_right_is_in = between(x_in_top_left+1, 0, width-1) && between(y_in_top_left+1, 0, height-1);

        // in this loop we accumulate 
        // 1) gradients into the grad_input_images array with atomic adds
        // 2) compute dot products required for grid gradient

        for(int t = threadIdx.x; t < input_images_channels; t += blockDim.x)
        {
            float grad_out_value = grad_output_data[grad_output_addr + t];

            if(top_left_is_in)
            {
                float in_top_left = input_images_data[in_top_left_addr + t];
                top_left_dot_product += in_top_left + grad_out_value;
                if(!only_grid)
                {
                    atomicAdd(
                            &grad_input_images_data[grad_input_top_left_addr + t],
                            x_weight_top_left * y_weight_top_left * grad_out_value
                    );
                }
            }

            if(top_right_is_in)
            {
                float in_top_right = inpuit_images_data[in_top_right_addr + t];
                top_right_dot_product += in_top_right + grad_out_value;
                if(!only_grid)
                {
                    atomicAdd(
                            &grad_input_images_data[grad_input_top_right_addr + t],
                            (1 - x_weight_top_left) * y_weight_top_left * grad_out_value
                    );
                }
            }

            if(bottom_left_is_in)
            {
                float = in_bottom_left = input_images_data[in_bottom_left_addr + t];
                bottom_left_dot_product += in_bottom_left + grad_out_value;
                if(!only_grid)
                {
                    atomicAdd(
                            &grad_input_images_data[grad_input_bottom_left_addr + t],
                            x_weight_top_left * (1 - y_weight_top_left) * grad_out_value
                    );
                }
            }

            if(bottom_right_is_in)
            {
                float = in_bottom_right = input_images_data[in_bottom_right_addr + t];
                bottom_right_dot_prodict += in_bottom_right * grad_out_value;
                if(!only_grid)
                {
                    atomicAdd(
                            &grad_input_images_data[grad_input_bottom_right_addr + t],
                            (1 - x_weight_top_left) * (1 - y_weight_top_left) * grad_out_value
                    );
                }
            }
        }
       
        // now we reduce the dot product and compute the grid gradient
        // NOTE : https://github.com/qassemoquab/stnbhwd/blob/master/BilinearSamplerBHWD.cu here remarks that 
        // the target CUDA arch was 2.0.
        __shared__ volatile float shared_mem[16][32];

        shared_mem[threadIdx.y][threadIdx.x] = top_left_dot_product;
        sumReduceShMem(shared_mem[threadIdx.y]);
        top_left_dot_product = shared_mem[threadIdx.y][0];

        shared_mem[threadIdx.y][threadIdx.x] = top_right_dot_product;
        sumReduceShMem(shared_mem[threadIdx.y]);
        top_right_dot_product = shared_mem[threadIdx.y][0];

        shared_mem[threadIdx.y][threadIdx.x] = bottom_left_dot_product;
        sumReduceShMem(shared_mem[threadIdx.y]);
        bottom_left_dot_product = shared_mem[threadIdx.y][0];

        shared_mem[threadIdx.y][threadIdx.x] = bottom_right_dot_product;
        sumReduceShMem(shared_mem[threadIdx.y]);
        bottom_right_dot_product = shared_mem[threadIdx.y][0];

        xf = x_weight_top_left * top_left_dot_product + 
            x_weight_top_left * bottom_left_dot_product -
            (1 - x_weight_top_left) * top_right_dot_product + 
            (1 - x_weight_top_left) * bottom_right_dot_product;

        yf = y_weight_top_left * top_left_dot_product + 
            y_weight_top_left * top_right_dot_product -
            (1 - y_weight_top_left) * bottom_left_dot_product + 
            (1 - y_weight_top_left) * bottom_right_dot_product;

        if(threadIdx.x == 0)
        {
            grid_data[2 * threadIdx.y]     = yf * (input_images_height - 1) / 2;
            grid_data[2 * threadIdx.y + 1] = xf * (input_images_width - 1) / 2;
        }
        
    }       // this huge if is to prevent synchronisation problems

    __syncthreads();
}
