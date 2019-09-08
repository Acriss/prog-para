__global__ void gpu_histo_kernel_shared(u_char* Source, int *res, unsigned height, unsigned width){
    __shared__ int hist[256];

    int j = blockIdx.x*blockDim.x + threadIdx.x;
    int i = blockIdx.y*blockDim.y + threadIdx.y;

    int index = threadIdx.x * BLOCKDIM_X + threadIdx.y;

    if( index < 256) {
        hist[index] = 0;
    }
    __syncthreads();


    if ((i<0)||(i>=height) || (j<0) || (j>=width)) {}
    else {
        atomicAdd(&hist[Source[i*width+j]], 1);
        __syncthreads();
        if( index < 256)
            atomicAdd(&res[index], hist[index]);
    }
}

void cpu_histo(u_char** Source, int (*res)[256], unsigned height, unsigned width){
    #pragma omp parallel for num_threads(8)
    for( int i = 0; i < height; i++){
        for( int j = 0; j < width; j++){
            #pragma omp atomic
            (*res)[Source[i][j]]++;
        }
    }
}
