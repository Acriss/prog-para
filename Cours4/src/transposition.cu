__global__ void gpu_transpo_kernel_not_optimal(u_char *Source, u_char *Resultat, unsigned width, unsigned height){
    int j = blockIdx.x*blockDim.x + threadIdx.x;
    int i = blockIdx.y*blockDim.y + threadIdx.y;

    if ((i<0)||(i>=height)||(j<0)||(j>=width)) {}
    else {
        Resultat[j*height + i]  = Source[i*width + j];
    }
}

__global__ void gpu_transpo_kernel_shared(u_char *Source, u_char *Resultat, unsigned width, unsigned height) {
    __shared__ u_char tuile[BLOCKDIM_X][BLOCKDIM_Y+1];
    
    int x = threadIdx.x;
    int y = threadIdx.y;
    int i = blockIdx.y*(BLOCKDIM_Y) + y;
    int j = blockIdx.x*(BLOCKDIM_X) + x;
    

    if ((i<0)||(i>=height)||(j<0)||(j>=width)) {}
    else {            
        tuile[y][x] = Source[i*width + j];
        __syncthreads();
        int i = blockIdx.y*(BLOCKDIM_Y) + x;
        int j = blockIdx.x*(BLOCKDIM_X) + y;
        Resultat[j*height + i] = tuile[x][y];
    }    
}

void cpu_transpo(u_char **Source, u_char **Resultat, unsigned width, unsigned height){
    #pragma omp parallel for num_threads(8)
    for (unsigned i = 0; i < height; i++) {
        for (unsigned j = 0; j < width; j++) {
            Resultat[j][i]  = Source[i][j];
        }
    }
}