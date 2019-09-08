#include <stdio.h>
#include <iostream>
#include <chrono>
#include <float.h>
#include "cuda_runtime.h"
#include "cuda.h"
#include "omp.h"
#include "device_launch_parameters.h"

using namespace std;

__global__ void gpu_func(int n, float *x, float *y, float *s)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < n) atomicAdd(s, x[i] * y[i]);
}

void cpu(int n, float *x, float *y, float *s)
{
  omp_set_num_threads(8);
  #pragma omp parallel
  {
      float local_result=0;
      #pragma omp for schedule(static)
      for (unsigned long int i = 0; i < n; i++) {
          local_result += x[i] * y[i];
      }
      #pragma omp critical
          s[0] += local_result;
  }
}

void cpu_mono(int n, float *x, float *y, float *s)
{
  s[0] = 0;
  for (unsigned long int i = 0; i < n; i++) {
     s[0] += x[i] * y[i];
  }
}


int main(void)
{
  // unsigned long int N = 1024*1024;
  float *x, *y, *d_x, *d_y, *d_s, s_gpu=0, s_cpu=0, s_mono=0;
  

  for (unsigned long int N = 1024; N < 4096*4096+1; N*=4) {
    printf("Size: %lu\n", N);
    x = (float*)malloc(N*sizeof(float));
    y = (float*)malloc(N*sizeof(float));
    // s_gpu = (float*)malloc(sizeof(float));
    // s_cpu = (float*)malloc(sizeof(float));
    // s_mono = (float*)malloc(sizeof(float));
    // s_cpu[0] = 0;
    // s_gpu[0] = 0;
    // s_mono[0] = 0;


    for (int i = 0; i < N; i++) {
      x[i] = 1.0f;
      y[i] = 2.0f;
    }
    cudaMalloc(&d_x, N*sizeof(float)); 
    cudaMalloc(&d_y, N*sizeof(float));
    cudaMalloc(&d_s, sizeof(float));
    cudaMemcpy(d_x, x, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_s, &s_gpu, sizeof(float), cudaMemcpyHostToDevice);



    std::chrono::high_resolution_clock::time_point t0 = std::chrono::high_resolution_clock::now();
    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
    double min_duration = DBL_MAX;
    for (auto it =0; it < 1000; it++) {
        s_mono = 0;
        t0 = std::chrono::high_resolution_clock::now();
        cpu_mono(N,x,y,&s_mono);
        t1 = std::chrono::high_resolution_clock::now();
        double duration = std::chrono::duration<double>(t1-t0).count();
        if (duration < min_duration) min_duration = duration;
    }
    std::cout <<  "MONO 0 " << min_duration << " " << (min_duration/N) << std::endl;

    for (int i=0; i < 11; i++) {
      int k = pow(2, i);
      t0 = std::chrono::high_resolution_clock::now();
      t1 = std::chrono::high_resolution_clock::now();
      min_duration = DBL_MAX;
      for (auto it =0; it < 1000; it++) {
          s_gpu = 0;
          cudaMemcpy(d_s, &s_gpu, sizeof(float), cudaMemcpyHostToDevice);
          t0 = std::chrono::high_resolution_clock::now();
          gpu_func<<<(N+k)/k, k>>>(N, d_x, d_y, d_s);
          cudaDeviceSynchronize();
          cudaMemcpy(&s_gpu, d_s, sizeof(float), cudaMemcpyDeviceToHost);
          t1 = std::chrono::high_resolution_clock::now();
          double duration = std::chrono::duration<double>(t1-t0).count();
          if (duration < min_duration) min_duration = duration;
      }
    
      std::cout <<  "GPU " << k << " " << min_duration << " " << (min_duration/N) << std::endl;
    }


    
    t0 = std::chrono::high_resolution_clock::now();
    t1 = std::chrono::high_resolution_clock::now();
    min_duration = DBL_MAX;
    for (auto it =0; it < 1000; it++) {
        s_cpu = 0;
        t0 = std::chrono::high_resolution_clock::now();
        cpu(N, x, y, &s_cpu);
        t1 = std::chrono::high_resolution_clock::now();
        double duration = std::chrono::duration<double>(t1-t0).count();
        if (duration < min_duration) min_duration = duration;
    }
    std::cout <<  "CPU 0 " << min_duration << " " << (min_duration/N) << std::endl;

    float maxError = 0.0f;
    // std::cout << s_gpu << std::endl;
    // std::cout << s_cpu << std::endl;
    // std::cout << s_mono << std::endl;
    maxError = pow(s_gpu - s_cpu, 2) + pow(s_gpu - s_mono, 2);
    printf("Max error: %f\n", maxError);

    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_s);
    free(x);
    free(y);
    // free(s_cpu);
    // free(s_gpu);
    // free(s_mono);
  }
  
}

