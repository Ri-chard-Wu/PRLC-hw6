#include <iostream>
#include <random>

#define N 1024

__global__ void cuda_reduction() {
    unsigned int tid = threadIdx.x;
    int tid_other = __shfl_down(tid, 2, 32);
}


int main() {             
    cuda_reduction<<<1, N>>>();
    cudaDeviceSynchronize();
    return 0;
}