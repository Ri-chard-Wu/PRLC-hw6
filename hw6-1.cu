#include <iostream>
#include <random>

#define N 1024
typedef unsigned int WORD;
typedef unsigned char BYTE;
using namespace std;

// double min(double a, double b) {return a < b ? a : b;}

double CPU_reduction(double *arr, int n) {

    double ret = arr[0];

    for (int i = 1; i < n; i++) {
        ret = min(ret, arr[i]);
    }

    return ret;
}


double fRand(double fMin, double fMax)
{
    double f = (double)rand() / RAND_MAX;
    return fMin + f * (fMax - fMin);
}


void generate_random_doubles(double *arr, int n){
    for(int i = 0; i < n; i++){
        arr[i] = fRand(0., 10000.);
    }
}



__global__ void cuda_reduction(double *arr, int n, double *ret) {
   
    unsigned int tid = threadIdx.x;

    __shared__ WORD sm[N];
    double *sm_double = (double *)sm;

    sm_double[tid] = arr[tid];
    __syncthreads();

    for(unsigned int s = 1; s < blockDim.x; s *= 2) {
        
        if (tid % (2 * s) == 0) {
            sm_double[tid] = min(sm_double[tid], sm_double[tid + s]);
        }
        __syncthreads();
    }


    if (tid == 0) *ret = sm_double[0];
}



int main() {

    double *ret = new double;
    double *arr = new double[N];
    
    generate_random_doubles(arr, N);

    std::cout << "Generated numbers:";
    for (int i = 0; i < N; i++) {
        std::cout << ' ' << arr[i];
    }
    std::cout << '\n';


    double *arr_dev, *ret_dev;
    cudaMalloc(&arr_dev, N * sizeof(double));
    cudaMalloc(&ret_dev, 1 * sizeof(double));
    cudaMemcpy((BYTE *)arr_dev, (BYTE *)arr,
                            N * sizeof(double), cudaMemcpyHostToDevice);
                                
    cuda_reduction<<<1, N>>>(arr_dev, N, ret_dev);

    cudaDeviceSynchronize();

    cudaMemcpy((BYTE *)ret, (BYTE *)ret_dev,
                            1 * sizeof(double), cudaMemcpyDeviceToHost);

    std::cout << "(cuda) The minimum value: " << *ret << '\n';

    *ret = CPU_reduction(arr, N);

    std::cout << "(cpu) The minimum value: " << *ret << '\n';
    
    delete ret;
    delete [] arr;
    return 0;
}

//fefe
