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


typedef unsigned long long int ull_t;


__global__ void cuda_reduction(ull_t *arr, int n, ull_t *ret) {
   
    unsigned int tid = threadIdx.x;

    ull_t item_local = arr[tid];
    __syncthreads();

    atomicMin(ret, item_local);
}




int main() {

    ull_t *ret_ull = new ull_t;
    double *ret_double = new double;
    double *arr = new double[N];
    ull_t *arr_ull = new ull_t[N];
    
    generate_random_doubles(arr, N);

    std::cout << "[main] Generated numbers:";
    for (int i = 0; i < N; i++) {
        std::cout << ' ' << arr[i];
    }
    std::cout << '\n';


    ull_t mask = ((ull_t)1) << 63; 

    for(int i = 0;i < N; i++){
        arr_ull[i] = ((ull_t)(arr[i])) ^ mask;
    }

    ull_t *ret_ull_dev;
    ull_t *arr_ull_dev;
    cudaMalloc(&arr_ull_dev, N * sizeof(ull_t));
    cudaMalloc(&ret_ull_dev, 1 * sizeof(ull_t));
    cudaMemcpy((BYTE *)arr_ull_dev, (BYTE *)arr_ull, N * sizeof(ull_t), cudaMemcpyHostToDevice);
    cudaMemcpy((BYTE *)ret_ull_dev, (BYTE *)arr_ull, 1 * sizeof(ull_t), cudaMemcpyHostToDevice);
                                   
    cuda_reduction<<<1, N>>>(arr_ull_dev, N, ret_ull_dev);

    cudaDeviceSynchronize();

    cudaMemcpy((BYTE *)ret_ull, (BYTE *)ret_ull_dev, 1 * sizeof(ull_t), cudaMemcpyDeviceToHost);
    std::cout << "[main] (cuda) The minimum value: " << ((double)((*ret_ull) ^ mask)) << '\n';

    *ret_double = CPU_reduction(arr, N);
    std::cout << "[main] (cpu) The minimum value: " << *ret_double << '\n';
    
    delete ret_ull;
    delete ret_double;
    delete [] arr;
    return 0;
}
