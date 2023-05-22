#include <iostream>
#include <random>
#include <cstring>

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



__device__ double atomicAdd_double(double* address, double val)
{
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old_value = *address_as_ull, assumed;

    do {
        assumed = old_value;
        old_value = atomicCAS(address_as_ull, assumed,
                 __double_as_longlong(val + __longlong_as_double(assumed)));

    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old_value);

    return __longlong_as_double(old_value);
}



__device__ double atomicMin_double(double *address, double val)
{
    ull_t *address_as_ull = (unsigned long long int*)address;
    ull_t assumed, new_value;
    ull_t old_value = *address_as_ull;
    do {
        assumed = old_value;
        if(val < __longlong_as_double(old_value)){
            new_value = __double_as_longlong(val);
        }
        else{
            new_value = old_value;
        }
        old_value = atomicCAS(address_as_ull, assumed, new_value);
                                
    // Note: uses integer comparison to avoid
            // hang in case of NaN (since NaN != NaN)
    } while (assumed != old_value);
    return __longlong_as_double(old_value);
}


__global__ void cuda_reduction(double *arr, int n, double *ret) {
   
    unsigned int tid = threadIdx.x;
    double item_local = arr[tid];
    atomicMin_double(ret, item_local);
}






int main() {

    // srand(time(0));


    double *ret = new double;
    double *arr = new double[N];
    
    generate_random_doubles(arr, N);

    std::cout << "[main] Generated numbers:";
    for (int i = 0; i < N; i++) {
        std::cout << ' ' << arr[i];
    }
    std::cout << '\n';


    double *ret_dev;
    double *arr_dev;
    cudaMalloc(&arr_dev, N * sizeof(double));
    cudaMalloc(&ret_dev, 1 * sizeof(double));
    cudaMemcpy((BYTE *)arr_dev, (BYTE *)arr, N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy((BYTE *)ret_dev, (BYTE *)arr, 1 * sizeof(double), cudaMemcpyHostToDevice);
                                   
    cuda_reduction<<<1, N>>>(arr_dev, N, ret_dev);

    cudaDeviceSynchronize();

    cudaMemcpy((BYTE *)ret, (BYTE *)ret_dev, 1 * sizeof(double), cudaMemcpyDeviceToHost);
    std::cout << "[main] (cuda) The minimum value: " << *ret << '\n';

    *ret= CPU_reduction(arr, N);
    std::cout << "[main] (cpu) The minimum value: " << *ret << '\n';
    
    delete ret;
    delete [] arr;
    return 0;
}
