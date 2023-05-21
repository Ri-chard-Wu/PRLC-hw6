#include <iostream>
#include <random>

#define N 32
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
        arr[i] = fRand(0., 50000.);
    }
}



__global__ void cuda_reduction(double *arr, int n, double *ret) {
   
    unsigned int tid = threadIdx.x;

    unsigned int warp_size = 32;
    unsigned int warp_tid = tid % warp_size;
    unsigned int warp_id = tid / warp_size;


    __shared__ WORD sm[N * 2];
    // double *sm_double = (double *)sm;

    double val = arr[tid];
    WORD *val_word_array = (WORD *)&val;

    sm[(warp_id * 2 + 0) * warp_size + warp_tid] = val_word_array[0];
    sm[(warp_id * 2 + 1) * warp_size + warp_tid] = val_word_array[1];


    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        
        if (tid < s) {
            
            val_word_array[0] = sm[(warp_id * 2 + 0) * warp_size + warp_tid];
            val_word_array[1] = sm[(warp_id * 2 + 1) * warp_size + warp_tid];
            double val1 = val;

            unsigned int tid_other = tid + s;
            unsigned int warp_tid_other = tid_other % warp_size;
            unsigned int warp_id_other = tid_other / warp_size;

            val_word_array[0] = sm[(warp_id_other * 2 + 0) * warp_size + warp_tid_other];
            val_word_array[1] = sm[(warp_id_other * 2 + 1) * warp_size + warp_tid_other];
            double val2 = val;

            val = min(val1, val2);
            sm[(warp_id * 2 + 0) * warp_size + warp_tid] = val_word_array[0];
            sm[(warp_id * 2 + 1) * warp_size + warp_tid] = val_word_array[1];            
        }
        __syncthreads();
    }

    if (tid == 0){
        val_word_array[0] = sm[(warp_id * 2 + 0) * warp_size + warp_tid];
        val_word_array[1] = sm[(warp_id * 2 + 1) * warp_size + warp_tid];
        *ret = val;      
    }

    // if (tid == 0) *ret = sm_double[0];
}



int main() {

    srand(time(0));

    double *ret = new double;
    double *arr = new double[N];
    
    generate_random_doubles(arr, N);

    std::cout << "[main] Generated numbers:";
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

    std::cout << "[main] (cuda) The minimum value: " << *ret << '\n';

    *ret = CPU_reduction(arr, N);

    std::cout << "[main] (cpu) The minimum value: " << *ret << '\n';
    
    delete ret;
    delete [] arr;
    return 0;
}
