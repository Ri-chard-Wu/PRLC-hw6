// #include <iostream>
// #include <random>

// #define N 1024
// typedef unsigned int WORD;
// typedef unsigned char BYTE;
// using namespace std;

// // double min(double a, double b) {return a < b ? a : b;}

// double CPU_reduction(double *arr, int n) {

//     double ret = arr[0];

//     for (int i = 1; i < n; i++) {
//         ret = min(ret, arr[i]);
//     }

//     return ret;
// }


// double fRand(double fMin, double fMax)
// {
//     double f = (double)rand() / RAND_MAX;
//     return fMin + f * (fMax - fMin);
// }


// void generate_random_doubles(double *arr, int n){
//     for(int i = 0; i < n; i++){
//         arr[i] = fRand(0., 10000.);
//     }
// }

// // linear addressing.

// __global__ void cuda_reduction(double *arr, int n, double *ret) {

//     unsigned int tid = threadIdx.x;

//     __shared__ WORD sm[N * 2];
//     double *sm_double = (double *)sm;

//     // __shared__ double sm_double[N];
    
    
//     sm_double[tid] = arr[tid];

    
//     __syncthreads();

//     for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
//         if (tid < s) {
//             sm_double[tid] = min(sm_double[tid], sm_double[tid + s]);
//         }
//         __syncthreads();
//     }
//     if (tid == 0) *ret = sm_double[0];
// }



// int main() {

//     // srand(time(0));

//     double *ret = new double;
//     double *arr = new double[N];
    
//     generate_random_doubles(arr, N);

//     std::cout << "[main] Generated numbers:";
//     for (int i = 0; i < N; i++) {
//         std::cout << ' ' << arr[i];
//     }
//     std::cout << '\n';


//     double *arr_dev, *ret_dev;
//     cudaMalloc(&arr_dev, N * sizeof(double));
//     cudaMalloc(&ret_dev, 1 * sizeof(double));
//     cudaMemcpy((BYTE *)arr_dev, (BYTE *)arr,
//                             N * sizeof(double), cudaMemcpyHostToDevice);
                                
//     cuda_reduction<<<1, N>>>(arr_dev, N, ret_dev);

//     cudaDeviceSynchronize();

//     cudaMemcpy((BYTE *)ret, (BYTE *)ret_dev,
//                             1 * sizeof(double), cudaMemcpyDeviceToHost);

//     std::cout << "[main] (cuda) The minimum value: " << *ret << '\n';

//     *ret = CPU_reduction(arr, N);

//     std::cout << "[main] (cpu) The minimum value: " << *ret << '\n';
    
//     delete ret;
//     delete [] arr;
//     return 0;
// }









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
        arr[i] = fRand(0., 50000.);
    }
}


__device__ __inline__ void write_double_sm(unsigned int tid, double val_g, WORD *sm){

    unsigned int warp_tid = tid % 32;
    unsigned int warp_id = tid / 32;

    double val = val_g;
    WORD *val_word_array = (WORD *)&val;

    sm[(warp_id * 2 + 0) * 32 + warp_tid] = val_word_array[0];
    sm[(warp_id * 2 + 1) * 32 + warp_tid] = val_word_array[1];
}


__device__ __inline__ double read_double_sm(unsigned int tid, WORD *sm){

    unsigned int warp_tid = tid % 32;
    unsigned int warp_id = tid / 32;

    double val;
    WORD *val_word_array = (WORD *)&val;

    val_word_array[0] = sm[(warp_id * 2 + 0) * 32 + warp_tid];
    val_word_array[1] = sm[(warp_id * 2 + 1) * 32 + warp_tid];
    return val;   
}


__global__ void cuda_reduction(double *arr, int n, double *ret) {
   
    unsigned int tid = threadIdx.x;
    __shared__ WORD sm[N * 2];

    double val1, val2;
    val1 = arr[tid];
    write_double_sm(tid, val1, sm);
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        
        if (tid < s) {
            val2 = read_double_sm(tid + s, sm);
            val1 = min(val1, val2);
            write_double_sm(tid, val1, sm);
        }
        __syncthreads();
    }

    if (tid == 0) *ret = read_double_sm(tid, sm);  
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