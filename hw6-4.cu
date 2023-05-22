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



// __device__ inline
// double __shfl_down_double(double var, unsigned int srcLane, int width=32) {
//   int2 a = *reinterpret_cast<int2*>(&var);
//   a.x = __shfl_down(a.x, srcLane, width);
//   a.y = __shfl_down(a.y, srcLane, width);
//   return *reinterpret_cast<double*>(&a);
// }




__inline__ __device__
double warpReduceSum(double val) {

    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {

        double val_other = __shfl_down(val, offset);
        val = min(val, val_other);

    }

    return val;
}





__inline__ __device__
double blockReduceSum(double val) {

    __shared__ double shared[32]; 
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;

    val = warpReduceSum(val);    

    if (lane == 0) shared[wid] = val; 

    __syncthreads();             



    val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 1000000.;

    if (wid==0) val = warpReduceSum(val); 

    return val;
}



__global__ void cuda_reduction(double *arr, int n, double *ret) {

    unsigned int tid = threadIdx.x;
    double val = arr[tid];
    
    __syncthreads();

    val = blockReduceSum(val);

    if (tid == 0) *ret = val;
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