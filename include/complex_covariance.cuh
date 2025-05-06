#ifndef COMPLEX_COVARIANCE_CUH
#define COMPLEX_COVARIANCE_CUH

#include <cuda_runtime.h>
#include <cuComplex.h>
// check cuda error
#define CHECK_CUDA_ERROR(call)                                                 \
  do {                                                                         \
    cudaError_t error = call;                                                  \
    if (error != cudaSuccess) {                                                \
      fprintf(stderr, "CUDA error: %s:%d, %s\n", __FILE__, __LINE__,           \
              cudaGetErrorString(error));                                      \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

// 每个样本元素的数量
constexpr int NUM_ELEMENTS = 112;
constexpr int THREADS_PER_BLOCK = 128;
// constexpr int MAX_BATCH_SIZE = 8;  

// 合并计算均值和去均值的函数
__global__ void computeMeanAndSubtract(cuFloatComplex* data, int num_samples, cuFloatComplex* means);

// 计算协方差，输出值以行优先的方式存储的一维数组中  112*112*样本数
__global__ void sampleCovariance(cuFloatComplex* data, int num_samples, cuFloatComplex* covariance_matrices);

// 包含上述两个核函数，用于调用的函数 data和covariance_matrices分别是输入数据和输出数据（存储在显存中），num_samples是样本数
void computeComplexCovariance(cuFloatComplex* data, int num_samples, cuFloatComplex* covariance_matrices);

#endif 