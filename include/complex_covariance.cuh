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

// 定义常量：复数单元数量
constexpr int NUM_ELEMENTS = 112;
constexpr int THREADS_PER_BLOCK = 128;
constexpr int MAX_BATCH_SIZE = 8;  // 一次处理的批次大小

// 合并计算均值和去均值的函数
__global__ void computeMeanAndSubtract(cuFloatComplex* data, int num_samples, cuFloatComplex* means);

// 每个线程块计算一个样本的协方差矩阵
__global__ void sampleCovariance(cuFloatComplex* data, int num_samples, cuFloatComplex* covariance_matrices);

// 主计算函数
void computeComplexCovariance(cuFloatComplex* data, int num_samples, cuFloatComplex* covariance_matrices);

#endif // COMPLEX_COVARIANCE_CUH 