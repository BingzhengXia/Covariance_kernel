#include "../include/complex_covariance.cuh"
#include <cuComplex.h>
#include <stdio.h>
#include <stdlib.h>



// 复数乘法函数
__device__ cuFloatComplex complexMultiplyConjB(cuFloatComplex a,
                                               cuFloatComplex b) {
  cuFloatComplex result;
  result.x = a.x * b.x + a.y * b.y; // 实部
  result.y = a.y * b.x - a.x * b.y; // 虚部
  return result;
}

// 计算样本均值并将均值减去
__global__ void computeMeanAndSubtract(cuFloatComplex *data, int num_samples,
                                       cuFloatComplex *means) {
  // 使用共享内存
  __shared__ cuFloatComplex shared_means[NUM_ELEMENTS];

  int element_id = threadIdx.x;

  // 第一步：计算均值
  if (element_id < NUM_ELEMENTS) {
    cuFloatComplex sum = make_cuFloatComplex(0.0f, 0.0f);

    for (int i = 0; i < num_samples; i++) {
      cuFloatComplex value = data[i * NUM_ELEMENTS + element_id];
      sum.x += value.x;
      sum.y += value.y;
    }

    shared_means[element_id].x = sum.x / num_samples;
    shared_means[element_id].y = sum.y / num_samples;

    // 可以选择保存均值数组
    if (means != NULL) {
      means[element_id] = shared_means[element_id];
    }
  }

  __syncthreads();

  // 第二步：去均值化
  for (int sample = blockIdx.x; sample < num_samples; sample += gridDim.x) {
    if (element_id < NUM_ELEMENTS) {
      int idx = sample * NUM_ELEMENTS + element_id;
      data[idx].x -= shared_means[element_id].x;
      data[idx].y -= shared_means[element_id].y;
    }
  }
}

// 每个线程块计算一个样本的协方差矩阵
__global__ void sampleCovariance(cuFloatComplex *data, int num_samples,
                                 cuFloatComplex *covariance_matrices) {

  __shared__ cuFloatComplex shared_elements[NUM_ELEMENTS];

  int sample_idx = blockIdx.x;
  int tid = threadIdx.x;

  if (sample_idx < num_samples) {
    // 加载当前样本的所有元素到共享内存
    if (tid < NUM_ELEMENTS) {
      shared_elements[tid] = data[sample_idx * NUM_ELEMENTS + tid];
    }
    __syncthreads();

    // 计算协方差矩阵
    for (int row = 0; row < NUM_ELEMENTS; row++) {
      cuFloatComplex row_val = shared_elements[row];
      if (tid < NUM_ELEMENTS) {
        int col = tid;
        cuFloatComplex col_val = shared_elements[col];

        // 计算row行，col列的协方差值
        cuFloatComplex cov = complexMultiplyConjB(row_val, col_val);

        covariance_matrices[sample_idx * NUM_ELEMENTS * NUM_ELEMENTS +
                            row * NUM_ELEMENTS + col] = cov;
      }

      __syncthreads();
    }
  }
}

void computeComplexCovariance(cuFloatComplex *d_data, int num_samples,
                              cuFloatComplex *d_covariance_matrices) {
  // 分配设备内存
  cuFloatComplex *d_means;
  size_t means_size = NUM_ELEMENTS * sizeof(cuFloatComplex);
  CHECK_CUDA_ERROR(cudaMalloc((void **)&d_means, means_size));

  // 计算均值并计算去均值的样本数据
  int blocksForMean =
      min(32, (num_samples + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
  computeMeanAndSubtract<<<blocksForMean, THREADS_PER_BLOCK>>>(
      d_data, num_samples, d_means);
  CHECK_CUDA_ERROR(cudaGetLastError());
  CHECK_CUDA_ERROR(cudaDeviceSynchronize());

  // 计算每个样本的协方差矩阵
  sampleCovariance<<<num_samples, THREADS_PER_BLOCK>>>(d_data, num_samples,
                                                       d_covariance_matrices);
  CHECK_CUDA_ERROR(cudaGetLastError());
  CHECK_CUDA_ERROR(cudaDeviceSynchronize());
  CHECK_CUDA_ERROR(cudaFree(d_means));
}