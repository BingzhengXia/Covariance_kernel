#include "../include/complex_covariance.cuh"
#include <cuComplex.h>
#include <stdio.h>
#include <stdlib.h>
// 计算示例
int main() {
  // 设定样本数，每次计算一个样本批次的协方差矩阵，首先计算多个样本的均值，让后每个样本减去均值，然后计算协方差矩阵
  size_t num_samples = 10;

  cuFloatComplex *h_data = (cuFloatComplex *)malloc(num_samples * NUM_ELEMENTS *
                                                    sizeof(cuFloatComplex));
  cuFloatComplex *h_covariance_matrices = (cuFloatComplex *)malloc(
      num_samples * NUM_ELEMENTS * NUM_ELEMENTS * sizeof(cuFloatComplex));

  if (!h_data || !h_covariance_matrices) {
    fprintf(stderr, "主机内存分配失败\n");
    exit(EXIT_FAILURE);
  }

  // 初始化随机复数数据
  for (int i = 0; i < num_samples * NUM_ELEMENTS; i++) {
    h_data[i].x = (float)rand() / RAND_MAX * 2.0f - 1.0f;
    h_data[i].y = (float)rand() / RAND_MAX * 2.0f - 1.0f;
  }

  cuFloatComplex *d_data, *d_covariance_matrices;
  size_t data_size = num_samples * NUM_ELEMENTS * sizeof(cuFloatComplex);
  size_t covariance_size =
      num_samples * NUM_ELEMENTS * NUM_ELEMENTS * sizeof(cuFloatComplex);

  CHECK_CUDA_ERROR(cudaMalloc((void **)&d_data, data_size));
  CHECK_CUDA_ERROR(
      cudaMalloc((void **)&d_covariance_matrices, covariance_size));

  CHECK_CUDA_ERROR(
      cudaMemcpy(d_data, h_data, data_size, cudaMemcpyHostToDevice));

  // 调用设备指针版本的协方差计算函数
  computeComplexCovariance(d_data, num_samples, d_covariance_matrices);

  CHECK_CUDA_ERROR(cudaMemcpy(h_covariance_matrices, d_covariance_matrices,
                              covariance_size, cudaMemcpyDeviceToHost));

  // 打印部分结果用来验证
  printf("第一个样本协方差矩阵的前3x3部分:\n");
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      printf("(%f, %f) ",
             h_covariance_matrices[0 * NUM_ELEMENTS * NUM_ELEMENTS +
                                   i * NUM_ELEMENTS + j]
                 .x,
             h_covariance_matrices[0 * NUM_ELEMENTS * NUM_ELEMENTS +
                                   i * NUM_ELEMENTS + j]
                 .y);
    }
    printf("\n");
  }

  if (num_samples > 1) {
    printf("\n第二个样本协方差矩阵的前3x3部分:\n");
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 3; j++) {
        printf("(%f, %f) ",
               h_covariance_matrices[1 * NUM_ELEMENTS * NUM_ELEMENTS +
                                     i * NUM_ELEMENTS + j]
                   .x,
               h_covariance_matrices[1 * NUM_ELEMENTS * NUM_ELEMENTS +
                                     i * NUM_ELEMENTS + j]
                   .y);
      }
      printf("\n");
    }
  }
  
  free(h_data);
  free(h_covariance_matrices);
  CHECK_CUDA_ERROR(cudaFree(d_data));
  CHECK_CUDA_ERROR(cudaFree(d_covariance_matrices));

  return 0;
}