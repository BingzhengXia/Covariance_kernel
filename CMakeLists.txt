cmake_minimum_required(VERSION 3.10)
project(ComplexCovariance CUDA CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_ARCHITECTURES 86 89)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O2")

find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(include)

add_executable(complex_covariance examples/complex_cov.cu src/complex_covariance.cu)

set_target_properties(complex_covariance PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# target_link_libraries(complex_covariance ${CUDA_LIBRARIES})
