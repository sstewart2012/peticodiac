cmake_minimum_required(VERSION 2.8)
project(peticodiac)
find_package(CUDA QUIET REQUIRED)
add_definitions(-std=c++11)
list(APPEND CUDA_NVCC_FLAGS "-O3;-std=c++11")
SET(CUDA_PROPAGATE_HOST_FLAGS OFF)
include_directories(include)
CUDA_ADD_EXECUTABLE(peticodiac src/main.cpp src/CpuSolver.cpp src/CudaSolver.cu src/generalSimplex.cu src/Solver.cpp)


