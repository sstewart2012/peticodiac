cmake_minimum_required(VERSION 2.8)

if (${APPLE})
  set(CMAKE_CXX_COMPILER "g++-6")
else()
  set(CMAKE_CXX_COMPILER "g++")
endif()

project(peticodiac)

option(CUDA "Enable CUDA Support" ON)
option(DEBUG "Enable Debug Support" OFF)

set(CMAKE_BINARY_DIR ${CMAKE_SOURCE_DIR}/bin)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR})
set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR})

include_directories(include)

if (CUDA)
  find_package(CUDA QUIET REQUIRED)
endif()

if (DEBUG)
  set(GCC_COVERAGE_COMPILE_FLAGS "${GCC_COVERAGE_COMPILE_FLAGS} -DDEBUG")
endif()

if (CUDA)
  add_definitions(-std=c++11 -DCUDA_ENABLED=1)
  list(APPEND CUDA_NVCC_FLAGS "-O3;-std=c++11")
  SET(CUDA_PROPAGATE_HOST_FLAGS OFF)
  CUDA_ADD_EXECUTABLE(peticodiac src/main.cpp src/CpuSolver.cpp src/CpuEagerSolver.cpp src/Solver.cpp src/AbstractSolver.cpp src/CudaSolver.cu src/generalSimplex.cu)
else()
  set(GCC_COVERAGE_COMPILE_FLAGS "${GCC_COVERAGE_COMPILE_FLAGS} -Wall -std=c++11 -fopenmp")
  set(GCC_EXE_LINKER_FLAGS "-fopenmp")

  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${GCC_COVERAGE_COMPILE_FLAGS}")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${GCC_EXE_LINKER_FLAGS}")
  add_executable(peticodiac src/main.cpp src/CpuSolver.cpp src/CpuEagerSolver.cpp src/Solver.cpp src/AbstractSolver.cpp src/SolverProfiler.cpp)
endif()
