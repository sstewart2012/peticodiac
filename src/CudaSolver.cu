#include <cassert>
#include <cuda_runtime.h>
#include "CudaSolver.h"
#include "cudaHelper.h"
#include "generalSimplex.h"

#include <thrust/device_vector.h>
#include <thrust/transform_reduce.h>

namespace solver {

CudaSolver::CudaSolver(const int num_vars, const int num_constrs,
                       const int device_id)
    : CpuSolver(num_vars, num_constrs),
      device_id_(device_id) {
  // Device initializations
  CHECK(cudaSetDevice(0));
  CHECK(cudaGetDeviceProperties(&prop_, device_id));
  const size_t sz_bounds = (ncols_ + nrows_) * sizeof(float);
  memalloc(&d_lower_, sz_bounds);
  memalloc(&d_upper_, sz_bounds);
  memalloc(&d_assigns_, sz_bounds);
  memalloc(&d_tableau_, nrows_ * ncols_ * sizeof(float));
  memalloc(&d_tableau_row_, ncols_ * sizeof(float));
  memalloc(&d_col_to_var_, ncols_ * sizeof(int));
  CHECK(cudaDeviceSynchronize());
}

CudaSolver::~CudaSolver() {
  memfree(d_tableau_);
  memfree(d_tableau_row_);
  memfree(d_lower_);
  memfree(d_upper_);
  memfree(d_assigns_);
  memfree(d_col_to_var_);
  cudaDeviceReset();
}

void CudaSolver::pre_solve() {
  const size_t sz_bounds = (ncols_ + nrows_) * sizeof(float);
  memcpyH2D(d_tableau_, tableau_, nrows_ * ncols_ * sizeof(float));
  memcpyH2D(d_col_to_var_, col_to_var_, ncols_ * sizeof(int));
  memcpyH2D(d_upper_, upper_, sz_bounds);
  memcpyH2D(d_assigns_, assigns_, sz_bounds);
  memcpyH2D(d_lower_, lower_, sz_bounds);

}

bool CudaSolver::find_suitable(const int broken_idx, int &suitable_idx) {
  const size_t offset = OFFSET(var_to_tableau_[broken_idx], 0, ncols_);
  memcpyD2H(&tableau_[offset], &d_tableau_[offset], ncols_ * sizeof(float));
  bool result = CpuSolver::find_suitable(broken_idx, suitable_idx);

  // Copy updated assignments to device
  if (result == true) {
    memcpyH2D(&d_assigns_[broken_idx], &assigns_[broken_idx], sizeof(float));
    memcpyH2D(&d_assigns_[suitable_idx], &assigns_[suitable_idx],
              sizeof(float));
  }
  return result;
}

void CudaSolver::pivot(const int broken_idx, const int suitable_idx) {
  const int pivot_row = var_to_tableau_[broken_idx];
  const int pivot_col = var_to_tableau_[suitable_idx];
  //printf("Pivot(%d,%d)\n", pivot_row, pivot_col);

  // Save the current pivot element (alpha)
  const int alpha_idx = OFFSET(pivot_row, pivot_col, ncols_);
  float alpha = tableau_[alpha_idx];

  // Kernel configurations
  const dim3 block_inner(32, 32, 1);
  const dim3 grid_inner((nrows_ + 31) / 32, (ncols_ + 31) / 32, 1);
  const dim3 block_row(256, 1, 1);
  const dim3 grid_row((ncols_ + 255) / 256, 1, 1);
  const dim3 block_column(256, 1, 1);
  const dim3 grid_column((ncols_ + 255) / 256, 1, 1);

  // Update the tableau_
  device::pivot_update_inner<<<grid_inner, block_inner>>>(alpha, pivot_row,
      pivot_col, nrows_, ncols_, d_tableau_);
  device::pivot_update_row<<<grid_row, block_row>>>(alpha, pivot_row, ncols_,
      d_tableau_);
  device::pivot_update_column<<<grid_column, block_column>>>(alpha, pivot_col,
      nrows_, ncols_, d_tableau_);

  // Update pivot element on the device
  alpha = 1.0f / alpha;
  memcpyH2D(&d_tableau_[alpha_idx], &alpha, sizeof(float));

  // Swap the basic_ and nonbasic_ variables
  swap(pivot_row, pivot_col, broken_idx, suitable_idx);
}

void CudaSolver::swap(const int row, const int col, const int basic_idx,
                      const int nonbasic_idx) {
  CpuSolver::swap(row, col, basic_idx, nonbasic_idx);

  // Update column to variable mapping on device
  memcpyH2D(&d_col_to_var_[col], &col_to_var_[col], sizeof(int));

}

float CudaSolver::compute_assignment(const int idx) const {
  // D2D copy of the current tableau_ row
  const int rowIdx = var_to_tableau_[idx];
  const size_t offset = rowIdx * ncols_;
  memcpyD2D(d_tableau_row_, &d_tableau_[offset], ncols_ * sizeof(float));

  // Run kernel that multiples each element of row by its respective variable assignment
  const int n = ncols_;
  const int block_size = 256;
  const int nblocks = (n + block_size - 1) / block_size;
  device::update_assignment_row_multiply<<<nblocks, block_size>>>(ncols_,
      d_col_to_var_, d_assigns_, d_tableau_row_);

  // Perform a sum reduction of the row
  thrust::device_ptr<float> dev_ptr = thrust::device_pointer_cast(
      d_tableau_row_);
  thrust::device_vector<float> vec(dev_ptr, dev_ptr + ncols_);
  float val = thrust::reduce(vec.begin(), vec.begin() + ncols_);

  // Update the assignment
  const_cast<CudaSolver*>(this)->assigns_[idx] = val;
  const_cast<CudaSolver*>(this)->map_assigns_[idx] = get_step_count();

  return val;
}

}
