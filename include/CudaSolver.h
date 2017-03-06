#ifndef CUDASOLVER_H_
#define CUDASOLVER_H_

#ifdef CUDA_ENABLED
#include <cuda_runtime.h>
#endif /* CUDA_ENABLED */

#include "CpuSolver.h"

namespace solver {
#ifdef CUDA_ENABLED
  class CudaSolver : public CpuSolver {
   public:
    CudaSolver(const int num_vars, const int num_constrs, const int device_id);
    virtual ~CudaSolver();

    virtual bool solve() {
      pre_solve();
      return CpuSolver::solve();
    }

   protected:
    virtual bool find_suitable(const int broken_idx, int &suitable_idx) override;
    virtual void pivot(const int broken_idx, const int suitable_idx) override;
    virtual void swap(const int row, const int col, const int basic_idx,
                      const int nonbasic_idx) override;
    virtual float compute_assignment(const int idx) const override;

   private:
    virtual void pre_solve();

   private:
    const int device_id_;
    cudaDeviceProp prop_;
    float* d_tableau_ = nullptr;
    float* d_tableau_row_ = nullptr;
    float* d_lower_ = nullptr;
    float* d_upper_ = nullptr;
    float* d_assigns_ = nullptr;
    int* d_col_to_var_ = nullptr;
  };
#endif /* CUDA_ENABLED */
}  // namespace solver

#endif /* CUDASOLVER_H_ */
