#include "Solver.h"
#include "CpuSolver.h"
#include "CpuEagerSolver.h"

#ifdef CUDA_ENABLED
#include "CudaSolver.h"
#endif

namespace solver {

// Implementation of factory method
template <typename T>
Solver<T>* Solver<T>::create(SolverType type, const int num_vars, const int max_num_constrs) {
  if (type == SolverType::CPU_LAZY) {
    printf("CPU_LAZY\n");
    return new CpuSolver<T>(num_vars, max_num_constrs);
  } else if (type == SolverType::CPU_EAGER) {
    printf("CPU_EAGER\n");
    return new CpuEagerSolver<T>(num_vars, max_num_constrs);
  } else if (type == SolverType::CUDA) {
#ifdef CUDA_ENABLED
    printf("CUDA\n");
    return new CudaSolver(num_vars, max_num_constrs, 0);
#else
    printf("CUDA not supported\n");
#endif
  }

  return nullptr;
}

template class Solver<float>;
template class Solver<Fraction>;

} // namespace solver
